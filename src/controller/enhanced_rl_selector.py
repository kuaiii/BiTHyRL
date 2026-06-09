# -*- coding: utf-8 -*-
"""
增强型 RL 控制器选择器

针对控制器选择任务的深度强化学习增强方案：
1. 图神经网络（GNN）策略 - 更好地捕捉图结构
2. Actor-Critic 架构 - 降低方差
3. 多目标奖励 - 综合优化多个指标
4. 课程学习 - 从简单到复杂
5. 元学习 - 快速适应新图
"""
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import networkx as nx
import numpy as np
from tqdm import tqdm
from collections import deque

from src.utils.logger import get_logger

logger = get_logger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. 增强特征提取
# ============================================================

class GraphFeatureExtractor:
    """
    图特征提取器：支持多种特征类型
    """
    
    @staticmethod
    def extract_node_features(G, feature_type='enhanced'):
        """
        提取节点特征
        
        Args:
            G: NetworkX图
            feature_type: 特征类型
                - 'basic': 5维基础特征
                - 'enhanced': 12维增强特征
                - 'structural': 15维结构特征
        """
        num_nodes = G.number_of_nodes()
        if num_nodes == 0:
            return torch.zeros((0, 5)), []
        
        node_list = list(G.nodes())
        
        if feature_type == 'basic':
            return GraphFeatureExtractor._extract_basic_features(G, node_list)
        elif feature_type == 'enhanced':
            return GraphFeatureExtractor._extract_enhanced_features(G, node_list)
        elif feature_type == 'structural':
            return GraphFeatureExtractor._extract_structural_features(G, node_list)
        else:
            return GraphFeatureExtractor._extract_basic_features(G, node_list)
    
    @staticmethod
    def _extract_basic_features(G, node_list):
        """5维基础特征"""
        num_nodes = len(node_list)
        device = DEVICE
        
        adj = nx.to_scipy_sparse_array(G, format='coo', dtype=np.float32)
        indices = torch.LongTensor(np.vstack((adj.row, adj.col))).to(device)
        values = torch.FloatTensor(adj.data).to(device)
        adj_tensor = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).to(device)
        
        # 度数
        degrees = torch.sparse.sum(adj_tensor, dim=1).to_dense()
        deg_norm = degrees / (num_nodes - 1 + 1e-6)
        
        # 聚类系数
        if num_nodes < 5000:
            adj_dense = adj_tensor.to_dense()
            adj2 = torch.mm(adj_dense, adj_dense)
            adj3 = torch.mm(adj2, adj_dense)
            triangles = torch.diagonal(adj3)
            potential = degrees * (degrees - 1)
            clust = triangles / (potential + 1e-6)
        else:
            cc = nx.clustering(G)
            clust = torch.tensor([cc[n] for n in node_list], dtype=torch.float32).to(device)
        
        # PageRank
        deg_inv = 1.0 / (degrees + 1e-6)
        norm_values = values * deg_inv[indices[1]]
        trans = torch.sparse_coo_tensor(indices, norm_values, (num_nodes, num_nodes)).to(device)
        pr = torch.ones(num_nodes, 1).to(device) / num_nodes
        for _ in range(10):
            pr = 0.85 * torch.sparse.mm(trans, pr) + 0.15 / num_nodes
        pr = pr.squeeze()
        
        # 特征向量中心性
        eig = torch.ones(num_nodes, 1).to(device)
        for _ in range(10):
            eig = torch.sparse.mm(adj_tensor, eig)
            n = torch.norm(eig)
            if n > 0:
                eig = eig / n
        eig = eig.squeeze()
        
        # Katz
        if num_nodes < 5000:
            katz = torch.sum(torch.mm(adj_dense, adj_dense), dim=1)
            katz = katz / (torch.max(katz) + 1e-6)
        else:
            katz = degrees / (torch.max(degrees) + 1e-6)
        
        # 确保维度正确
        if pr.dim() == 0: pr = pr.unsqueeze(0)
        if eig.dim() == 0: eig = eig.unsqueeze(0)
        if num_nodes == 1:
            if deg_norm.dim() == 0: deg_norm = deg_norm.unsqueeze(0)
            if clust.dim() == 0: clust = clust.unsqueeze(0)
            if katz.dim() == 0: katz = katz.unsqueeze(0)
        
        features = torch.stack([deg_norm, clust, pr, eig, katz], dim=1).cpu()
        return features, node_list
    
    @staticmethod
    def _extract_enhanced_features(G, node_list):
        """12维增强特征"""
        num_nodes = len(node_list)
        
        degrees = dict(G.degree())
        clustering = nx.clustering(G)
        
        try:
            pagerank = nx.pagerank(G, max_iter=100)
        except:
            pagerank = {n: 1.0/num_nodes for n in G.nodes()}
        
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            eigenvector = {n: 1.0/num_nodes for n in G.nodes()}
        
        # 介数中心性
        if num_nodes <= 500:
            betweenness = nx.betweenness_centrality(G)
        else:
            betweenness = nx.betweenness_centrality(G, k=min(100, num_nodes))
        
        # 接近中心性
        if nx.is_connected(G):
            closeness = nx.closeness_centrality(G)
        else:
            closeness = {n: 0.0 for n in G.nodes()}
            for comp in nx.connected_components(G):
                subG = G.subgraph(comp)
                for node, val in nx.closeness_centrality(subG).items():
                    closeness[node] = val
        
        core_number = nx.core_number(G)
        avg_neighbor_degree = nx.average_neighbor_degree(G)
        
        if nx.is_connected(G):
            eccentricity = nx.eccentricity(G)
        else:
            eccentricity = {}
            for comp in nx.connected_components(G):
                subG = G.subgraph(comp)
                eccentricity.update(nx.eccentricity(subG))
        
        triangles = nx.triangles(G)
        
        features = []
        for node in node_list:
            feat = [
                degrees[node] / max(num_nodes - 1, 1),
                clustering[node],
                pagerank[node] * num_nodes,
                eigenvector[node],
                betweenness[node],
                closeness[node],
                core_number[node] / max(max(core_number.values()), 1),
                avg_neighbor_degree[node] / max(num_nodes - 1, 1) if degrees[node] > 0 else 0,
                1.0 - eccentricity.get(node, num_nodes) / max(num_nodes, 1),
                triangles[node] / max(degrees[node] * (degrees[node] - 1) / 2, 1) if degrees[node] >= 2 else 0,
                np.std([degrees[n] for n in G.neighbors(node)]) / max(num_nodes, 1) if degrees[node] > 0 else 0,
                GraphFeatureExtractor._local_bridge_score(G, node, degrees),
            ]
            features.append(feat)
        
        features = torch.tensor(features, dtype=torch.float32)
        
        # 归一化
        if num_nodes > 1:
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True) + 1e-6
            features = (features - mean) / std
        
        return features, node_list
    
    @staticmethod
    def _extract_structural_features(G, node_list):
        """15维结构特征（包含更多图结构信息）"""
        # 先获取12维增强特征
        base_features, _ = GraphFeatureExtractor._extract_enhanced_features(G, node_list)
        
        num_nodes = len(node_list)
        degrees = dict(G.degree())
        
        # 额外3维结构特征
        extra_features = []
        
        # 2跳邻居数
        two_hop_neighbors = {}
        for node in node_list:
            neighbors_1 = set(G.neighbors(node))
            neighbors_2 = set()
            for n in neighbors_1:
                neighbors_2.update(G.neighbors(n))
            neighbors_2 -= neighbors_1
            neighbors_2.discard(node)
            two_hop_neighbors[node] = len(neighbors_2)
        
        max_2hop = max(two_hop_neighbors.values()) if two_hop_neighbors else 1
        
        for node in node_list:
            extra = [
                two_hop_neighbors[node] / max_2hop,  # 2跳邻居比例
                len(list(G.neighbors(node))) / max(num_nodes - 1, 1),  # 直接邻居比例
                GraphFeatureExtractor._structural_hole_score(G, node),  # 结构洞分数
            ]
            extra_features.append(extra)
        
        extra_features = torch.tensor(extra_features, dtype=torch.float32)
        
        # 合并特征
        features = torch.cat([base_features, extra_features], dim=1)
        
        return features, node_list
    
    @staticmethod
    def _local_bridge_score(G, node, degrees):
        """计算本地桥接分数"""
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            return 0.0
        
        neighbor_edges = sum(1 for i, n1 in enumerate(neighbors) 
                           for n2 in neighbors[i+1:] if G.has_edge(n1, n2))
        max_possible = len(neighbors) * (len(neighbors) - 1) / 2
        
        return 1.0 - neighbor_edges / max_possible if max_possible > 0 else 0.0
    
    @staticmethod
    def _structural_hole_score(G, node):
        """计算结构洞分数（约束度量）"""
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            return 0.0
        
        # 简化的结构洞计算：邻居之间的稀疏程度
        neighbor_edges = sum(1 for i, n1 in enumerate(neighbors)
                           for n2 in neighbors[i+1:] if G.has_edge(n1, n2))
        max_possible = len(neighbors) * (len(neighbors) - 1) / 2
        
        # 结构洞分数 = 1 - 邻居间连接密度
        return 1.0 - neighbor_edges / max_possible if max_possible > 0 else 0.0


# ============================================================
# 2. Actor-Critic 网络架构
# ============================================================

class ActorCriticPolicy(nn.Module):
    """
    Actor-Critic 策略网络
    
    - Actor: 输出节点选择概率
    - Critic: 输出状态价值估计
    """
    
    def __init__(self, num_features=12, hidden_dim=128, num_layers=3):
        super().__init__()
        
        self.num_features = num_features
        
        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Actor头
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Critic头（全局状态价值）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # 温度参数
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x, mask=None, return_value=False):
        """
        Args:
            x: 节点特征 [N, F]
            mask: 已选节点掩码 [N]
            return_value: 是否返回价值估计
        """
        # 共享特征
        h = self.shared(x)  # [N, H]
        
        # Actor: 节点选择分数
        scores = self.actor(h).squeeze(-1)  # [N]
        
        # 温度缩放
        temp = torch.clamp(self.temperature, min=0.1, max=2.0)
        scores = scores / temp
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        probs = F.softmax(scores, dim=0)
        
        if return_value:
            # Critic: 全局状态价值（平均池化）
            global_h = h.mean(dim=0, keepdim=True)  # [1, H]
            value = self.critic(global_h).squeeze()  # scalar
            return probs, value
        
        return probs
    
    def get_action(self, x, mask=None, deterministic=False):
        """获取动作、log概率和价值估计"""
        probs, value = self.forward(x, mask, return_value=True)
        
        if deterministic:
            action = torch.argmax(probs)
            log_prob = torch.log(probs[action] + 1e-8)
        else:
            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
        
        entropy = Categorical(probs).entropy()
        
        return action, log_prob, value, entropy


# ============================================================
# 3. 多目标奖励函数
# ============================================================

class MultiObjectiveReward:
    """
    多目标奖励函数
    
    综合考虑多个优化目标：
    - GCC保持率
    - 控制器分散性
    - 连通分量覆盖
    - 控制器保护（避免高度数节点）
    """
    
    def __init__(self, 
                 gcc_weight=0.4,
                 dispersion_weight=0.25,
                 coverage_weight=0.2,
                 protection_weight=0.15):
        self.gcc_weight = gcc_weight
        self.dispersion_weight = dispersion_weight
        self.coverage_weight = coverage_weight
        self.protection_weight = protection_weight
    
    def __call__(self, G, centers, attack_ratios=None):
        """计算多目标奖励"""
        if not centers:
            return 0.0
        
        num_nodes = G.number_of_nodes()
        
        # 根据图规模调整攻击比例
        if attack_ratios is None:
            if num_nodes <= 30:
                attack_ratios = [0.15, 0.25, 0.35]
            elif num_nodes <= 100:
                attack_ratios = [0.1, 0.2, 0.3]
            else:
                attack_ratios = [0.05, 0.1, 0.15, 0.2]
        
        # 1. GCC保持率（多攻击比例）
        gcc_reward = self._calculate_gcc_reward(G, centers, attack_ratios)
        
        # 2. 分散性奖励
        dispersion_reward = self._calculate_dispersion_reward(G, centers)
        
        # 3. 覆盖奖励
        coverage_reward = self._calculate_coverage_reward(G, centers)
        
        # 4. 保护奖励
        protection_reward = self._calculate_protection_reward(G, centers)
        
        # 组合奖励
        total_reward = (
            self.gcc_weight * gcc_reward +
            self.dispersion_weight * dispersion_reward +
            self.coverage_weight * coverage_reward +
            self.protection_weight * protection_reward
        )
        
        return total_reward
    
    def _calculate_gcc_reward(self, G, centers, attack_ratios):
        """计算多攻击比例下的GCC保持率"""
        centers_set = set(centers)
        num_nodes = G.number_of_nodes()
        
        gcc_scores = []
        for ratio in attack_ratios:
            G_attacked = G.copy()
            remaining_centers = centers_set.copy()
            
            num_remove = max(1, int(num_nodes * ratio))
            degrees = dict(G_attacked.degree())
            targets = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:num_remove]
            
            for node in targets:
                if G_attacked.has_node(node):
                    G_attacked.remove_node(node)
                    remaining_centers.discard(node)
            
            if G_attacked.number_of_nodes() > 0 and remaining_centers:
                components = list(nx.connected_components(G_attacked))
                controlled_sizes = [len(c) for c in components if not remaining_centers.isdisjoint(c)]
                gcc_ratio = max(controlled_sizes) / num_nodes if controlled_sizes else 0.0
            else:
                gcc_ratio = 0.0
            
            gcc_scores.append(gcc_ratio)
        
        # 加权平均（后期攻击权重更高）
        weights = np.linspace(0.5, 1.5, len(gcc_scores))
        weights = weights / weights.sum()
        
        return sum(w * s for w, s in zip(weights, gcc_scores))
    
    def _calculate_dispersion_reward(self, G, centers):
        """计算控制器分散性"""
        if len(centers) <= 1:
            return 0.5
        
        centers_list = list(centers)
        distances = []
        
        for i, c1 in enumerate(centers_list):
            for c2 in centers_list[i+1:]:
                if G.has_node(c1) and G.has_node(c2):
                    try:
                        if nx.has_path(G, c1, c2):
                            d = nx.shortest_path_length(G, c1, c2)
                            distances.append(d)
                    except:
                        pass
        
        if not distances:
            return 0.5
        
        avg_dist = np.mean(distances)
        min_dist = min(distances)
        
        try:
            diameter = nx.diameter(G) if nx.is_connected(G) else G.number_of_nodes() // 3
        except:
            diameter = G.number_of_nodes() // 3
        
        ideal_dist = max(diameter / 3, 2)
        
        avg_score = min(avg_dist / ideal_dist, 1.0)
        min_score = min(min_dist / (ideal_dist * 0.5), 1.0)
        
        return 0.6 * avg_score + 0.4 * min_score
    
    def _calculate_coverage_reward(self, G, centers):
        """计算连通分量覆盖"""
        centers_set = set(centers)
        components = list(nx.connected_components(G))
        
        covered = sum(1 for c in components if not centers_set.isdisjoint(c))
        return covered / len(components) if components else 1.0
    
    def _calculate_protection_reward(self, G, centers):
        """计算控制器保护奖励"""
        degrees = dict(G.degree())
        sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
        
        top_k = max(1, int(len(sorted_nodes) * 0.1))
        high_degree_nodes = set(sorted_nodes[:top_k])
        
        protected = len([c for c in centers if c not in high_degree_nodes])
        return protected / len(centers) if centers else 0.0


# ============================================================
# 4. 增强型RL训练器
# ============================================================

class EnhancedRLTrainer:
    """
    增强型RL训练器
    
    特点：
    - Actor-Critic架构
    - 多目标奖励
    - GAE优势估计
    - 课程学习
    - 经验回放
    """
    
    def __init__(
        self,
        num_features=12,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.02,
        value_coef=0.5,
        max_grad_norm=0.5,
    ):
        self.device = DEVICE
        
        self.model = ActorCriticPolicy(
            num_features=num_features,
            hidden_dim=hidden_dim,
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=lr * 0.01
        )
        
        self.reward_fn = MultiObjectiveReward()
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # 训练状态
        self.best_reward = -float('inf')
        self.best_state = None
        self.history = []
        
        # 经验回放
        self.replay_buffer = deque(maxlen=1000)
    
    def train(
        self,
        graphs,
        epochs=100,
        k_ratio=0.1,
        save_path=None,
        curriculum=True,
        verbose=True,
    ):
        """
        训练模型
        
        Args:
            graphs: 训练图列表
            epochs: 训练轮数
            k_ratio: 控制器比例
            save_path: 保存路径
            curriculum: 是否使用课程学习
            verbose: 是否显示进度
        """
        if not graphs:
            logger.error("没有训练图")
            return None
        
        # 课程学习：按图大小排序
        if curriculum:
            graphs = sorted(graphs, key=lambda g: g.number_of_nodes())
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Enhanced RL Controller Selector Training")
            print(f"{'='*60}")
            print(f"  Graphs: {len(graphs)}")
            print(f"  Epochs: {epochs}")
            print(f"  Device: {self.device}")
            print(f"{'='*60}\n")
        
        epoch_pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
        
        for epoch in epoch_pbar:
            self.model.train()
            
            epoch_rewards = []
            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_entropies = []
            
            # 课程学习
            if curriculum:
                progress = (epoch + 1) / epochs
                num_graphs = int(len(graphs) * min(1.0, 0.3 + 0.7 * progress))
                epoch_graphs = graphs[:num_graphs]
            else:
                epoch_graphs = graphs
            
            random.shuffle(epoch_graphs)
            
            for G in epoch_graphs:
                if G.number_of_nodes() < 3:
                    continue
                
                # 收集轨迹
                trajectory = self._collect_trajectory(G, k_ratio)
                if trajectory is None:
                    continue
                
                # 计算优势
                advantages, returns = self._compute_gae(trajectory)
                
                # 更新策略
                policy_loss, value_loss, entropy = self._update(trajectory, advantages, returns)
                
                epoch_rewards.append(trajectory['reward'])
                epoch_policy_losses.append(policy_loss)
                epoch_value_losses.append(value_loss)
                epoch_entropies.append(entropy)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录
            if epoch_rewards:
                avg_reward = np.mean(epoch_rewards)
                avg_policy_loss = np.mean(epoch_policy_losses)
                avg_value_loss = np.mean(epoch_value_losses)
                avg_entropy = np.mean(epoch_entropies)
                
                self.history.append({
                    'epoch': epoch + 1,
                    'reward': avg_reward,
                    'policy_loss': avg_policy_loss,
                    'value_loss': avg_value_loss,
                    'entropy': avg_entropy,
                })
                
                # 保存最佳模型
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                
                epoch_pbar.set_postfix({
                    'reward': f'{avg_reward:.4f}',
                    'best': f'{self.best_reward:.4f}',
                    'entropy': f'{avg_entropy:.4f}',
                })
        
        # 保存模型
        if save_path:
            self._save_model(save_path)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"  Best Reward: {self.best_reward:.4f}")
            print(f"{'='*60}\n")
        
        return {
            'history': self.history,
            'best_reward': self.best_reward,
        }
    
    def _collect_trajectory(self, G, k_ratio):
        """收集单个图的轨迹"""
        x, node_list = GraphFeatureExtractor.extract_node_features(G, 'enhanced')
        
        if x.shape[1] != self.model.num_features:
            return None
        
        x = x.to(self.device)
        k = max(1, int(G.number_of_nodes() * k_ratio))
        
        log_probs = []
        values = []
        entropies = []
        centers = []
        mask = torch.zeros(len(node_list), dtype=torch.bool, device=self.device)
        
        for _ in range(k):
            action, log_prob, value, entropy = self.model.get_action(x, mask)
            
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
            
            idx = action.item()
            centers.append(node_list[idx])
            mask = mask.clone()
            mask[idx] = True
        
        # 计算奖励
        reward = self.reward_fn(G, centers)
        
        return {
            'log_probs': log_probs,
            'values': values,
            'entropies': entropies,
            'reward': reward,
            'centers': centers,
        }
    
    def _compute_gae(self, trajectory):
        """计算GAE优势估计"""
        values = trajectory['values']
        reward = trajectory['reward']
        
        # 简化：所有步骤共享同一个最终奖励
        returns = [reward] * len(values)
        
        # GAE
        advantages = []
        gae = 0
        
        for t in reversed(range(len(values))):
            if t == len(values) - 1:
                next_value = 0
            else:
                next_value = values[t + 1].item()
            
            delta = reward / len(values) - values[t].item() + self.gamma * next_value
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, device=self.device)
        returns = torch.tensor(returns, device=self.device)
        
        # 归一化优势
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _update(self, trajectory, advantages, returns):
        """更新策略"""
        log_probs = torch.stack(trajectory['log_probs'])
        values = torch.stack(trajectory['values'])
        entropies = torch.stack(trajectory['entropies'])
        
        # 策略损失
        policy_loss = -(log_probs * advantages).mean()
        
        # 价值损失
        value_loss = F.mse_loss(values, returns)
        
        # 熵损失
        entropy_loss = -entropies.mean()
        
        # 总损失
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropies.mean().item()
    
    def _save_model(self, save_path):
        """保存模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.best_state or self.model.state_dict(),
            'num_features': self.model.num_features,
            'best_reward': self.best_reward,
            'history': self.history,
        }
        torch.save(checkpoint, save_path)
        logger.info(f"模型已保存: {save_path}")
    
    def select_controllers(self, G, k, deterministic=True):
        """使用模型选择控制器"""
        self.model.eval()
        
        x, node_list = GraphFeatureExtractor.extract_node_features(G, 'enhanced')
        x = x.to(self.device)
        
        centers = []
        mask = torch.zeros(len(node_list), dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            for _ in range(k):
                action, _, _, _ = self.model.get_action(x, mask, deterministic=deterministic)
                
                idx = action.item()
                centers.append(node_list[idx])
                mask[idx] = True
        
        return centers


# ============================================================
# 5. 工具函数和快速入口
# ============================================================

def load_enhanced_rl_model(model_path):
    """加载增强型RL模型"""
    if not os.path.exists(model_path):
        logger.warning(f"模型文件不存在: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    model = ActorCriticPolicy(
        num_features=checkpoint.get('num_features', 12),
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def train_enhanced_rl_selector(
    graphs,
    epochs=100,
    save_path=None,
    lr=3e-4,
    verbose=True,
):
    """
    训练增强型RL控制器选择器
    
    Args:
        graphs: 训练图列表
        epochs: 训练轮数
        save_path: 保存路径
        lr: 学习率
        verbose: 是否显示进度
    """
    if save_path is None:
        save_path = 'src/train/v1/checkpoints/enhanced_rl_selector.pth'
    
    trainer = EnhancedRLTrainer(lr=lr)
    result = trainer.train(
        graphs,
        epochs=epochs,
        save_path=save_path,
        curriculum=True,
        verbose=verbose,
    )
    
    return result, trainer


def enhanced_rl_select(G, k, model_path=None, model=None):
    """
    使用增强型RL选择控制器
    
    Args:
        G: NetworkX图
        k: 控制器数量
        model_path: 模型路径
        model: 预加载的模型
    """
    if model is None and model_path:
        model = load_enhanced_rl_model(model_path)
    
    if model is None:
        logger.warning("模型不可用，使用CI算法")
        from src.bit_hyrl.selection import ci_select_subset
        return ci_select_subset(G, list(G.nodes()), k, [], radius=2)
    
    trainer = EnhancedRLTrainer()
    trainer.model = model
    
    return trainer.select_controllers(G, k, deterministic=True)
