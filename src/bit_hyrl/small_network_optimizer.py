# -*- coding: utf-8 -*-
"""
BiT-HyRL 小规模网络优化器

针对小规模网络（节点数 < 100）的专门优化模块：
1. 增强特征提取（不依赖Node2Vec）
2. 自适应奖励函数（根据网络规模调整攻击比例）
3. 专门的小图训练策略
4. 多尺度训练（课程学习）
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

from src.utils.logger import get_logger
from . import config

logger = get_logger(__name__)
DEVICE = config.DEVICE


# ============================================================
# 1. 增强特征提取（针对小规模网络）
# ============================================================

def get_enhanced_small_graph_features(G, normalize=True):
    """
    针对小规模网络的增强特征提取（不依赖Node2Vec）
    
    特征维度：12维
    1. 度数归一化
    2. 聚类系数
    3. PageRank
    4. 特征向量中心性
    5. 介数中心性（对小图可行）
    6. 接近中心性（对小图可行）
    7. 核数（K-core number）
    8. 平均邻居度
    9. 到最远节点的距离（偏心率）
    10. 本地效率
    11. 度-度相关性
    12. 三角形参与度
    """
    num_nodes = G.number_of_nodes()
    if num_nodes == 0:
        return torch.zeros((0, 12)), []
    
    node_list = list(G.nodes())
    features = []
    
    # 预计算一些图属性
    degrees = dict(G.degree())
    
    # 聚类系数
    clustering = nx.clustering(G)
    
    # PageRank
    try:
        pagerank = nx.pagerank(G, max_iter=100)
    except:
        pagerank = {n: 1.0/num_nodes for n in G.nodes()}
    
    # 特征向量中心性
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigenvector = {n: 1.0/num_nodes for n in G.nodes()}
    
    # 介数中心性（小图可行）
    if num_nodes <= 500:
        betweenness = nx.betweenness_centrality(G)
    else:
        # 大图使用近似
        betweenness = nx.betweenness_centrality(G, k=min(100, num_nodes))
    
    # 接近中心性（仅对连通图有意义）
    if nx.is_connected(G):
        closeness = nx.closeness_centrality(G)
    else:
        closeness = {n: 0.0 for n in G.nodes()}
        for comp in nx.connected_components(G):
            subG = G.subgraph(comp)
            sub_closeness = nx.closeness_centrality(subG)
            for node, val in sub_closeness.items():
                closeness[node] = val
    
    # K-core number
    core_number = nx.core_number(G)
    
    # 平均邻居度
    avg_neighbor_degree = nx.average_neighbor_degree(G)
    
    # 偏心率（到最远节点的距离）
    if nx.is_connected(G):
        eccentricity = nx.eccentricity(G)
    else:
        eccentricity = {}
        for comp in nx.connected_components(G):
            subG = G.subgraph(comp)
            sub_ecc = nx.eccentricity(subG)
            eccentricity.update(sub_ecc)
    
    # 三角形数
    triangles = nx.triangles(G)
    
    # 构建特征向量
    for node in node_list:
        feat = [
            degrees[node] / max(num_nodes - 1, 1),  # 度数归一化
            clustering[node],  # 聚类系数
            pagerank[node] * num_nodes,  # PageRank（放大）
            eigenvector[node],  # 特征向量中心性
            betweenness[node],  # 介数中心性
            closeness[node],  # 接近中心性
            core_number[node] / max(max(core_number.values()), 1),  # 核数归一化
            avg_neighbor_degree[node] / max(num_nodes - 1, 1) if degrees[node] > 0 else 0,  # 平均邻居度
            1.0 - eccentricity.get(node, num_nodes) / max(num_nodes, 1),  # 偏心率（倒数）
            triangles[node] / max(degrees[node] * (degrees[node] - 1) / 2, 1) if degrees[node] >= 2 else 0,  # 三角形参与度
            # 度-度相关性：与邻居度数的相关
            np.std([degrees[n] for n in G.neighbors(node)]) / max(num_nodes, 1) if degrees[node] > 0 else 0,
            # 本地桥接性：如果移除该节点，邻居之间的连通性
            _calculate_local_bridge_score(G, node, degrees),
        ]
        features.append(feat)
    
    features = torch.tensor(features, dtype=torch.float32)
    
    # 归一化
    if normalize and num_nodes > 1:
        # 对每个特征进行z-score标准化
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True) + 1e-6
        features = (features - mean) / std
        # 再做min-max缩放到[0,1]
        min_val = features.min(dim=0, keepdim=True)[0]
        max_val = features.max(dim=0, keepdim=True)[0]
        features = (features - min_val) / (max_val - min_val + 1e-6)
    
    return features, node_list


def _calculate_local_bridge_score(G, node, degrees):
    """计算节点的本地桥接分数：移除后邻居间连通性的变化"""
    neighbors = list(G.neighbors(node))
    if len(neighbors) < 2:
        return 0.0
    
    # 检查邻居之间的连接数
    neighbor_edges = 0
    for i, n1 in enumerate(neighbors):
        for n2 in neighbors[i+1:]:
            if G.has_edge(n1, n2):
                neighbor_edges += 1
    
    max_possible = len(neighbors) * (len(neighbors) - 1) / 2
    if max_possible == 0:
        return 0.0
    
    # 如果邻居之间连接少，说明这个节点是重要的桥接节点
    bridge_score = 1.0 - neighbor_edges / max_possible
    return bridge_score


# ============================================================
# 2. 自适应奖励函数
# ============================================================

def calculate_adaptive_reward(G, centers, attack_strategy='degree'):
    """
    自适应奖励函数：根据网络规模自动调整攻击参数
    
    小规模网络：
    - 更高的攻击比例（15-25%）以获得更强的信号
    - 多轮攻击模拟
    - 更重视控制器分散性
    
    Args:
        G: NetworkX图
        centers: 控制器列表
        attack_strategy: 攻击策略 ('degree', 'betweenness', 'random')
    """
    if not centers:
        return 0.0
    
    num_nodes = G.number_of_nodes()
    centers_set = set(centers)
    
    # 根据网络规模调整攻击比例
    if num_nodes <= 30:
        attack_ratios = [0.15, 0.25, 0.35]  # 小图用更高比例
    elif num_nodes <= 100:
        attack_ratios = [0.1, 0.2, 0.3]
    else:
        attack_ratios = [0.05, 0.1, 0.15, 0.2]
    
    # 多轮攻击奖励
    gcc_scores = []
    for ratio in attack_ratios:
        G_attacked = G.copy()
        remaining_centers = centers_set.copy()
        
        num_remove = max(1, int(num_nodes * ratio))
        
        # 获取攻击目标
        if attack_strategy == 'degree':
            degrees = dict(G_attacked.degree())
            targets = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:num_remove]
        elif attack_strategy == 'betweenness':
            betweenness = nx.betweenness_centrality(G_attacked)
            targets = sorted(betweenness.keys(), key=lambda x: betweenness[x], reverse=True)[:num_remove]
        else:
            targets = random.sample(list(G_attacked.nodes()), min(num_remove, G_attacked.number_of_nodes()))
        
        # 执行攻击
        for node in targets:
            if G_attacked.has_node(node):
                G_attacked.remove_node(node)
                remaining_centers.discard(node)
        
        # 计算GCC保持率
        if G_attacked.number_of_nodes() > 0 and remaining_centers:
            components = list(nx.connected_components(G_attacked))
            controlled_sizes = [len(c) for c in components if not remaining_centers.isdisjoint(c)]
            gcc_ratio = max(controlled_sizes) / num_nodes if controlled_sizes else 0.0
        else:
            gcc_ratio = 0.0
        
        gcc_scores.append(gcc_ratio)
    
    # GCC奖励：加权平均（后期攻击权重更高）
    weights = [0.2, 0.3, 0.5] if len(attack_ratios) == 3 else [0.1, 0.2, 0.3, 0.4]
    weights = weights[:len(gcc_scores)]
    gcc_reward = sum(w * s for w, s in zip(weights, gcc_scores))
    
    # 控制器分散性奖励（对小图更重要）
    dispersion_reward = _calculate_dispersion_reward(G, centers)
    
    # 控制器保护奖励（控制器不应在高度数节点）
    protection_reward = _calculate_protection_reward(G, centers)
    
    # 覆盖奖励（覆盖所有连通分量）
    coverage_reward = _calculate_coverage_reward(G, centers)
    
    # 组合奖励（小图更重视分散性和保护）
    if num_nodes <= 50:
        # 小图：40% GCC + 25% 分散性 + 20% 保护 + 15% 覆盖
        return 0.40 * gcc_reward + 0.25 * dispersion_reward + 0.20 * protection_reward + 0.15 * coverage_reward
    else:
        # 大图：50% GCC + 20% 分散性 + 15% 保护 + 15% 覆盖
        return 0.50 * gcc_reward + 0.20 * dispersion_reward + 0.15 * protection_reward + 0.15 * coverage_reward


def _calculate_dispersion_reward(G, centers):
    """计算控制器分散性奖励"""
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
    
    # 理想距离：图直径的1/3
    try:
        diameter = nx.diameter(G) if nx.is_connected(G) else G.number_of_nodes() // 3
    except:
        diameter = G.number_of_nodes() // 3
    
    ideal_dist = max(diameter / 3, 2)
    
    # 分散性得分：结合平均距离和最小距离
    avg_score = min(avg_dist / ideal_dist, 1.0)
    min_score = min(min_dist / (ideal_dist * 0.5), 1.0)
    
    return 0.6 * avg_score + 0.4 * min_score


def _calculate_protection_reward(G, centers):
    """计算控制器保护奖励（不在高度数节点）"""
    if not centers:
        return 0.0
    
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    
    # 前10%的高度数节点
    top_k = max(1, int(len(sorted_nodes) * 0.1))
    high_degree_nodes = set(sorted_nodes[:top_k])
    
    # 控制器不在高度数节点中的比例
    protected = len([c for c in centers if c not in high_degree_nodes])
    return protected / len(centers)


def _calculate_coverage_reward(G, centers):
    """计算连通分量覆盖奖励"""
    if not centers:
        return 0.0
    
    centers_set = set(centers)
    components = list(nx.connected_components(G))
    
    covered = sum(1 for c in components if not centers_set.isdisjoint(c))
    return covered / len(components) if components else 1.0


# ============================================================
# 3. 小规模网络专用模型
# ============================================================

class SmallGraphPolicy(nn.Module):
    """
    针对小规模网络的策略网络
    
    特点：
    - 更深的网络（4层）捕捉复杂模式
    - 注意力机制关注节点间关系
    - 残差连接防止梯度消失
    - 可学习的温度参数
    """
    
    def __init__(self, num_features=12, hidden_dim=64, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 深度残差块
        self.res_blocks = nn.ModuleList([
            self._make_res_block(hidden_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        # 自注意力层（用于捕捉节点间关系）
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # 可学习温度
        self.temperature = nn.Parameter(torch.ones(1))
        
    def _make_res_block(self, dim, dropout):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: 节点特征 [N, F]
            mask: 已选节点掩码 [N]
        """
        # 输入投影
        h = self.input_proj(x)  # [N, H]
        
        # 残差块
        for res_block in self.res_blocks:
            h = h + F.relu(res_block(h))
        
        # 自注意力（添加batch维度）
        h = h.unsqueeze(0)  # [1, N, H]
        attn_out, _ = self.self_attention(h, h, h)
        h = self.attn_norm(h + attn_out)
        h = h.squeeze(0)  # [N, H]
        
        # 输出分数
        scores = self.output(h).squeeze(-1)  # [N]
        
        # 温度缩放
        temp = torch.clamp(self.temperature, min=0.1, max=2.0)
        scores = scores / temp
        
        # 掩码
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        return F.softmax(scores, dim=0)


# ============================================================
# 4. 小规模网络专用训练器
# ============================================================

class SmallNetworkTrainer:
    """
    小规模网络专用训练器
    
    特点：
    - 课程学习：从小图到大图
    - 多策略奖励
    - 增强探索
    - 早停机制
    """
    
    def __init__(
        self,
        model=None,
        lr=1e-3,
        entropy_coef=0.02,  # 更高的熵系数鼓励探索
        gamma=0.99,
        device=None,
    ):
        self.device = device or DEVICE
        self.model = model or SmallGraphPolicy().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01
        )
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        
        # 训练状态
        self.baseline = 0.0
        self.best_reward = -float('inf')
        self.best_state = None
        self.history = []
        
    def train_on_graphs(
        self,
        graphs,
        epochs=100,
        k_ratio=0.1,
        save_path=None,
        curriculum=True,
        verbose=True,
    ):
        """
        在图集合上训练
        
        Args:
            graphs: 训练图列表
            epochs: 训练轮数
            k_ratio: 控制器比例
            save_path: 模型保存路径
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
            print(f"Small Network Trainer")
            print(f"{'='*60}")
            print(f"  Graphs: {len(graphs)}")
            print(f"  Size range: {graphs[0].number_of_nodes()} - {graphs[-1].number_of_nodes()}")
            print(f"  Epochs: {epochs}")
            print(f"  Features: {self.model.num_features}D")
            print(f"{'='*60}\n")
        
        epoch_pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
        
        for epoch in epoch_pbar:
            self.model.train()
            total_reward = 0.0
            total_loss = 0.0
            graph_count = 0
            
            # 课程学习：逐渐引入更大的图
            if curriculum:
                # 前半段只用小图，后半段用所有图
                progress = (epoch + 1) / epochs
                num_graphs = int(len(graphs) * min(1.0, 0.3 + 0.7 * progress))
                epoch_graphs = graphs[:num_graphs]
            else:
                epoch_graphs = graphs
            
            random.shuffle(epoch_graphs)
            self.optimizer.zero_grad()
            
            for i, G in enumerate(epoch_graphs):
                if G.number_of_nodes() < 3:
                    continue
                
                # 获取特征
                x, node_list = get_enhanced_small_graph_features(G)
                if x.shape[1] != self.model.num_features:
                    continue
                
                x = x.to(self.device)
                k = max(1, int(G.number_of_nodes() * k_ratio))
                
                # 采样动作
                log_probs = []
                entropies = []
                centers = []
                mask = torch.zeros(len(node_list), dtype=torch.bool, device=self.device)
                
                for _ in range(k):
                    probs = self.model(x, mask)
                    m = Categorical(probs)
                    action = m.sample()
                    
                    log_probs.append(m.log_prob(action))
                    entropies.append(m.entropy())
                    
                    idx = action.item()
                    centers.append(node_list[idx])
                    mask = mask.clone()
                    mask[idx] = True
                
                # 计算奖励
                reward = calculate_adaptive_reward(G, centers)
                total_reward += reward
                graph_count += 1
                
                # 更新baseline
                if self.baseline == 0.0:
                    self.baseline = reward
                else:
                    self.baseline = 0.9 * self.baseline + 0.1 * reward
                
                # 计算advantage
                advantage = reward - self.baseline
                
                # 策略损失
                policy_loss = sum(-lp * advantage for lp in log_probs)
                
                # 熵损失（鼓励探索）
                entropy_loss = -self.entropy_coef * sum(entropies)
                
                # 总损失
                loss = (policy_loss + entropy_loss) / 8  # 梯度累积
                loss.backward()
                total_loss += loss.item()
                
                # 梯度累积
                if (i + 1) % 8 == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # 处理剩余梯度
            if len(epoch_graphs) % 8 != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录
            avg_reward = total_reward / graph_count if graph_count > 0 else 0.0
            self.history.append([epoch + 1, avg_reward])
            
            # 保存最佳模型
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            epoch_pbar.set_postfix({
                'reward': f'{avg_reward:.4f}',
                'best': f'{self.best_reward:.4f}',
                'graphs': graph_count,
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
    
    def _save_model(self, save_path):
        """保存模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.best_state or self.model.state_dict(),
            'num_features': self.model.num_features,
            'hidden_dim': self.model.hidden_dim,
            'best_reward': self.best_reward,
            'history': self.history,
        }
        torch.save(checkpoint, save_path)
        logger.info(f"模型已保存: {save_path}")
    
    def select_controllers(self, G, k, deterministic=True):
        """使用训练好的模型选择控制器"""
        self.model.eval()
        
        x, node_list = get_enhanced_small_graph_features(G)
        x = x.to(self.device)
        
        centers = []
        mask = torch.zeros(len(node_list), dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            for _ in range(k):
                probs = self.model(x, mask)
                
                if deterministic:
                    action = torch.argmax(probs)
                else:
                    m = Categorical(probs)
                    action = m.sample()
                
                idx = action.item()
                centers.append(node_list[idx])
                mask[idx] = True
        
        return centers


# ============================================================
# 5. 工具函数
# ============================================================

def load_small_graph_model(model_path):
    """加载小图专用模型"""
    if not os.path.exists(model_path):
        logger.warning(f"模型文件不存在: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    model = SmallGraphPolicy(
        num_features=checkpoint.get('num_features', 12),
        hidden_dim=checkpoint.get('hidden_dim', 64),
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def hybrid_small_graph_select(G, k, model_path=None, model=None, use_ci_fallback=True):
    """
    混合小图选择：RL选择所有控制器（小图）或RL+CI组合（大图）
    
    对于小图（<50节点），完全使用RL选择
    对于中等图（50-200节点），RL选择一半，CI选择另一半
    对于大图（>200节点），使用标准混合策略
    """
    num_nodes = G.number_of_nodes()
    
    # 加载模型
    if model is None and model_path:
        model = load_small_graph_model(model_path)
    
    if model is None:
        if use_ci_fallback:
            logger.warning("模型不可用，使用CI算法")
            from .selection import ci_select_subset
            return ci_select_subset(G, list(G.nodes()), k, [], radius=2)
        return []
    
    # 小图：完全使用RL
    if num_nodes <= 50:
        trainer = SmallNetworkTrainer(model=model)
        return trainer.select_controllers(G, k, deterministic=True)
    
    # 中等图：RL选择一半
    elif num_nodes <= 200:
        rl_k = max(1, k // 2)
        trainer = SmallNetworkTrainer(model=model)
        rl_centers = trainer.select_controllers(G, rl_k, deterministic=True)
        
        # CI选择剩余
        remaining_k = k - len(rl_centers)
        if remaining_k > 0 and use_ci_fallback:
            from .selection import ci_select_subset
            ci_centers = ci_select_subset(G, list(G.nodes()), remaining_k, rl_centers, radius=2)
            return rl_centers + ci_centers
        return rl_centers
    
    # 大图：标准混合策略
    else:
        from .selection import hybrid_rl_select
        return hybrid_rl_select(G, k, model_path=model_path)


# ============================================================
# 6. 快速训练入口
# ============================================================

def train_small_network_model(
    graphs,
    epochs=100,
    save_path=None,
    lr=1e-3,
    verbose=True,
):
    """
    训练小规模网络专用模型的快速入口
    
    Args:
        graphs: 训练图列表（建议包含各种规模的小图）
        epochs: 训练轮数
        save_path: 保存路径
        lr: 学习率
        verbose: 是否显示进度
    """
    if save_path is None:
        save_path = os.path.join(config.MODEL_DIR, 'small_network_agent.pth')
    
    trainer = SmallNetworkTrainer(lr=lr)
    result = trainer.train_on_graphs(
        graphs,
        epochs=epochs,
        save_path=save_path,
        curriculum=True,
        verbose=verbose,
    )
    
    return result
