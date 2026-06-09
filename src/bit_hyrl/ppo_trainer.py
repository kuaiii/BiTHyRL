# -*- coding: utf-8 -*-
"""
BiT-HyRL PPO 训练器

实现 Proximal Policy Optimization 算法，
支持 GAT 策略网络的训练。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from tqdm import tqdm
import random

from . import config
from .gnn_model import GATPolicy, graph_to_pyg_data, get_gnn_node_features
from .reward import set_current_epoch

DEVICE = config.DEVICE

# 经验数据结构
Experience = namedtuple('Experience', [
    'x',              # 节点特征
    'edge_index',     # 边索引
    'action',         # 选择的节点
    'log_prob',       # 动作的对数概率
    'value',          # 状态价值估计
    'reward',         # 奖励
    'selected_mask',  # 已选节点 mask
    'done',           # 是否完成
])


class RolloutBuffer:
    """轨迹数据缓冲区"""
    
    def __init__(self):
        self.experiences = []
        self.episode_rewards = []
        
    def add(self, exp):
        self.experiences.append(exp)
        
    def add_episode_reward(self, reward):
        self.episode_rewards.append(reward)
        
    def clear(self):
        self.experiences = []
        self.episode_rewards = []
        
    def __len__(self):
        return len(self.experiences)


class PPOTrainer:
    """
    PPO 训练器
    
    特性:
    - Clipped surrogate objective
    - GAE (Generalized Advantage Estimation)
    - Mini-batch 更新
    - 早停机制
    
    Args:
        model: GATPolicy 模型
        lr: 学习率
        gamma: 折扣因子
        gae_lambda: GAE lambda 参数
        clip_eps: PPO clipping epsilon
        value_coef: 价值损失系数
        entropy_coef: 熵正则化系数
        max_grad_norm: 梯度裁剪阈值
        n_epochs: 每次更新的 epoch 数
        batch_size: Mini-batch 大小
    """
    
    def __init__(
        self,
        model,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64,
    ):
        self.model = model.to(DEVICE)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.buffer = RolloutBuffer()
        
    def collect_trajectory(self, G, k, reward_fn, node_features=None, embed_dim=128):
        """
        收集单个图的轨迹数据
        
        使用 Node2Vec 嵌入作为节点特征。
        
        Args:
            G: NetworkX 图
            k: 要选择的控制器数量
            reward_fn: 奖励函数 reward_fn(G, centers) -> float
            node_features: 预计算的节点特征（可选）
            embed_dim: Node2Vec 嵌入维度 (默认128)
            
        Returns:
            centers: 选择的控制器列表
            total_reward: 总奖励
        """
        self.model.eval()
        
        # 获取 Node2Vec 节点特征
        if node_features is None:
            x, node_list = get_gnn_node_features(G, device=DEVICE, embed_dim=embed_dim)
        else:
            x = node_features.to(DEVICE)
            node_list = list(G.nodes())
        
        # 转换图结构
        _, edge_index, _ = graph_to_pyg_data(G, device=DEVICE)
        
        num_nodes = len(node_list)
        selected_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
        centers = []
        experiences = []
        
        with torch.no_grad():
            for step in range(k):
                # 获取动作
                action, log_prob, value, probs = self.model.get_action(
                    x, edge_index, selected_mask=selected_mask, deterministic=False
                )
                
                # 执行动作
                action_idx = action.item()
                center = node_list[action_idx]
                centers.append(center)
                
                # 计算 step-wise 奖励（使用增量奖励）
                if step < k - 1:
                    # 中间步骤使用小的增量奖励
                    step_reward = 0.01  # 存活奖励
                else:
                    # 最后一步计算完整奖励
                    step_reward = reward_fn(G, centers)
                
                # 保存经验
                done = (step == k - 1)
                exp = Experience(
                    x=x.clone(),
                    edge_index=edge_index.clone(),
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=step_reward,
                    selected_mask=selected_mask.clone(),
                    done=done,
                )
                experiences.append(exp)
                
                # 更新 mask
                selected_mask = selected_mask.clone()
                selected_mask[action_idx] = True
        
        # 计算最终奖励
        final_reward = reward_fn(G, centers)
        
        # 使用最终奖励更新最后一个经验的奖励
        if experiences:
            exp = experiences[-1]
            experiences[-1] = Experience(
                x=exp.x,
                edge_index=exp.edge_index,
                action=exp.action,
                log_prob=exp.log_prob,
                value=exp.value,
                reward=final_reward,
                selected_mask=exp.selected_mask,
                done=exp.done,
            )
        
        # 添加到缓冲区
        for exp in experiences:
            self.buffer.add(exp)
        self.buffer.add_episode_reward(final_reward)
        
        return centers, final_reward
    
    def compute_gae(self, rewards, values, dones):
        """
        计算 Generalized Advantage Estimation
        
        Args:
            rewards: 奖励列表
            values: 价值估计列表
            dones: 完成标志列表
            
        Returns:
            advantages: 优势函数值
            returns: 回报值
        """
        advantages = []
        returns = []
        gae = 0
        
        # 反向计算
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1] if not dones[t] else 0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self):
        """
        执行 PPO 更新
        
        Returns:
            dict: 训练统计信息
        """
        if len(self.buffer) == 0:
            return {}
        
        self.model.train()
        
        experiences = self.buffer.experiences
        
        # 提取数据
        rewards = [exp.reward for exp in experiences]
        values = [exp.value.item() for exp in experiences]
        dones = [exp.done for exp in experiences]
        
        # 计算 GAE
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # 转换为张量
        advantages = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        
        # 归一化优势函数
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 收集旧的 log_probs
        old_log_probs = torch.stack([exp.log_prob for exp in experiences])
        
        # Mini-batch 更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        indices = list(range(len(experiences)))
        
        for epoch in range(self.n_epochs):
            random.shuffle(indices)
            
            for start in range(0, len(indices), self.batch_size):
                end = min(start + self.batch_size, len(indices))
                batch_indices = indices[start:end]
                
                # 批量处理
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_entropy = 0
                
                for idx in batch_indices:
                    exp = experiences[idx]
                    
                    # 重新计算概率和价值
                    probs, value = self.model(
                        exp.x, exp.edge_index, selected_mask=exp.selected_mask
                    )
                    
                    dist = torch.distributions.Categorical(probs)
                    new_log_prob = dist.log_prob(exp.action)
                    entropy = dist.entropy()
                    
                    # 计算比率
                    ratio = torch.exp(new_log_prob - old_log_probs[idx].detach())
                    
                    # Clipped surrogate loss
                    adv = advantages[idx]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2)
                    
                    # Value loss
                    value_loss = F.mse_loss(value.squeeze(), returns[idx])
                    
                    batch_policy_loss += policy_loss
                    batch_value_loss += value_loss
                    batch_entropy += entropy
                
                # 平均损失
                batch_size = len(batch_indices)
                batch_policy_loss /= batch_size
                batch_value_loss /= batch_size
                batch_entropy /= batch_size
                
                # 总损失
                loss = (
                    batch_policy_loss 
                    + self.value_coef * batch_value_loss 
                    - self.entropy_coef * batch_entropy
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += batch_policy_loss.item()
                total_value_loss += batch_value_loss.item()
                total_entropy += batch_entropy.item()
                num_updates += 1
        
        # 清空缓冲区
        avg_reward = np.mean(self.buffer.episode_rewards) if self.buffer.episode_rewards else 0
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
            'avg_reward': avg_reward,
        }


def train_gnn_ppo(
    graphs,
    epochs=100,
    save_path=None,
    reward_fn=None,
    lr=3e-4,
    in_channels=128,        # Node2Vec 嵌入维度
    hidden_channels=128,    # 隐藏层维度
    heads=8,                # GAT 注意力头数
    num_layers=5,           # GAT 层数（深层）
    k_ratio=0.1,
    collect_per_epoch=20,
    verbose=True,
    resume_from=None,       # 增量训练：从此 checkpoint 加载继续训练
):
    """
    使用深层 GAT + PPO 训练控制器选择策略
    
    使用 Node2Vec 嵌入作为节点特征，全部由 GAT+RL 选择控制器。
    支持 resume_from 增量训练：加载已有 checkpoint 在新数据上继续训练。
    
    Args:
        graphs: 训练图列表
        epochs: 训练轮数
        save_path: 模型保存路径
        reward_fn: 奖励函数，如果为 None 则使用默认的 GCC 奖励
        lr: 学习率
        in_channels: Node2Vec 嵌入维度 (默认128)
        hidden_channels: 隐藏层维度 (默认128)
        heads: GAT 注意力头数 (默认8)
        num_layers: GAT 层数 (默认5)
        k_ratio: 控制器比例
        collect_per_epoch: 每轮收集的轨迹数
        verbose: 是否显示进度
        resume_from: 增量训练时加载的 checkpoint 路径（与当前架构需一致）
        
    Returns:
        dict: 训练结果
    """
    import os
    from .reward import calculate_gcc_focused_reward
    
    if reward_fn is None:
        reward_fn = calculate_gcc_focused_reward
    
    # 创建深层 GAT 模型（或从 checkpoint 加载）
    model = GATPolicy(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        heads=heads,
        num_layers=num_layers,
    ).to(DEVICE)
    
    history = []
    best_reward = -float('inf')
    best_model_state = None
    start_epoch = 0
    
    if resume_from and os.path.exists(resume_from):
        try:
            ck = torch.load(resume_from, map_location=DEVICE, weights_only=False)
            ck_in = ck.get('in_channels', in_channels)
            ck_hidden = ck.get('hidden_channels', hidden_channels)
            ck_heads = ck.get('heads', heads)
            ck_layers = ck.get('num_layers', num_layers)
            if (ck_in, ck_hidden, ck_heads, ck_layers) != (in_channels, hidden_channels, heads, num_layers):
                if verbose:
                    print(f"Warning: checkpoint 架构与当前不一致，使用当前架构重新初始化")
            else:
                model.load_state_dict(ck['model_state_dict'], strict=False)
                history = ck.get('history', [])
                best_reward = ck.get('best_reward', -float('inf'))
                best_model_state = ck.get('model_state_dict')
                if best_model_state is not None:
                    best_model_state = dict(best_model_state)
                start_epoch = len(history)
                if verbose:
                    print(f"增量训练: 从 {resume_from} 加载，已训练 {start_epoch} 轮，best_reward={best_reward:.4f}")
        except Exception as e:
            if verbose:
                print(f"加载 checkpoint 失败: {e}，从头训练")
    
    if verbose and start_epoch == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"Architecture: {num_layers}-layer GAT, {heads} heads, {hidden_channels} hidden dim")
        print(f"Input: {in_channels}D Node2Vec embeddings")
    
    # 创建训练器
    trainer = PPOTrainer(model, lr=lr)
    
    # 训练循环（从 start_epoch 继续）
    epoch_pbar = tqdm(range(start_epoch, start_epoch + epochs), desc="Training Deep GAT+PPO", disable=not verbose)
    
    for epoch in epoch_pbar:
        set_current_epoch(epoch, start_epoch + epochs)
        epoch_rewards = []
        
        # 随机采样图进行训练
        sampled_graphs = random.sample(graphs, min(collect_per_epoch, len(graphs)))
        
        for G in sampled_graphs:
            if G.number_of_nodes() < 3:
                continue
            
            # 计算控制器数量
            k = max(1, int(G.number_of_nodes() * k_ratio))
            
            # 收集轨迹（使用 Node2Vec 嵌入）
            try:
                centers, reward = trainer.collect_trajectory(G, k, reward_fn, embed_dim=in_channels)
                epoch_rewards.append(reward)
            except Exception as e:
                if verbose and epoch == 0:
                    print(f"Warning: {e}")
                continue
        
        # PPO 更新
        stats = trainer.update()
        
        # 记录
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        history.append([len(history) + 1, avg_reward])
        
        # 保存最佳模型
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model_state = model.state_dict().copy()
        
        # 更新进度条
        epoch_pbar.set_postfix({
            'reward': f'{avg_reward:.4f}',
            'best': f'{best_reward:.4f}',
            'policy_loss': f'{stats.get("policy_loss", 0):.4f}',
        })
    
    # 保存模型
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': best_model_state if best_model_state else model.state_dict(),
            'model_type': 'GATPolicy',
            'in_channels': in_channels,
            'hidden_channels': hidden_channels,
            'heads': heads,
            'num_layers': num_layers,
            'best_reward': best_reward,
            'history': history,
        }
        torch.save(checkpoint, save_path)
        
        if verbose:
            print(f"\nModel saved to {save_path}")
            print(f"Best reward: {best_reward:.4f}")
    
    return {
        'history': history,
        'best_reward': best_reward,
        'final_reward': history[-1][1] if history else 0,
        'model_path': save_path,
    }


# ============================================================
# 统一模型训练函数（支持课程学习和规模平衡采样）
# ============================================================

class UnifiedPPOTrainer:
    """
    统一模型的PPO训练器
    
    支持:
    - UnifiedGATPolicy 模型
    - 规模编码
    - 规模自适应特征提取
    """
    
    def __init__(
        self,
        model,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=32,
    ):
        self.model = model.to(DEVICE)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.buffer = RolloutBuffer()
    
    def collect_trajectory(self, G, k, reward_fn, use_unified_features=True):
        """
        收集单个图的轨迹数据（使用统一特征）
        
        Args:
            G: NetworkX 图
            k: 要选择的控制器数量
            reward_fn: 奖励函数
            use_unified_features: 是否使用统一特征提取
            
        Returns:
            centers: 选择的控制器列表
            total_reward: 总奖励
        """
        from .gnn_model import (
            get_unified_features, 
            graph_to_pyg_data,
            get_scale_encoding
        )
        
        self.model.eval()
        
        # 获取统一特征
        if use_unified_features:
            node_features, scale_encoding, node_list, raw_dim = get_unified_features(G, device=DEVICE)
            
            # 使用模型的特征投影器投影到统一维度
            with torch.no_grad():
                x = self.model.feature_projector(node_features, raw_dim)
        else:
            # 回退到旧方式
            x, node_list = get_gnn_node_features(G, device=DEVICE, embed_dim=128)
            scale_encoding = get_scale_encoding(G, dim=8, device=DEVICE)
        
        # 转换图结构
        _, edge_index, _ = graph_to_pyg_data(G, device=DEVICE)
        
        num_nodes = len(node_list)
        selected_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
        centers = []
        experiences = []
        
        with torch.no_grad():
            for step in range(k):
                # 获取动作
                action, log_prob, value, probs = self.model.get_action(
                    x, edge_index, scale_encoding=scale_encoding,
                    selected_mask=selected_mask, deterministic=False
                )
                
                # 执行动作
                action_idx = action.item()
                center = node_list[action_idx]
                centers.append(center)
                
                # Step-wise 奖励
                if step < k - 1:
                    step_reward = 0.01  # 存活奖励
                else:
                    step_reward = reward_fn(G, centers)
                
                # 保存经验（包含scale_encoding）
                done = (step == k - 1)
                exp = UnifiedExperience(
                    x=x.clone(),
                    edge_index=edge_index.clone(),
                    scale_encoding=scale_encoding.clone(),
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=step_reward,
                    selected_mask=selected_mask.clone(),
                    done=done,
                )
                experiences.append(exp)
                
                # 更新 mask
                selected_mask = selected_mask.clone()
                selected_mask[action_idx] = True
        
        # 计算最终奖励
        final_reward = reward_fn(G, centers)
        
        # 更新最后一个经验的奖励
        if experiences:
            exp = experiences[-1]
            experiences[-1] = UnifiedExperience(
                x=exp.x,
                edge_index=exp.edge_index,
                scale_encoding=exp.scale_encoding,
                action=exp.action,
                log_prob=exp.log_prob,
                value=exp.value,
                reward=final_reward,
                selected_mask=exp.selected_mask,
                done=exp.done,
            )
        
        # 添加到缓冲区
        for exp in experiences:
            self.buffer.add(exp)
        self.buffer.add_episode_reward(final_reward)
        
        return centers, final_reward
    
    def compute_gae(self, rewards, values, dones):
        """计算 GAE"""
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1] if not dones[t] else 0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self):
        """执行 PPO 更新"""
        if len(self.buffer) == 0:
            return {}
        
        self.model.train()
        
        experiences = self.buffer.experiences
        
        # 提取数据
        rewards = [exp.reward for exp in experiences]
        values = [exp.value.item() for exp in experiences]
        dones = [exp.done for exp in experiences]
        
        # 计算 GAE
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # 转换为张量
        advantages = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        
        # 归一化优势函数
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        old_log_probs = torch.stack([exp.log_prob for exp in experiences])
        
        # Mini-batch 更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        indices = list(range(len(experiences)))
        
        for epoch in range(self.n_epochs):
            random.shuffle(indices)
            
            for start in range(0, len(indices), self.batch_size):
                end = min(start + self.batch_size, len(indices))
                batch_indices = indices[start:end]
                
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_entropy = 0
                
                for idx in batch_indices:
                    exp = experiences[idx]
                    
                    # 重新计算概率和价值（包含scale_encoding）
                    probs, value = self.model(
                        exp.x, exp.edge_index, 
                        scale_encoding=exp.scale_encoding,
                        selected_mask=exp.selected_mask
                    )
                    
                    dist = torch.distributions.Categorical(probs)
                    new_log_prob = dist.log_prob(exp.action)
                    entropy = dist.entropy()
                    
                    ratio = torch.exp(new_log_prob - old_log_probs[idx].detach())
                    
                    adv = advantages[idx]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2)
                    
                    value_loss = F.mse_loss(value.squeeze(), returns[idx])
                    
                    batch_policy_loss += policy_loss
                    batch_value_loss += value_loss
                    batch_entropy += entropy
                
                batch_size = len(batch_indices)
                batch_policy_loss /= batch_size
                batch_value_loss /= batch_size
                batch_entropy /= batch_size
                
                loss = (
                    batch_policy_loss 
                    + self.value_coef * batch_value_loss 
                    - self.entropy_coef * batch_entropy
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += batch_policy_loss.item()
                total_value_loss += batch_value_loss.item()
                total_entropy += batch_entropy.item()
                num_updates += 1
        
        avg_reward = np.mean(self.buffer.episode_rewards) if self.buffer.episode_rewards else 0
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
            'avg_reward': avg_reward,
        }


# 统一经验数据结构（包含scale_encoding）
from collections import namedtuple
UnifiedExperience = namedtuple('UnifiedExperience', [
    'x',              # 节点特征
    'edge_index',     # 边索引
    'scale_encoding', # 规模编码
    'action',         # 选择的节点
    'log_prob',       # 动作的对数概率
    'value',          # 状态价值估计
    'reward',         # 奖励
    'selected_mask',  # 已选节点 mask
    'done',           # 是否完成
])


def weighted_sample_graphs(graphs, focus_range=None, focus_weight=2.0, sample_size=20):
    """
    加权采样图
    
    对focus_range内的图赋予更高的采样权重。
    
    Args:
        graphs: 图列表
        focus_range: 重点关注的节点数范围 (min_nodes, max_nodes)
        focus_weight: 重点范围的权重倍数
        sample_size: 采样数量
        
    Returns:
        sampled_graphs: 采样的图列表
    """
    if not graphs:
        return []
    
    if focus_range is None:
        # 无重点范围，均匀采样
        return random.sample(graphs, min(sample_size, len(graphs)))
    
    min_focus, max_focus = focus_range
    
    # 计算每个图的权重
    weights = []
    for g in graphs:
        n = g.number_of_nodes()
        if min_focus <= n <= max_focus:
            weights.append(focus_weight)
        else:
            weights.append(1.0)
    
    # 归一化权重
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]
    
    # 加权采样
    sample_size = min(sample_size, len(graphs))
    sampled_indices = np.random.choice(
        len(graphs), 
        size=sample_size, 
        replace=False, 
        p=probs
    )
    
    return [graphs[i] for i in sampled_indices]


def curriculum_sample_graphs(graphs, epoch, total_epochs, sample_size=20):
    """
    课程学习采样
    
    训练前期专注小网络，后期逐渐扩展到大网络。
    
    Args:
        graphs: 图列表
        epoch: 当前epoch
        total_epochs: 总epoch数
        sample_size: 采样数量
        
    Returns:
        sampled_graphs: 采样的图列表
    """
    if not graphs:
        return []
    
    # 按节点数排序
    sorted_graphs = sorted(graphs, key=lambda g: g.number_of_nodes())
    
    # 计算当前阶段应该使用的图范围
    progress = epoch / total_epochs
    
    if progress < 0.33:
        # 前1/3: 只使用小图（前40%的图）
        end_idx = max(1, int(len(sorted_graphs) * 0.4))
        available_graphs = sorted_graphs[:end_idx]
    elif progress < 0.67:
        # 中1/3: 使用小图和中图（前70%的图）
        end_idx = max(1, int(len(sorted_graphs) * 0.7))
        available_graphs = sorted_graphs[:end_idx]
    else:
        # 后1/3: 使用全部图
        available_graphs = sorted_graphs
    
    return random.sample(available_graphs, min(sample_size, len(available_graphs)))


def scale_balanced_sample_graphs(graphs, sample_size=20, max_nodes=1000):
    """
    规模平衡采样
    
    确保 20-max_nodes 范围内不同规模的图有相近的采样概率。
    支持 20-1000 全范围，5 个规模档位均等采样。
    
    Args:
        graphs: 图列表
        sample_size: 采样数量
        max_nodes: 最大节点数(用于确定档位边界，默认1000)
        
    Returns:
        sampled_graphs: 采样的图列表
    """
    if not graphs:
        return []
    
    # 按规模分组 (20-1000 全范围，5档均等)
    scale_bins = {
        'tiny': [],       # 20-50
        'small': [],      # 50-100
        'medium': [],     # 100-200
        'large': [],      # 200-500
        'xlarge': [],     # 500-1000
    }
    
    for g in graphs:
        n = g.number_of_nodes()
        if n < 50:
            scale_bins['tiny'].append(g)
        elif n < 100:
            scale_bins['small'].append(g)
        elif n < 200:
            scale_bins['medium'].append(g)
        elif n < 500:
            scale_bins['large'].append(g)
        else:
            scale_bins['xlarge'].append(g)
    
    # 从每个非空组采样
    non_empty_bins = {k: v for k, v in scale_bins.items() if v}
    if not non_empty_bins:
        return []
    
    # 每个组的采样数量
    per_bin_size = max(1, sample_size // len(non_empty_bins))
    
    sampled = []
    for bin_graphs in non_empty_bins.values():
        n_sample = min(per_bin_size, len(bin_graphs))
        sampled.extend(random.sample(bin_graphs, n_sample))
    
    # 如果还没够，随机补充
    remaining = sample_size - len(sampled)
    if remaining > 0:
        available = [g for g in graphs if g not in sampled]
        if available:
            sampled.extend(random.sample(available, min(remaining, len(available))))
    
    return sampled


def train_unified_gnn_ppo(
    graphs,
    epochs=200,
    save_path=None,
    reward_fn=None,
    lr=3e-4,
    in_channels=64,              # 统一的Node2Vec投影维度
    hidden_channels=96,          # 隐藏层维度
    scale_encoding_dim=8,        # 规模编码维度
    heads=4,                     # GAT 注意力头数
    num_layers=3,                # GAT 层数
    k_ratio=0.1,
    collect_per_epoch=30,
    use_curriculum=True,         # 启用课程学习
    use_scale_balance=True,      # 启用规模平衡采样
    focus_range=(50, 150),       # 重点关注范围
    focus_weight=2.0,            # 重点范围权重
    verbose=True,
    resume_from=None,
):
    """
    使用统一GAT模型 + PPO 训练控制器选择策略
    
    支持:
    - 规模自适应Node2Vec特征提取
    - 规模编码
    - 课程学习
    - 规模平衡采样
    - 增量训练
    
    Args:
        graphs: 训练图列表
        epochs: 训练轮数
        save_path: 模型保存路径
        reward_fn: 奖励函数
        lr: 学习率
        in_channels: Node2Vec投影后的维度
        hidden_channels: 隐藏层维度
        scale_encoding_dim: 规模编码维度
        heads: GAT注意力头数
        num_layers: GAT层数
        k_ratio: 控制器比例
        collect_per_epoch: 每轮收集的轨迹数
        use_curriculum: 是否使用课程学习
        use_scale_balance: 是否使用规模平衡采样
        focus_range: 重点关注的节点数范围
        focus_weight: 重点范围的采样权重
        verbose: 是否显示进度
        resume_from: 增量训练的checkpoint路径
        
    Returns:
        dict: 训练结果
    """
    import os
    from .gnn_model import UnifiedGATPolicy
    from .reward import calculate_unified_gcc_reward
    
    if reward_fn is None:
        reward_fn = calculate_unified_gcc_reward
    
    # 创建统一GAT模型
    model = UnifiedGATPolicy(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        scale_encoding_dim=scale_encoding_dim,
        heads=heads,
        num_layers=num_layers,
    ).to(DEVICE)
    
    history = []
    best_reward = -float('inf')
    best_model_state = None
    start_epoch = 0
    
    # 尝试加载checkpoint
    if resume_from and os.path.exists(resume_from):
        try:
            ck = torch.load(resume_from, map_location=DEVICE, weights_only=False)
            
            # 检查架构一致性
            ck_config = ck.get('config', {})
            current_config = {
                'in_channels': in_channels,
                'hidden_channels': hidden_channels,
                'scale_encoding_dim': scale_encoding_dim,
                'heads': heads,
                'num_layers': num_layers,
            }
            
            if ck_config == current_config or not ck_config:
                model.load_state_dict(ck['model_state_dict'], strict=False)
                history = ck.get('history', [])
                best_reward = ck.get('best_reward', -float('inf'))
                best_model_state = ck.get('model_state_dict')
                if best_model_state is not None:
                    best_model_state = dict(best_model_state)
                start_epoch = len(history)
                if verbose:
                    print(f"增量训练: 从 {resume_from} 加载，已训练 {start_epoch} 轮，best_reward={best_reward:.4f}")
            else:
                if verbose:
                    print(f"Warning: checkpoint 架构与当前不一致，从头训练")
        except Exception as e:
            if verbose:
                print(f"加载 checkpoint 失败: {e}，从头训练")
    
    if verbose and start_epoch == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"Architecture: UnifiedGATPolicy, {num_layers}-layer GAT, {heads} heads, {hidden_channels} hidden dim")
        print(f"Input: {in_channels}D projected features + {scale_encoding_dim}D scale encoding")
        print(f"Training strategy: curriculum={use_curriculum}, scale_balance={use_scale_balance}")
        if focus_range:
            print(f"Focus range: {focus_range[0]}-{focus_range[1]} nodes (weight={focus_weight})")
    
    # 创建训练器
    trainer = UnifiedPPOTrainer(model, lr=lr)
    
    # 学习率调度器: 奖励停滞时降低学习率，促进收敛
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode='max', factor=0.7, patience=25, min_lr=1e-5
    )
    
    # 分析训练数据分布 (20-1000 全范围)
    if verbose:
        scale_dist = {'tiny(20-50)': 0, 'small(50-100)': 0, 'medium(100-200)': 0,
                      'large(200-500)': 0, 'xlarge(500-1000)': 0}
        for g in graphs:
            n = g.number_of_nodes()
            if n < 50:
                scale_dist['tiny(20-50)'] += 1
            elif n < 100:
                scale_dist['small(50-100)'] += 1
            elif n < 200:
                scale_dist['medium(100-200)'] += 1
            elif n < 500:
                scale_dist['large(200-500)'] += 1
            else:
                scale_dist['xlarge(500-1000)'] += 1
        print(f"Training data distribution: {scale_dist}")
    
    # 训练循环
    total_epochs = start_epoch + epochs
    epoch_pbar = tqdm(range(start_epoch, total_epochs), desc="Training UnifiedGAT+PPO", disable=not verbose)
    
    for epoch in epoch_pbar:
        set_current_epoch(epoch, total_epochs)
        epoch_rewards = []
        
        # 选择采样策略
        if use_curriculum:
            # 课程学习采样
            sampled_graphs = curriculum_sample_graphs(
                graphs, epoch - start_epoch, epochs, collect_per_epoch
            )
        elif use_scale_balance:
            # 规模平衡采样
            sampled_graphs = scale_balanced_sample_graphs(graphs, collect_per_epoch)
        elif focus_range:
            # 加权采样
            sampled_graphs = weighted_sample_graphs(
                graphs, focus_range, focus_weight, collect_per_epoch
            )
        else:
            # 随机采样
            sampled_graphs = random.sample(graphs, min(collect_per_epoch, len(graphs)))
        
        for G in sampled_graphs:
            if G.number_of_nodes() < 3:
                continue
            
            # 计算控制器数量
            k = max(1, int(G.number_of_nodes() * k_ratio))
            
            # 收集轨迹
            try:
                centers, reward = trainer.collect_trajectory(G, k, reward_fn)
                epoch_rewards.append(reward)
            except Exception as e:
                if verbose and epoch == start_epoch:
                    print(f"Warning: {e}")
                continue
        
        # PPO 更新
        stats = trainer.update()
        
        # 记录
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        history.append([len(history) + 1, avg_reward])
        
        # 保存最佳模型
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model_state = model.state_dict().copy()
        
        # 学习率调度: 奖励停滞时降低学习率，促进收敛
        lr_scheduler.step(avg_reward)
        
        # 更新进度条
        epoch_pbar.set_postfix({
            'reward': f'{avg_reward:.4f}',
            'best': f'{best_reward:.4f}',
            'loss': f'{stats.get("policy_loss", 0):.4f}',
        })
    
    # 保存模型
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        checkpoint = {
            'model_state_dict': best_model_state if best_model_state else model.state_dict(),
            'model_type': 'UnifiedGATPolicy',
            'config': {
                'in_channels': in_channels,
                'hidden_channels': hidden_channels,
                'scale_encoding_dim': scale_encoding_dim,
                'heads': heads,
                'num_layers': num_layers,
            },
            'best_reward': best_reward,
            'history': history,
            'training_config': {
                'use_curriculum': use_curriculum,
                'use_scale_balance': use_scale_balance,
                'focus_range': focus_range,
                'focus_weight': focus_weight,
            },
        }
        torch.save(checkpoint, save_path)
        
        if verbose:
            print(f"\nModel saved to {save_path}")
            print(f"Best reward: {best_reward:.4f}")
    
    return {
        'history': history,
        'best_reward': best_reward,
        'final_reward': history[-1][1] if history else 0,
        'model_path': save_path,
    }
