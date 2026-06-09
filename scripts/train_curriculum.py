# -*- coding: utf-8 -*-
"""
BiT-HyRL 课程学习训练脚本

实现四阶段课程学习策略：
- 阶段1：只奖励连通性（GCC），在小图上训练
- 阶段2：加入覆盖率奖励，扩展到中等规模图
- 阶段3：加入分散度和抗攻击奖励，使用全规模图
- 阶段4：真实网络增强训练，使用真实网络数据微调

支持三种模型架构：
- DynamicGAT: 带动态覆盖特征的 GAT
- GraphTransformer: 使用全图注意力的 Transformer
- UnifiedGAT: 原有的统一 GAT 模型

Usage:
    # 阶段1: 小图 + GCC 奖励
    python scripts/train_curriculum.py --phase 1 --epochs 100 --max-nodes 50
    
    # 阶段2: 中等图 + 覆盖率奖励（从阶段1继续）
    python scripts/train_curriculum.py --phase 2 --epochs 150 --max-nodes 150 --resume src/train/v1/checkpoints/curriculum_phase1.pth
    
    # 阶段3: 全规模图 + 完整奖励（从阶段2继续）
    python scripts/train_curriculum.py --phase 3 --epochs 200 --max-nodes 500 --resume src/train/v1/checkpoints/curriculum_phase2.pth
    
    # 阶段4: 真实网络增强训练（从阶段3继续）
    python scripts/train_curriculum.py --phase 4 --epochs 100 --data-dir dataset/all/real --resume src/train/v1/checkpoints/curriculum_phase3_dynamic_gat.pth
    
    # 一次性运行全部阶段（包括阶段4）
    python scripts/train_curriculum.py --full-curriculum --total-epochs 550 --include-phase4
"""

import os
import sys
import argparse
import random
import glob
import time

import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import namedtuple

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.bit_hyrl import config
from src.bit_hyrl.gnn_model import (
    DynamicGATPolicy,
    GraphTransformerPolicy,
    UnifiedGATPolicy,
    FeatureProjector,
    compute_coverage_features,
    compute_dispersion_features,
    get_unified_features,
    get_scale_encoding,
    graph_to_pyg_data,
    get_laplacian_pe,
)
from src.bit_hyrl.reward import (
    get_curriculum_reward_fn,
    calculate_curriculum_reward,
    CurriculumRewardPhase,
)
from src.topology.generators import load_graph as _load_graph_file

DEVICE = config.DEVICE


# ============================================================
# 数据加载
# ============================================================

def load_training_graphs(data_dir, min_nodes=20, max_nodes=500, max_graphs=None):
    """
    加载训练图
    
    Args:
        data_dir: 数据目录
        min_nodes: 最小节点数
        max_nodes: 最大节点数
        max_graphs: 最大图数量
        
    Returns:
        graphs: 图列表
    """
    graphs = []
    
    # 查找所有图文件
    gml_files = glob.glob(os.path.join(data_dir, '**', '*.gml'), recursive=True)
    graphml_files = glob.glob(os.path.join(data_dir, '**', '*.graphml'), recursive=True)
    all_files = gml_files + graphml_files
    
    print(f"Found {len(all_files)} graph files")
    
    for fpath in tqdm(all_files, desc="Loading graphs"):
        try:
            G, _ = _load_graph_file(fpath, verbose=False)
            if G is None:
                continue
            
            n = G.number_of_nodes()
            if min_nodes <= n <= max_nodes:
                # 确保图是连通的（使用最大连通分量）
                if not nx.is_connected(G):
                    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
                
                if G.number_of_nodes() >= min_nodes:
                    graphs.append(G)
                    
            if max_graphs and len(graphs) >= max_graphs:
                break
                
        except Exception as e:
            continue
    
    print(f"Loaded {len(graphs)} graphs (nodes: {min_nodes}-{max_nodes})")
    
    return graphs


def filter_graphs_by_size(graphs, min_nodes, max_nodes):
    """按大小过滤图"""
    filtered = [g for g in graphs if min_nodes <= g.number_of_nodes() <= max_nodes]
    return filtered


def curriculum_sample_graphs(graphs, epoch, phase_epochs, sample_size=30):
    """
    课程学习采样
    
    在每个阶段内，也从小图逐渐过渡到大图
    """
    if not graphs:
        return []
    
    sorted_graphs = sorted(graphs, key=lambda g: g.number_of_nodes())
    progress = min(epoch / phase_epochs, 1.0)
    
    if progress < 0.33:
        # 前1/3：只用小图
        end_idx = max(1, int(len(sorted_graphs) * 0.4))
        available = sorted_graphs[:end_idx]
    elif progress < 0.67:
        # 中1/3：小图+中图
        end_idx = max(1, int(len(sorted_graphs) * 0.7))
        available = sorted_graphs[:end_idx]
    else:
        # 后1/3：全部
        available = sorted_graphs
    
    return random.sample(available, min(sample_size, len(available)))


# ============================================================
# 经验数据结构
# ============================================================

DynamicExperience = namedtuple('DynamicExperience', [
    'x',                # 静态节点特征
    'edge_index',       # 边索引
    'scale_encoding',   # 规模编码
    'dynamic_features', # 动态特征
    'laplacian_pe',     # 拉普拉斯位置编码（可选）
    'action',           # 选择的节点
    'log_prob',         # 动作对数概率
    'value',            # 状态价值
    'reward',           # 奖励
    'selected_mask',    # 已选 mask
    'done',             # 是否完成
])


class DynamicRolloutBuffer:
    """动态特征轨迹缓冲区"""
    
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


# ============================================================
# 课程学习 PPO 训练器
# ============================================================

class CurriculumPPOTrainer:
    """
    支持课程学习和动态特征的 PPO 训练器
    """
    
    def __init__(
        self,
        model,
        model_type='dynamic_gat',  # 'dynamic_gat', 'graph_transformer', 'unified_gat'
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
        self.model_type = model_type
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.buffer = DynamicRolloutBuffer()
        
        # 特征投影器（用于不同维度的 Node2Vec）
        self.feature_projector = model.feature_projector if hasattr(model, 'feature_projector') else None
    
    def collect_trajectory(self, G, k, reward_fn, use_dynamic_features=True):
        """
        收集轨迹（支持动态特征更新）
        
        Args:
            G: NetworkX 图
            k: 控制器数量
            reward_fn: 奖励函数
            use_dynamic_features: 是否使用动态特征
            
        Returns:
            centers: 选择的控制器列表
            total_reward: 总奖励
        """
        self.model.eval()
        
        # 获取基础特征
        node_features, scale_encoding, node_list, raw_dim = get_unified_features(G, device=DEVICE)
        
        # 特征投影（如果需要）
        if self.feature_projector is not None:
            with torch.no_grad():
                x = self.feature_projector(node_features, raw_dim)
        else:
            x = node_features
        
        # 获取边索引
        _, edge_index, _ = graph_to_pyg_data(G, device=DEVICE)
        
        # 拉普拉斯位置编码（仅 Graph Transformer 使用）
        laplacian_pe = None
        if self.model_type == 'graph_transformer':
            hidden_dim = self.model.hidden_channels
            laplacian_pe = get_laplacian_pe(G, hidden_dim=hidden_dim, device=DEVICE)
        
        num_nodes = len(node_list)
        selected_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
        centers = []
        experiences = []
        
        with torch.no_grad():
            for step in range(k):
                # 计算动态特征（每一步更新）
                if use_dynamic_features:
                    coverage_feat = compute_coverage_features(G, centers, node_list, device=DEVICE)
                    dispersion_feat = compute_dispersion_features(G, centers, node_list, device=DEVICE)
                    dynamic_features = torch.cat([coverage_feat, dispersion_feat], dim=-1)
                else:
                    dynamic_features = torch.zeros((num_nodes, 6), device=DEVICE)
                
                # 获取动作
                if self.model_type == 'graph_transformer':
                    action, log_prob, value, probs = self.model.get_action(
                        x, edge_index, scale_encoding, dynamic_features, laplacian_pe,
                        selected_mask=selected_mask, deterministic=False
                    )
                else:
                    action, log_prob, value, probs = self.model.get_action(
                        x, edge_index, scale_encoding, dynamic_features,
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
                
                # 保存经验
                done = (step == k - 1)
                exp = DynamicExperience(
                    x=x.clone(),
                    edge_index=edge_index.clone(),
                    scale_encoding=scale_encoding.clone(),
                    dynamic_features=dynamic_features.clone(),
                    laplacian_pe=laplacian_pe.clone() if laplacian_pe is not None else None,
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
        
        # 最终奖励
        final_reward = reward_fn(G, centers)
        
        # 更新最后一个经验的奖励
        if experiences:
            exp = experiences[-1]
            experiences[-1] = DynamicExperience(
                x=exp.x,
                edge_index=exp.edge_index,
                scale_encoding=exp.scale_encoding,
                dynamic_features=exp.dynamic_features,
                laplacian_pe=exp.laplacian_pe,
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
        """PPO 更新"""
        if len(self.buffer) == 0:
            return {}
        
        self.model.train()
        
        experiences = self.buffer.experiences
        
        rewards = [exp.reward for exp in experiences]
        values = [exp.value.item() for exp in experiences]
        dones = [exp.done for exp in experiences]
        
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        old_log_probs = torch.stack([exp.log_prob for exp in experiences])
        
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
                    
                    # 根据模型类型调用 forward
                    if self.model_type == 'graph_transformer':
                        probs, value = self.model(
                            exp.x, exp.edge_index, exp.scale_encoding,
                            exp.dynamic_features, exp.laplacian_pe,
                            selected_mask=exp.selected_mask
                        )
                    else:
                        probs, value = self.model(
                            exp.x, exp.edge_index, exp.scale_encoding,
                            exp.dynamic_features,
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
                    
                    value_loss = torch.nn.functional.mse_loss(value.squeeze(), returns[idx])
                    
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


# ============================================================
# 训练函数
# ============================================================

def train_single_phase(
    graphs,
    phase,
    epochs,
    save_path,
    model_type='dynamic_gat',
    resume_from=None,
    lr=3e-4,
    k_ratio=0.1,
    collect_per_epoch=30,
    verbose=True,
):
    """
    训练单个阶段
    
    Args:
        graphs: 训练图列表
        phase: 课程学习阶段 (1, 2, 3)
        epochs: 训练轮数
        save_path: 模型保存路径
        model_type: 模型类型
        resume_from: 继续训练的模型路径
        lr: 学习率
        k_ratio: 控制器比例
        collect_per_epoch: 每轮采样图数量
        verbose: 是否显示进度
        
    Returns:
        dict: 训练结果
    """
    # 获取阶段奖励函数
    reward_fn = get_curriculum_reward_fn(phase)
    
    # 模型配置
    model_config = {
        'in_channels': 64,
        'hidden_channels': 96,
        'scale_encoding_dim': 8,
        'dynamic_feature_dim': 6,
        'num_heads': 4,
        'num_layers': 3,
        'dropout': 0.15,
    }
    
    # 创建模型 (DynamicGATPolicy 使用 heads 而非 num_heads)
    if model_type == 'dynamic_gat':
        gat_config = {k: v for k, v in model_config.items() if k != 'num_heads'}
        gat_config['heads'] = model_config['num_heads']
        model = DynamicGATPolicy(**gat_config).to(DEVICE)
    elif model_type == 'graph_transformer':
        model = GraphTransformerPolicy(**model_config, use_laplacian_pe=True).to(DEVICE)
    else:
        model = UnifiedGATPolicy(
            in_channels=model_config['in_channels'],
            hidden_channels=model_config['hidden_channels'],
            scale_encoding_dim=model_config['scale_encoding_dim'],
            heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
        ).to(DEVICE)
    
    history = []
    best_reward = -float('inf')
    best_model_state = None
    start_epoch = 0
    
    # 加载检查点
    if resume_from and os.path.exists(resume_from):
        try:
            ck = torch.load(resume_from, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck['model_state_dict'], strict=False)
            history = ck.get('history', [])
            best_reward = ck.get('best_reward', -float('inf'))
            start_epoch = len(history)
            
            if verbose:
                print(f"Loaded checkpoint from {resume_from}")
                print(f"  Previous epochs: {start_epoch}, Best reward: {best_reward:.4f}")
        except Exception as e:
            if verbose:
                print(f"Failed to load checkpoint: {e}")
    
    # 创建训练器
    trainer = CurriculumPPOTrainer(model, model_type=model_type, lr=lr)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode='max', factor=0.7, patience=20, min_lr=1e-5
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Curriculum Learning Phase {phase}")
        print(f"{'='*60}")
        print(f"  Model: {model_type}")
        print(f"  Epochs: {epochs}")
        print(f"  Graphs: {len(graphs)}")
        print(f"  Reward: Phase {phase} reward function")
        print(f"  Device: {DEVICE}")
        print(f"{'='*60}\n")
    
    # 训练循环
    epoch_pbar = tqdm(range(epochs), desc=f"Phase {phase}", disable=not verbose)
    
    for epoch in epoch_pbar:
        epoch_rewards = []
        
        # 课程学习采样
        sampled_graphs = curriculum_sample_graphs(graphs, epoch, epochs, collect_per_epoch)
        
        for G in sampled_graphs:
            if G.number_of_nodes() < 3:
                continue
            
            k = max(1, int(G.number_of_nodes() * k_ratio))
            
            try:
                centers, reward = trainer.collect_trajectory(G, k, reward_fn)
                epoch_rewards.append(reward)
            except Exception as e:
                if verbose and epoch == 0:
                    print(f"Warning: {e}")
                continue
        
        # PPO 更新
        stats = trainer.update()
        
        # 记录
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        history.append([start_epoch + epoch + 1, avg_reward])
        
        # 保存最佳模型
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model_state = model.state_dict().copy()
        
        # 学习率调度
        scheduler.step(avg_reward)
        
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
            'model_type': model_type,
            'model_config': model_config,
            'phase': phase,
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


def train_full_curriculum(
    data_dir,
    total_epochs=450,
    model_type='dynamic_gat',
    save_dir='models',
    verbose=True,
    include_phase4=False,
    real_network_dir=None,
    phase4_epochs=100,
):
    """
    完整的课程学习训练（支持3阶段或4阶段）
    
    Args:
        data_dir: 数据目录
        total_epochs: 前三阶段总训练轮数（将按比例分配）
        model_type: 模型类型
        save_dir: 模型保存目录
        verbose: 是否显示进度
        include_phase4: 是否包含第四阶段（真实网络增强训练）
        real_network_dir: 真实网络数据目录（默认: dataset/all/real）
        phase4_epochs: 第四阶段训练轮数
    """
    # 加载所有图
    all_graphs = load_training_graphs(data_dir, min_nodes=20, max_nodes=500)
    
    if not all_graphs:
        print("Error: No graphs found!")
        return
    
    # 阶段配置
    # 阶段1: 小图 (20-50节点), 占总epochs的20%
    # 阶段2: 中等图 (20-150节点), 占总epochs的30%
    # 阶段3: 全规模图 (20-500节点), 占总epochs的50%
    phase_configs = [
        {
            'phase': 1,
            'epochs': int(total_epochs * 0.20),
            'min_nodes': 20,
            'max_nodes': 50,
            'lr': 3e-4,
        },
        {
            'phase': 2,
            'epochs': int(total_epochs * 0.30),
            'min_nodes': 20,
            'max_nodes': 150,
            'lr': 2e-4,
        },
        {
            'phase': 3,
            'epochs': int(total_epochs * 0.50),
            'min_nodes': 20,
            'max_nodes': 500,
            'lr': 1e-4,
        },
    ]
    
    os.makedirs(save_dir, exist_ok=True)
    
    results = []
    resume_path = None
    
    for config in phase_configs:
        phase = config['phase']
        
        # 过滤图
        phase_graphs = filter_graphs_by_size(
            all_graphs, config['min_nodes'], config['max_nodes']
        )
        
        if not phase_graphs:
            print(f"Warning: No graphs for phase {phase}, skipping...")
            continue
        
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# PHASE {phase}: {config['min_nodes']}-{config['max_nodes']} nodes")
            print(f"# Epochs: {config['epochs']}, Graphs: {len(phase_graphs)}")
            print(f"{'#'*60}")
        
        save_path = os.path.join(save_dir, f'curriculum_phase{phase}_{model_type}.pth')
        
        result = train_single_phase(
            graphs=phase_graphs,
            phase=phase,
            epochs=config['epochs'],
            save_path=save_path,
            model_type=model_type,
            resume_from=resume_path,
            lr=config['lr'],
            verbose=verbose,
        )
        
        results.append(result)
        resume_path = save_path  # 下一阶段从这里继续
    
    # 阶段4: 真实网络增强训练（可选）
    if include_phase4:
        if real_network_dir is None:
            # 默认使用 dataset/all/real 目录
            real_network_dir = os.path.join(os.path.dirname(data_dir), 'raw', 'true')
            if not os.path.exists(real_network_dir):
                real_network_dir = os.path.join(data_dir, 'true')
        
        if os.path.exists(real_network_dir):
            if verbose:
                print(f"\n{'#'*60}")
                print(f"# PHASE 4: Real Network Enhancement Training")
                print(f"# Data: {real_network_dir}")
                print(f"# Epochs: {phase4_epochs}")
                print(f"{'#'*60}")
            
            # 加载真实网络数据
            real_graphs = load_training_graphs(real_network_dir, min_nodes=10, max_nodes=1000)
            
            if real_graphs:
                save_path = os.path.join(save_dir, f'curriculum_phase4_{model_type}.pth')
                
                result = train_single_phase(
                    graphs=real_graphs,
                    phase=4,
                    epochs=phase4_epochs,
                    save_path=save_path,
                    model_type=model_type,
                    resume_from=resume_path,
                    lr=5e-5,  # 更低的学习率进行微调
                    k_ratio=0.1,
                    collect_per_epoch=30,
                    verbose=verbose,
                )
                
                results.append(result)
                resume_path = save_path
            else:
                print(f"Warning: No real network graphs found in {real_network_dir}")
        else:
            print(f"Warning: Real network directory not found: {real_network_dir}")
    
    # 最终模型
    final_path = os.path.join(save_dir, f'curriculum_final_{model_type}.pth')
    if resume_path and os.path.exists(resume_path):
        import shutil
        shutil.copy(resume_path, final_path)
        if verbose:
            print(f"\nFinal model saved to {final_path}")
    
    return results


def train_phase4_only(
    real_network_dir,
    epochs=100,
    model_type='dynamic_gat',
    save_dir='models',
    resume_from=None,
    lr=5e-5,
    k_ratio=0.1,
    collect_per_epoch=30,
    verbose=True,
):
    """
    仅运行第四阶段：真实网络增强训练
    
    从现有的阶段3模型继续训练，专门针对真实网络进行微调。
    
    Args:
        real_network_dir: 真实网络数据目录 (e.g., dataset/all/real)
        epochs: 训练轮数
        model_type: 模型类型
        save_dir: 模型保存目录
        resume_from: 从哪个模型继续（默认: curriculum_phase3_dynamic_gat.pth）
        lr: 学习率（默认较低，用于微调）
        k_ratio: 控制器比例
        collect_per_epoch: 每轮采样图数量
        verbose: 是否显示进度
        
    Returns:
        dict: 训练结果
    """
    if not os.path.exists(real_network_dir):
        print(f"Error: Directory not found: {real_network_dir}")
        return None
    
    # 加载真实网络数据
    real_graphs = load_training_graphs(real_network_dir, min_nodes=10, max_nodes=1000)
    
    if not real_graphs:
        print(f"Error: No graphs found in {real_network_dir}")
        return None
    
    # 默认从阶段3模型继续
    if resume_from is None:
        resume_from = os.path.join(save_dir, f'curriculum_phase3_{model_type}.pth')
        if not os.path.exists(resume_from):
            # 尝试 final 模型
            resume_from = os.path.join(save_dir, f'curriculum_final_{model_type}.pth')
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 4: Real Network Enhancement Training")
        print(f"{'='*60}")
        print(f"  Data directory: {real_network_dir}")
        print(f"  Number of graphs: {len(real_graphs)}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {lr}")
        print(f"  Resume from: {resume_from}")
        print(f"{'='*60}\n")
    
    save_path = os.path.join(save_dir, f'curriculum_phase4_{model_type}.pth')
    
    result = train_single_phase(
        graphs=real_graphs,
        phase=4,
        epochs=epochs,
        save_path=save_path,
        model_type=model_type,
        resume_from=resume_from,
        lr=lr,
        k_ratio=k_ratio,
        collect_per_epoch=collect_per_epoch,
        verbose=verbose,
    )
    
    return result


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='BiT-HyRL Curriculum Learning Training')
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, default='dataset/all',
                        help='Training data directory (default: dataset/all, 图文件在 .gml/.graphml)')
    parser.add_argument('--min-nodes', type=int, default=20,
                        help='Minimum nodes for graphs')
    parser.add_argument('--max-nodes', type=int, default=500,
                        help='Maximum nodes for graphs')
    parser.add_argument('--max-graphs', type=int, default=None,
                        help='Maximum number of graphs to load')
    
    # 训练参数
    parser.add_argument('--phase', type=int, default=None, choices=[1, 2, 3, 4],
                        help='Training phase (1=GCC, 2=+Coverage, 3=+Dispersion, 4=RealNetwork)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for single phase')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--k-ratio', type=float, default=0.1,
                        help='Controller ratio')
    parser.add_argument('--collect-per-epoch', type=int, default=30,
                        help='Graphs to sample per epoch')
    
    # 模型参数
    parser.add_argument('--model-type', type=str, default='dynamic_gat',
                        choices=['dynamic_gat', 'graph_transformer', 'unified_gat'],
                        help='Model architecture')
    
    # 保存/加载
    parser.add_argument('--save-path', type=str, default=None,
                        help='Model save path')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Save directory')
    
    # 完整课程学习
    parser.add_argument('--full-curriculum', action='store_true',
                        help='Run full curriculum learning (3 or 4 phases)')
    parser.add_argument('--total-epochs', type=int, default=450,
                        help='Total epochs for phases 1-3')
    parser.add_argument('--include-phase4', action='store_true',
                        help='Include phase 4 (real network enhancement) in full curriculum')
    parser.add_argument('--phase4-epochs', type=int, default=100,
                        help='Number of epochs for phase 4')
    parser.add_argument('--real-network-dir', type=str, default='dataset/all/real',
                        help='Directory containing real network data for phase 4')
    
    # 仅运行阶段4
    parser.add_argument('--phase4-only', action='store_true',
                        help='Run only phase 4 (real network enhancement training)')
    
    # 其他
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Show progress')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Device: {DEVICE}")
    
    if args.phase4_only:
        # 仅运行阶段4：真实网络增强训练
        train_phase4_only(
            real_network_dir=args.real_network_dir,
            epochs=args.phase4_epochs if args.phase4_epochs else args.epochs,
            model_type=args.model_type,
            save_dir=args.save_dir,
            resume_from=args.resume,
            lr=args.lr if args.lr != 3e-4 else 5e-5,  # 默认使用更低学习率
            k_ratio=args.k_ratio,
            collect_per_epoch=args.collect_per_epoch,
            verbose=args.verbose,
        )
    elif args.full_curriculum:
        # 完整课程学习
        train_full_curriculum(
            data_dir=args.data_dir,
            total_epochs=args.total_epochs,
            model_type=args.model_type,
            save_dir=args.save_dir,
            verbose=args.verbose,
            include_phase4=args.include_phase4,
            real_network_dir=args.real_network_dir,
            phase4_epochs=args.phase4_epochs,
        )
    else:
        # 单阶段训练
        if args.phase is None:
            print("Error: Please specify --phase (1, 2, 3, or 4), --full-curriculum, or --phase4-only")
            return
        
        # 阶段4特殊处理
        if args.phase == 4:
            train_phase4_only(
                real_network_dir=args.real_network_dir,
                epochs=args.epochs,
                model_type=args.model_type,
                save_dir=args.save_dir,
                resume_from=args.resume,
                lr=args.lr if args.lr != 3e-4 else 5e-5,
                k_ratio=args.k_ratio,
                collect_per_epoch=args.collect_per_epoch,
                verbose=args.verbose,
            )
            return
        
        # 加载图（阶段1-3）
        graphs = load_training_graphs(
            args.data_dir,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            max_graphs=args.max_graphs,
        )
        
        if not graphs:
            print("Error: No graphs found!")
            return
        
        # 确定保存路径
        if args.save_path is None:
            args.save_path = os.path.join(
                args.save_dir, f'curriculum_phase{args.phase}_{args.model_type}.pth'
            )
        
        # 训练
        train_single_phase(
            graphs=graphs,
            phase=args.phase,
            epochs=args.epochs,
            save_path=args.save_path,
            model_type=args.model_type,
            resume_from=args.resume,
            lr=args.lr,
            k_ratio=args.k_ratio,
            collect_per_epoch=args.collect_per_epoch,
            verbose=args.verbose,
        )


if __name__ == '__main__':
    main()
