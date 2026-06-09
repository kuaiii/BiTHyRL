# -*- coding: utf-8 -*-
"""
BiT-HyRL 改进版训练脚本

主要改进:
1. 增加每轮采样数量 (collect_per_epoch: 20 -> 50)
2. 降低学习率 (3e-4 -> 1e-4)
3. 使用更好的 step-wise 奖励
4. 添加学习率预热和余弦退火
5. 增加 PPO 更新次数 (4 -> 10)
6. 增加熵系数鼓励探索 (0.01 -> 0.02)
7. 使用梯度累积稳定训练
8. 添加早停机制

用法:
  # 推荐配置（稳定训练）
  python scripts/train_bit_hyrl_improved.py --epochs 200 --mode robustness
  
  # 快速测试
  python scripts/train_bit_hyrl_improved.py --epochs 50 --fast
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import random
import time
import logging
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm

from src.topology.generators import load_graph
from src.topology.reconstruction import create_bimodal_network_exact
from src.bit_hyrl import config
from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)
DEVICE = config.DEVICE


def load_training_data(train_dir, max_graphs=None, min_nodes=10, max_nodes=500,
                       use_bimodal=True, hub_ratio=0.15, focus_range=None):
    """
    加载训练数据
    
    Args:
        train_dir: 训练数据目录
        max_graphs: 最大加载图数量
        min_nodes: 最小节点数
        max_nodes: 最大节点数
        use_bimodal: 是否使用双峰拓扑重构
        hub_ratio: hub 节点比例
        focus_range: 重点关注的节点范围 (min, max)，会增加该范围的采样权重
    """
    if not os.path.exists(train_dir):
        print(f"错误: 训练数据目录不存在: {train_dir}")
        return [], {}
    
    graphs = []
    graph_sizes = []
    all_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.gml')]
    
    print(f"发现 {len(all_files)} 个 .gml 文件")
    
    if max_graphs and max_graphs < len(all_files):
        random.shuffle(all_files)
        all_files = all_files[:max_graphs]
        print(f"随机采样 {max_graphs} 个图")
    
    for filepath in tqdm(all_files, desc='Loading data', ncols=80):
        try:
            G, _ = load_graph(filepath, verbose=False)
            if G is None or G.number_of_nodes() < min_nodes:
                continue
            if G.number_of_nodes() > max_nodes:
                continue
            if not nx.is_connected(G):
                continue
            
            n_nodes = G.number_of_nodes()
            
            if use_bimodal:
                G = create_bimodal_network_exact(G, hub_ratio=hub_ratio, seed=len(graphs))
            
            graphs.append(G)
            graph_sizes.append(n_nodes)
        except Exception as e:
            continue
    
    print(f"加载完成: {len(graphs)} 个有效图")
    
    # 分析节点数分布
    if graph_sizes:
        size_dist = analyze_size_distribution(graph_sizes, focus_range)
    else:
        size_dist = {}
    
    return graphs, size_dist


def analyze_size_distribution(sizes, focus_range=None):
    """分析节点数分布"""
    ranges = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 500)]
    
    print("\n节点数分布:")
    dist = {}
    for lo, hi in ranges:
        count = sum(1 for s in sizes if lo <= s < hi)
        pct = count / len(sizes) * 100
        dist[(lo, hi)] = count
        
        # 标记重点关注范围
        marker = ""
        if focus_range and lo >= focus_range[0] and hi <= focus_range[1]:
            marker = " ← 重点"
        
        bar = '█' * int(pct / 2)
        print(f"  {lo:4d}-{hi:4d}: {count:4d} ({pct:5.1f}%) {bar}{marker}")
    
    return dist


class ImprovedPPOTrainer:
    """
    改进版 PPO 训练器
    
    改进点:
    1. 更好的奖励塑形
    2. 学习率预热 + 余弦退火
    3. 增加熵系数鼓励探索
    4. 更多的 PPO 更新次数
    5. 早停机制
    """
    
    def __init__(
        self,
        model,
        lr=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.02,  # 增加熵系数
        max_grad_norm=0.5,
        n_epochs=10,  # 增加更新次数
        batch_size=32,  # 减小 batch size
        warmup_steps=100,
    ):
        from collections import namedtuple
        
        self.Experience = namedtuple('Experience', [
            'x', 'edge_index', 'action', 'log_prob', 'value', 'reward', 'selected_mask', 'done'
        ])
        
        self.model = model.to(DEVICE)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        
        self.experiences = []
        self.episode_rewards = []
        self.step_count = 0
        self.base_lr = lr
        
    def get_lr_multiplier(self):
        """学习率预热 + 余弦退火"""
        if self.step_count < self.warmup_steps:
            return self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, 1000 - self.warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
    
    def update_lr(self):
        """更新学习率"""
        multiplier = self.get_lr_multiplier()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr * multiplier
        return self.base_lr * multiplier
    
    def collect_trajectory(self, G, k, reward_fn, node_features=None):
        """收集轨迹并使用改进的奖励塑形"""
        from src.bit_hyrl.gnn_model import get_simple_node_features, graph_to_pyg_data
        
        self.model.eval()
        
        if node_features is None:
            x, node_list = get_simple_node_features(G, device=DEVICE)
        else:
            x = node_features.to(DEVICE)
            node_list = list(G.nodes())
        
        _, edge_index, _ = graph_to_pyg_data(G, device=DEVICE)
        
        num_nodes = len(node_list)
        selected_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
        centers = []
        experiences = []
        
        with torch.no_grad():
            for step in range(k):
                action, log_prob, value, probs = self.model.get_action(
                    x, edge_index, selected_mask=selected_mask, deterministic=False
                )
                
                action_idx = action.item()
                center = node_list[action_idx]
                centers.append(center)
                
                # 改进的 step-wise 奖励
                if step < k - 1:
                    # 使用增量奖励而非固定值
                    partial_reward = reward_fn(G, centers) * 0.1  # 部分奖励
                    step_reward = partial_reward
                else:
                    step_reward = reward_fn(G, centers)
                
                done = (step == k - 1)
                exp = self.Experience(
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
                
                selected_mask = selected_mask.clone()
                selected_mask[action_idx] = True
        
        final_reward = reward_fn(G, centers)
        
        # 用最终奖励替换最后一步的奖励
        if experiences:
            exp = experiences[-1]
            experiences[-1] = self.Experience(
                x=exp.x, edge_index=exp.edge_index, action=exp.action,
                log_prob=exp.log_prob, value=exp.value, reward=final_reward,
                selected_mask=exp.selected_mask, done=exp.done,
            )
        
        for exp in experiences:
            self.experiences.append(exp)
        self.episode_rewards.append(final_reward)
        
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
        import torch.nn.functional as F
        
        if len(self.experiences) == 0:
            return {}
        
        self.model.train()
        self.step_count += 1
        current_lr = self.update_lr()
        
        experiences = self.experiences
        
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
                    
                    probs, value = self.model(
                        exp.x, exp.edge_index, selected_mask=exp.selected_mask
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
        
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        
        # 清空缓冲区
        self.experiences = []
        self.episode_rewards = []
        
        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
            'avg_reward': avg_reward,
            'lr': current_lr,
        }


def weighted_sample_graphs(graphs, sample_size, focus_range=None, focus_weight=3.0):
    """
    加权采样图
    
    Args:
        graphs: 图列表
        sample_size: 采样数量
        focus_range: 重点关注的节点范围 (min, max)
        focus_weight: 重点范围的采样权重倍数
        
    Returns:
        采样后的图列表
    """
    if focus_range is None:
        return random.sample(graphs, min(sample_size, len(graphs)))
    
    # 计算每个图的权重
    weights = []
    for G in graphs:
        n = G.number_of_nodes()
        if focus_range[0] <= n <= focus_range[1]:
            weights.append(focus_weight)
        else:
            weights.append(1.0)
    
    # 归一化权重
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]
    
    # 加权采样（不重复）
    indices = list(range(len(graphs)))
    sampled_indices = []
    remaining_probs = probs.copy()
    
    for _ in range(min(sample_size, len(graphs))):
        # 归一化剩余概率
        total = sum(remaining_probs[i] for i in indices if i not in sampled_indices)
        if total == 0:
            break
        
        # 采样
        r = random.random() * total
        cumsum = 0
        for i in indices:
            if i in sampled_indices:
                continue
            cumsum += remaining_probs[i]
            if cumsum >= r:
                sampled_indices.append(i)
                break
    
    return [graphs[i] for i in sampled_indices]


def train_improved_gnn_ppo(
    graphs,
    epochs=200,
    save_path=None,
    mode='robustness',
    lr=1e-4,
    hidden_channels=64,
    heads=4,
    k_ratio=0.1,
    collect_per_epoch=50,  # 增加采样数量
    patience=30,  # 早停耐心值
    focus_range=None,  # 重点关注的节点范围
    focus_weight=3.0,  # 重点范围的采样权重
    curriculum=False,  # 是否使用课程学习
    verbose=True,
):
    """
    改进版 GNN+PPO 训练
    
    主要改进:
    1. 更多采样 (50/epoch vs 20/epoch)
    2. 更低学习率 (1e-4 vs 3e-4)
    3. 学习率预热+余弦退火
    4. 更多 PPO 更新次数 (10 vs 4)
    5. 更高熵系数 (0.02 vs 0.01)
    6. 早停机制
    7. 支持加权采样（重点关注特定规模）
    8. 支持课程学习（从小图到大图）
    """
    from src.bit_hyrl.gnn_model import GATPolicy
    from src.bit_hyrl.reward import calculate_gcc_focused_reward, calculate_adaptive_reward
    
    # 根据模式选择奖励函数
    if mode == 'robustness':
        reward_fn = lambda G, c: calculate_adaptive_reward(G, c, mode='combined')
    elif mode == 'gcc':
        reward_fn = calculate_gcc_focused_reward
    else:
        reward_fn = lambda G, c: calculate_adaptive_reward(G, c, mode=mode)
    
    if save_path is None:
        save_path = os.path.join(ROOT, 'models', f'gnn_ppo_agent_{mode}.pth')
    
    # 创建模型
    model = GATPolicy(
        in_channels=5,
        hidden_channels=hidden_channels,
        heads=heads,
    ).to(DEVICE)
    
    # 创建训练器
    trainer = ImprovedPPOTrainer(
        model,
        lr=lr,
        n_epochs=10,
        batch_size=32,
        entropy_coef=0.02,
    )
    
    # 课程学习：按节点数排序
    if curriculum:
        graphs = sorted(graphs, key=lambda G: G.number_of_nodes())
        print("启用课程学习: 从小图到大图逐步训练")
    
    # 训练历史
    history = []
    best_reward = -float('inf')
    best_model_state = None
    no_improve_count = 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"BiT-HyRL 改进版 GNN+PPO 训练")
        print(f"{'='*60}")
        print(f"  模式: {mode}")
        print(f"  训练轮数: {epochs}")
        print(f"  训练图数: {len(graphs)}")
        print(f"  每轮采样: {collect_per_epoch}")
        print(f"  学习率: {lr}")
        print(f"  隐藏层维度: {hidden_channels}")
        print(f"  注意力头数: {heads}")
        print(f"  早停耐心值: {patience}")
        if focus_range:
            print(f"  重点关注范围: {focus_range[0]}-{focus_range[1]} 节点")
            print(f"  重点权重: {focus_weight}x")
        if curriculum:
            print(f"  课程学习: 启用")
        print(f"  设备: {DEVICE}")
        print(f"{'='*60}\n")
    
    epoch_pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
    
    for epoch in epoch_pbar:
        epoch_rewards = []
        focus_rewards = []  # 重点范围内的奖励
        
        # 课程学习：逐步扩大训练范围
        if curriculum:
            # 前 1/3 epochs 只用小图，中间 1/3 用中等图，后 1/3 用全部
            progress = epoch / epochs
            if progress < 0.33:
                max_idx = len(graphs) // 3
            elif progress < 0.66:
                max_idx = 2 * len(graphs) // 3
            else:
                max_idx = len(graphs)
            available_graphs = graphs[:max_idx]
        else:
            available_graphs = graphs
        
        # 加权采样
        sampled_graphs = weighted_sample_graphs(
            available_graphs, collect_per_epoch, focus_range, focus_weight
        )
        
        for G in sampled_graphs:
            if G.number_of_nodes() < 5:
                continue
            
            k = max(1, int(G.number_of_nodes() * k_ratio))
            
            try:
                centers, reward = trainer.collect_trajectory(G, k, reward_fn)
                epoch_rewards.append(reward)
                
                # 记录重点范围的奖励
                if focus_range and focus_range[0] <= G.number_of_nodes() <= focus_range[1]:
                    focus_rewards.append(reward)
            except Exception as e:
                continue
        
        # PPO 更新
        stats = trainer.update()
        
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        history.append([epoch + 1, avg_reward])
        
        # 早停检查
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # 更新进度条
        postfix = {
            'reward': f'{avg_reward:.4f}',
            'best': f'{best_reward:.4f}',
            'lr': f'{stats.get("lr", lr):.2e}',
        }
        if focus_rewards:
            postfix['focus'] = f'{np.mean(focus_rewards):.4f}'
        epoch_pbar.set_postfix(postfix)
        
        # 早停
        if no_improve_count >= patience:
            if verbose:
                print(f"\n早停触发: {patience} 轮无改进")
            break
    
    # 保存模型
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': best_model_state if best_model_state else model.state_dict(),
            'model_type': 'GATPolicy',
            'in_channels': 5,
            'hidden_channels': hidden_channels,
            'heads': heads,
            'best_reward': best_reward,
            'history': history,
            'mode': mode,
            'focus_range': focus_range,
        }
        torch.save(checkpoint, save_path)
        
        if verbose:
            print(f"\n模型已保存: {save_path}")
            print(f"最佳奖励: {best_reward:.4f}")
    
    # 绘制训练曲线
    plot_training_curve(history, save_path.replace('.pth', '_training_curve.png'))
    
    return {
        'history': history,
        'best_reward': best_reward,
        'final_reward': history[-1][1] if history else 0,
        'model_path': save_path,
        'total_epochs': len(history),
    }


def plot_training_curve(history, save_path):
    """绘制训练曲线"""
    try:
        import matplotlib.pyplot as plt
        
        epochs = [h[0] for h in history]
        rewards = [h[1] for h in history]
        
        plt.figure(figsize=(12, 6))
        
        # 主图
        plt.subplot(1, 2, 1)
        plt.plot(epochs, rewards, 'b-', linewidth=1.5, alpha=0.7, label='Reward')
        
        # 移动平均
        if len(rewards) > 10:
            window = 10
            moving_avg = []
            for i in range(len(rewards)):
                start = max(0, i - window + 1)
                moving_avg.append(sum(rewards[start:i+1]) / (i - start + 1))
            plt.plot(epochs, moving_avg, 'r-', linewidth=2, label=f'MA({window})')
        
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.title('Training Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 标注最佳点
        best_idx = rewards.index(max(rewards))
        plt.annotate(f'Best: {rewards[best_idx]:.4f}', 
                     xy=(epochs[best_idx], rewards[best_idx]),
                     xytext=(epochs[best_idx] + len(epochs)*0.05, rewards[best_idx]),
                     arrowprops=dict(arrowstyle='->', color='green'),
                     fontsize=10, color='green')
        
        # 奖励分布直方图
        plt.subplot(1, 2, 2)
        plt.hist(rewards, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.axvline(x=np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.4f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"训练曲线已保存: {save_path}")
        
    except ImportError:
        print("警告: matplotlib 未安装")
    except Exception as e:
        print(f"绘图失败: {e}")


def main():
    ap = argparse.ArgumentParser(description='BiT-HyRL 改进版训练')
    
    ap.add_argument('--mode', type=str, default='robustness',
                    choices=['robustness', 'gcc', 'combined', 'survival', 'integral'],
                    help='优化模式')
    ap.add_argument('--epochs', type=int, default=200, help='训练轮数')
    ap.add_argument('--lr', type=float, default=1e-4, help='学习率')
    ap.add_argument('--collect_per_epoch', type=int, default=50, help='每轮采样数')
    ap.add_argument('--hidden_channels', type=int, default=64, help='隐藏层维度')
    ap.add_argument('--heads', type=int, default=4, help='注意力头数')
    ap.add_argument('--k_ratio', type=float, default=0.1, help='控制器比例')
    ap.add_argument('--patience', type=int, default=30, help='早停耐心值')
    
    # 数据参数
    ap.add_argument('--max_graphs', type=int, default=None, help='最大训练图数量')
    ap.add_argument('--min_nodes', type=int, default=10, help='最小节点数')
    ap.add_argument('--max_nodes', type=int, default=500, help='最大节点数')
    ap.add_argument('--use_bimodal', action='store_true', default=True)
    ap.add_argument('--no_bimodal', dest='use_bimodal', action='store_false')
    ap.add_argument('--hub_ratio', type=float, default=0.15, help='Hub节点比例')
    
    # 重点关注特定规模
    ap.add_argument('--focus_min', type=int, default=None, help='重点关注的最小节点数')
    ap.add_argument('--focus_max', type=int, default=None, help='重点关注的最大节点数')
    ap.add_argument('--focus_weight', type=float, default=3.0, help='重点范围的采样权重倍数')
    
    # 课程学习
    ap.add_argument('--curriculum', action='store_true', help='启用课程学习（从小图到大图）')
    
    # 快速模式
    ap.add_argument('--fast', action='store_true', help='快速测试模式')
    
    # 输出
    ap.add_argument('--output_dir', type=str, default=None, help='输出目录')
    
    args = ap.parse_args()
    
    # 快速模式
    if args.fast:
        args.epochs = 50
        args.collect_per_epoch = 20
        args.max_graphs = 100
        args.max_nodes = 200
        args.patience = 15
        print("启用快速测试模式")
    
    # 设置日志
    setup_logger(log_dir='logs', log_filename='training_improved.log', 
                 level=logging.WARNING, console_level=logging.WARNING)
    
    # 确定重点关注范围
    focus_range = None
    if args.focus_min is not None and args.focus_max is not None:
        focus_range = (args.focus_min, args.focus_max)
        print(f"重点关注 {args.focus_min}-{args.focus_max} 节点范围 (权重 {args.focus_weight}x)")
    
    # 加载数据
    train_dir = os.path.join(ROOT, 'dataset', 'all', 'syn')
    graphs, size_dist = load_training_data(
        train_dir,
        max_graphs=args.max_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        use_bimodal=args.use_bimodal,
        hub_ratio=args.hub_ratio,
        focus_range=focus_range,
    )
    
    if not graphs:
        print("无可用训练数据")
        return
    
    # 打印数据统计
    nodes_list = [G.number_of_nodes() for G in graphs]
    print(f"\n训练数据统计:")
    print(f"  总图数: {len(graphs)}")
    print(f"  节点数范围: {min(nodes_list)} - {max(nodes_list)}")
    print(f"  平均节点数: {np.mean(nodes_list):.1f}")
    
    # 确定保存路径
    out_dir = args.output_dir or os.path.join(ROOT, 'models')
    suffix = '_improved'
    if focus_range:
        suffix = f'_focus{focus_range[0]}-{focus_range[1]}'
    save_path = os.path.join(out_dir, f'gnn_ppo_agent_{args.mode}{suffix}.pth')
    
    # 训练
    result = train_improved_gnn_ppo(
        graphs,
        epochs=args.epochs,
        save_path=save_path,
        mode=args.mode,
        lr=args.lr,
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        k_ratio=args.k_ratio,
        collect_per_epoch=args.collect_per_epoch,
        patience=args.patience,
        focus_range=focus_range,
        focus_weight=args.focus_weight,
        curriculum=args.curriculum,
        verbose=True,
    )
    
    # 打印结果
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")
    print(f"  最佳奖励: {result['best_reward']:.4f}")
    print(f"  最终奖励: {result['final_reward']:.4f}")
    print(f"  总轮数: {result['total_epochs']}")
    print(f"  模型路径: {result['model_path']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
