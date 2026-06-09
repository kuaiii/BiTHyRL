# -*- coding: utf-8 -*-
"""
BiT-HyRL 改进版训练系统

主要改进:
1. Actor-Critic 架构 (减少方差)
2. 攻击感知奖励函数 (直接优化抗攻击能力)
3. 稳定的优势估计 (GAE)
4. 动态熵正则化
5. 渐进式训练策略
"""
import os
import random
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
import networkx as nx

from src.utils.logger import get_logger
from . import config
from . import features

logger = get_logger(__name__)
DEVICE = config.DEVICE


# ============================================================
# 改进的 Actor-Critic 模型
# ============================================================

class ImprovedActorCritic(nn.Module):
    """
    Actor-Critic 策略网络
    
    改进:
    - 添加 Value head (Critic) 用于优势估计
    - 更稳定的初始化
    - 自适应温度参数
    """
    
    def __init__(self, num_features, hidden_dim=128, dropout=0.1):
        super().__init__()
        
        # 共享特征提取器
        self.shared = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Actor head: 输出节点选择分数
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Critic head: 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # 可学习温度参数
        self.log_temperature = nn.Parameter(torch.zeros(1))
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """正交初始化 - 对RL更稳定"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, mask=None, return_value=False):
        """
        前向传播
        
        Args:
            x: 节点特征 [N, F]
            mask: 已选节点掩码 [N]
            return_value: 是否返回状态价值
            
        Returns:
            probs: 动作概率分布 [N]
            value: 状态价值 (如果 return_value=True)
        """
        # 共享特征
        features = self.shared(x)  # [N, hidden_dim]
        
        # Actor: 计算节点分数
        scores = self.actor(features).squeeze(-1)  # [N]
        
        # 温度缩放
        temperature = torch.clamp(self.log_temperature.exp(), min=0.5, max=2.0)
        scores = scores / temperature
        
        # 掩码
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        probs = F.softmax(scores, dim=0)
        
        if return_value:
            # Critic: 计算状态价值 (全局池化)
            global_feature = features.mean(dim=0, keepdim=True)  # [1, hidden_dim]
            value = self.critic(global_feature).squeeze()  # scalar
            return probs, value
        
        return probs
    
    def get_action_and_value(self, x, mask=None, action=None):
        """
        获取动作、对数概率、熵和状态价值
        
        用于 PPO 风格的训练
        """
        probs, value = self.forward(x, mask, return_value=True)
        dist = Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


# ============================================================
# 攻击感知奖励函数
# ============================================================

def calculate_attack_aware_reward(G, centers, attack_type='degree', attack_steps=30):
    """
    攻击感知奖励函数
    
    直接模拟攻击过程，计算 GCC 保持能力
    这与测试评估完全一致
    
    Args:
        G: NetworkX 图
        centers: 控制器列表
        attack_type: 攻击类型 ('degree', 'betweenness')
        attack_steps: 攻击步数
        
    Returns:
        float: 奖励值 (R值，越大越好)
    """
    if not centers:
        return 0.0
    
    centers_set = set(centers)
    total_nodes = G.number_of_nodes()
    
    if total_nodes < 3:
        return 0.0
    
    # 模拟攻击
    G_sim = G.copy()
    remaining_centers = centers_set.copy()
    
    # 计算 R 值 (曲线下面积)
    y_values = [1.0]  # 初始 GCC 比例为 1
    
    for step in range(min(attack_steps, total_nodes - 1)):
        if G_sim.number_of_nodes() <= 1:
            break
        
        # 选择攻击目标
        if attack_type == 'degree':
            degrees = dict(G_sim.degree())
            if not degrees:
                break
            target = max(degrees, key=degrees.get)
        else:
            # betweenness 攻击
            try:
                bc = nx.betweenness_centrality(G_sim)
                target = max(bc, key=bc.get)
            except:
                degrees = dict(G_sim.degree())
                target = max(degrees, key=degrees.get) if degrees else None
        
        if target is None:
            break
        
        # 移除节点
        G_sim.remove_node(target)
        remaining_centers.discard(target)
        
        # 计算当前 GCC 比例
        if G_sim.number_of_nodes() == 0 or len(remaining_centers) == 0:
            y_values.append(0.0)
        else:
            components = list(nx.connected_components(G_sim))
            controlled_sizes = [len(c) for c in components if not remaining_centers.isdisjoint(c)]
            if controlled_sizes:
                gcc_ratio = max(controlled_sizes) / total_nodes
            else:
                gcc_ratio = 0.0
            y_values.append(gcc_ratio)
    
    # 计算 R 值 (梯形积分)
    r_value = 0.0
    for i in range(len(y_values) - 1):
        r_value += (y_values[i] + y_values[i + 1]) / 2
    
    # 归一化
    max_r = len(y_values) - 1  # 最大可能 R 值
    normalized_r = r_value / max_r if max_r > 0 else 0.0
    
    return normalized_r


def calculate_position_penalty(G, centers):
    """
    位置惩罚：惩罚选择高度数节点作为控制器
    
    高度数节点是攻击的首要目标，不应该作为控制器
    
    Returns:
        float: 惩罚值 (0-1，越小越好)
    """
    if not centers:
        return 0.0
    
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    
    # 前 10% 的高度数节点是危险区
    danger_zone_size = max(1, G.number_of_nodes() // 10)
    danger_zone = set(sorted_nodes[:danger_zone_size])
    
    # 计算在危险区的控制器比例
    in_danger = len(set(centers) & danger_zone)
    penalty = in_danger / len(centers)
    
    return penalty


def calculate_dispersion_bonus(G, centers):
    """
    分散奖励：鼓励控制器均匀分布
    
    Returns:
        float: 奖励值 (0-1，越大越好)
    """
    if len(centers) < 2:
        return 0.5
    
    # 计算控制器之间的最小距离
    min_distances = []
    centers_list = list(centers)
    
    for i, c1 in enumerate(centers_list):
        for c2 in centers_list[i+1:]:
            if G.has_node(c1) and G.has_node(c2):
                try:
                    if nx.has_path(G, c1, c2):
                        d = nx.shortest_path_length(G, c1, c2)
                        min_distances.append(d)
                except:
                    pass
    
    if not min_distances:
        return 0.5
    
    avg_dist = sum(min_distances) / len(min_distances)
    
    # 理想距离是图直径的 1/4
    try:
        diameter = nx.diameter(G) if nx.is_connected(G) else G.number_of_nodes() // 3
    except:
        diameter = G.number_of_nodes() // 3
    
    ideal_dist = max(diameter / 4, 2)
    dispersion = min(avg_dist / ideal_dist, 1.0)
    
    return dispersion


def calculate_combined_reward(G, centers, attack_steps=30):
    """
    综合奖励函数
    
    组合攻击感知奖励、位置惩罚和分散奖励
    
    Args:
        G: NetworkX 图
        centers: 控制器列表
        attack_steps: 攻击步数
        
    Returns:
        float: 综合奖励值
    """
    if not centers:
        return 0.0
    
    # 攻击感知奖励 (主要)
    r_value = calculate_attack_aware_reward(G, centers, attack_steps=attack_steps)
    
    # 位置惩罚 (避免高度数节点)
    position_penalty = calculate_position_penalty(G, centers)
    
    # 分散奖励
    dispersion = calculate_dispersion_bonus(G, centers)
    
    # 综合奖励: 70% R值 + 20% 分散 - 10% 位置惩罚
    reward = 0.7 * r_value + 0.2 * dispersion - 0.1 * position_penalty
    
    return max(0.0, reward)


def calculate_stepwise_attack_reward(G, previous_centers, new_center, attack_steps=20):
    """
    攻击感知的 Step-wise 奖励
    
    计算添加新控制器后 R 值的增量
    """
    if new_center is None or not G.has_node(new_center):
        return 0.0
    
    centers_before = list(previous_centers) if previous_centers else []
    centers_after = centers_before + [new_center]
    
    # 计算 R 值增量
    r_before = calculate_attack_aware_reward(G, centers_before, attack_steps=attack_steps) if centers_before else 0.0
    r_after = calculate_attack_aware_reward(G, centers_after, attack_steps=attack_steps)
    
    delta_r = r_after - r_before
    
    # 位置奖励/惩罚
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    danger_zone_size = max(1, G.number_of_nodes() // 10)
    danger_zone = set(sorted_nodes[:danger_zone_size])
    
    position_bonus = -0.1 if new_center in danger_zone else 0.05
    
    # 存活奖励 (保证每步都有信号)
    survival_bonus = 0.02
    
    return delta_r + position_bonus + survival_bonus


# ============================================================
# 改进的训练函数
# ============================================================

def train_improved(
    graphs,
    epochs=200,
    save_path=None,
    use_node2vec=True,
    lr=0.0003,
    gamma=0.99,
    gae_lambda=0.95,
    entropy_coef=0.05,
    value_coef=0.5,
    max_grad_norm=0.5,
    attack_steps=30,
    verbose=True,
    resume_from=None,
):
    """
    改进版训练函数
    
    主要改进:
    1. Actor-Critic 架构
    2. GAE 优势估计
    3. 攻击感知奖励
    4. 动态熵调整
    5. 更稳定的超参数
    
    Args:
        graphs: 训练图列表
        epochs: 训练轮数
        save_path: 模型保存路径
        use_node2vec: 是否使用 Node2Vec 特征
        lr: 学习率
        gamma: 折扣因子
        gae_lambda: GAE lambda 参数
        entropy_coef: 初始熵系数
        value_coef: 价值损失系数
        max_grad_norm: 梯度裁剪
        attack_steps: 奖励计算时的攻击步数
        verbose: 是否显示进度
        resume_from: 继续训练的模型路径
        
    Returns:
        dict: 训练结果
    """
    if not graphs:
        logger.error("没有训练图")
        return None
    
    # 确定保存路径
    if save_path is None:
        save_path = os.path.join(config.MODEL_DIR, 'rl_agent_improved.pth')
    
    # 特征维度
    nf = 64 + 5 if use_node2vec else 5
    
    # 创建模型
    model = ImprovedActorCritic(num_features=nf, hidden_dim=128).to(DEVICE)
    
    # 加载已有模型
    start_epoch = 0
    prev_history = []
    if resume_from and os.path.exists(resume_from):
        try:
            checkpoint = torch.load(resume_from, map_location=DEVICE, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                prev_history = checkpoint.get('history', [])
                logger.info(f"从 epoch {start_epoch} 继续训练")
        except Exception as e:
            logger.warning(f"加载模型失败: {e}")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(20, epochs // 5), T_mult=2, eta_min=lr * 0.01
    )
    
    # 训练历史
    history = prev_history.copy()
    best_reward = -float('inf')
    best_model_state = None
    
    # 熵系数退火
    initial_entropy_coef = entropy_coef
    final_entropy_coef = 0.01
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"BiT-HyRL Improved Training")
        print(f"{'='*60}")
        print(f"  Epochs: {epochs}")
        print(f"  Graphs: {len(graphs)}")
        print(f"  Features: {nf}D")
        print(f"  Learning Rate: {lr}")
        print(f"  Gamma: {gamma}")
        print(f"  GAE Lambda: {gae_lambda}")
        print(f"  Entropy Coef: {entropy_coef} → {final_entropy_coef}")
        print(f"  Attack Steps: {attack_steps}")
        print(f"  Device: {DEVICE}")
        print(f"{'='*60}\n")
    
    training_start = time.time()
    
    epoch_pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
    
    for epoch in epoch_pbar:
        actual_epoch = start_epoch + epoch + 1
        model.train()
        
        # 动态熵系数
        progress = epoch / epochs
        current_entropy_coef = initial_entropy_coef * (1 - progress) + final_entropy_coef * progress
        
        epoch_rewards = []
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []
        
        random.shuffle(graphs)
        
        for G in graphs:
            if G.number_of_nodes() < 5:
                continue
            
            # 获取特征
            try:
                x, node_list = features.get_node_features(G, use_node2vec=use_node2vec)
                if x.shape[1] != nf:
                    continue
            except Exception:
                continue
            
            x = x.to(DEVICE)
            n_nodes = len(node_list)
            k = max(1, min(int(n_nodes * 0.1), n_nodes - 1))
            
            # 收集轨迹
            log_probs = []
            values = []
            rewards = []
            entropies = []
            centers = []
            mask = torch.zeros(n_nodes, dtype=torch.bool, device=DEVICE)
            
            for step in range(k):
                # 获取动作和价值
                action, log_prob, entropy, value = model.get_action_and_value(x, mask)
                
                log_probs.append(log_prob)
                values.append(value)
                entropies.append(entropy)
                
                # 更新状态
                idx = action.item()
                mask = mask.clone()
                mask[idx] = True
                centers.append(node_list[idx])
                
                # 计算 step-wise 奖励
                if step < k - 1:
                    r = calculate_stepwise_attack_reward(
                        G, centers[:-1], centers[-1], attack_steps=attack_steps // 2
                    )
                else:
                    # 最后一步使用完整奖励
                    r = calculate_combined_reward(G, centers, attack_steps=attack_steps)
                
                rewards.append(r)
            
            # 计算 GAE
            advantages = []
            returns = []
            gae = 0
            
            # 最终状态的价值估计
            with torch.no_grad():
                _, final_value = model(x, mask, return_value=True)
            
            # 反向计算 GAE
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = final_value
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + gamma * next_value - values[t]
                gae = delta + gamma * gae_lambda * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])
            
            # 转换为张量
            advantages = torch.stack(advantages)
            returns = torch.stack(returns)
            log_probs_t = torch.stack(log_probs)
            values_t = torch.stack(values)
            entropies_t = torch.stack(entropies)
            
            # 优势标准化
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 计算损失
            policy_loss = -(log_probs_t * advantages.detach()).mean()
            value_loss = F.mse_loss(values_t, returns.detach())
            entropy_loss = -entropies_t.mean()
            
            total_loss = policy_loss + value_coef * value_loss + current_entropy_coef * entropy_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            # 记录
            final_reward = calculate_combined_reward(G, centers, attack_steps=attack_steps)
            epoch_rewards.append(final_reward)
            epoch_policy_losses.append(policy_loss.item())
            epoch_value_losses.append(value_loss.item())
            epoch_entropies.append(entropies_t.mean().item())
        
        # 更新学习率
        scheduler.step()
        
        # 计算统计
        if epoch_rewards:
            avg_reward = sum(epoch_rewards) / len(epoch_rewards)
            avg_policy_loss = sum(epoch_policy_losses) / len(epoch_policy_losses)
            avg_value_loss = sum(epoch_value_losses) / len(epoch_value_losses)
            avg_entropy = sum(epoch_entropies) / len(epoch_entropies)
        else:
            avg_reward = 0.0
            avg_policy_loss = 0.0
            avg_value_loss = 0.0
            avg_entropy = 0.0
        
        history.append([actual_epoch, avg_reward])
        
        # 保存最佳模型
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model_state = model.state_dict().copy()
        
        # 更新进度条
        epoch_pbar.set_postfix({
            'reward': f'{avg_reward:.4f}',
            'best': f'{best_reward:.4f}',
            'ent': f'{avg_entropy:.3f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
        
        # 详细日志
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            logger.debug(
                f"Epoch {actual_epoch} | Reward: {avg_reward:.4f} | "
                f"Policy: {avg_policy_loss:.4f} | Value: {avg_value_loss:.4f} | "
                f"Entropy: {avg_entropy:.4f}"
            )
    
    # 保存模型
    training_time = time.time() - training_start
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'model_state_dict': best_model_state if best_model_state else model.state_dict(),
        'epoch': start_epoch + epochs,
        'history': history,
        'best_reward': best_reward,
        'use_node2vec': use_node2vec,
        'model_type': 'ImprovedActorCritic',
    }
    torch.save(checkpoint, save_path)
    
    # 保存训练历史
    hist_path = save_path.replace('.pth', '_training_history.csv')
    with open(hist_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Avg_Reward'])
        writer.writerows(history)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"  Total Time: {training_time:.1f}s ({training_time/60:.1f} min)")
        print(f"  Best Reward: {best_reward:.4f}")
        print(f"  Model saved: {save_path}")
        print(f"{'='*60}\n")
    
    return {
        'history': history,
        'best_reward': best_reward,
        'model_path': save_path,
        'training_time': training_time,
    }


def select_with_improved_model(G, k, model_path, use_node2vec=True):
    """
    使用改进模型选择控制器
    
    Args:
        G: NetworkX 图
        k: 控制器数量
        model_path: 模型路径
        use_node2vec: 是否使用 Node2Vec 特征
        
    Returns:
        list: 控制器列表
    """
    if not os.path.exists(model_path):
        logger.warning(f"模型不存在: {model_path}，使用度数中心性")
        degrees = dict(G.degree())
        sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
        # 避免选择最高度数的节点（攻击目标）
        safe_start = max(1, G.number_of_nodes() // 20)
        return sorted_nodes[safe_start:safe_start + k]
    
    # 加载模型
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        nf = 64 + 5 if use_node2vec else 5
        model = ImprovedActorCritic(num_features=nf).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return list(G.nodes())[:k]
    
    # 获取特征
    try:
        x, node_list = features.get_node_features(G, use_node2vec=use_node2vec)
        if x.shape[1] != nf:
            x, node_list = features.get_node_features(G, use_node2vec=False)
    except Exception as e:
        logger.error(f"特征提取失败: {e}")
        return list(G.nodes())[:k]
    
    x = x.to(DEVICE)
    n_nodes = len(node_list)
    k = min(k, n_nodes - 1)
    
    # 选择控制器
    centers = []
    mask = torch.zeros(n_nodes, dtype=torch.bool, device=DEVICE)
    
    with torch.no_grad():
        for _ in range(k):
            probs = model(x, mask)
            # 确定性选择（greedy）
            idx = probs.argmax().item()
            centers.append(node_list[idx])
            mask[idx] = True
    
    return centers


# ============================================================
# 评估函数
# ============================================================

def evaluate_improved_model(model_path, test_graphs, use_node2vec=True, attack_steps=50, verbose=True):
    """
    评估改进模型
    
    Args:
        model_path: 模型路径
        test_graphs: 测试图列表
        use_node2vec: 是否使用 Node2Vec
        attack_steps: 攻击步数
        verbose: 是否显示进度
        
    Returns:
        dict: 评估结果
    """
    if not os.path.exists(model_path):
        logger.error(f"模型不存在: {model_path}")
        return None
    
    results = []
    
    eval_pbar = tqdm(test_graphs, desc="Evaluating", disable=not verbose)
    
    for G in eval_pbar:
        if G.number_of_nodes() < 5:
            continue
        
        n_nodes = G.number_of_nodes()
        k = max(1, int(n_nodes * 0.1))
        
        # 使用模型选择
        centers = select_with_improved_model(G, k, model_path, use_node2vec)
        
        # 计算 R 值
        r_value = calculate_attack_aware_reward(G, centers, attack_steps=attack_steps)
        
        results.append({
            'nodes': n_nodes,
            'controllers': len(centers),
            'r_value': r_value,
        })
        
        if results:
            avg_r = sum(r['r_value'] for r in results) / len(results)
            eval_pbar.set_postfix({'avg_R': f'{avg_r:.4f}'})
    
    if not results:
        return None
    
    r_values = [r['r_value'] for r in results]
    
    summary = {
        'num_graphs': len(results),
        'avg_r_value': sum(r_values) / len(r_values),
        'min_r_value': min(r_values),
        'max_r_value': max(r_values),
        'details': results,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluation Results")
        print(f"{'='*60}")
        print(f"  Graphs: {summary['num_graphs']}")
        print(f"  Avg R-value: {summary['avg_r_value']:.4f}")
        print(f"  Min R-value: {summary['min_r_value']:.4f}")
        print(f"  Max R-value: {summary['max_r_value']:.4f}")
        print(f"{'='*60}\n")
    
    return summary
