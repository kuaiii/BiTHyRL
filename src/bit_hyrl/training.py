# -*- coding: utf-8 -*-
"""BiT-HyRL 训练与选择（在线微调、离线批量训练、train_and_select）。"""
import os
import random
import csv
import warnings
import time
import torch
from torch.distributions import Categorical
from tqdm import tqdm

# 抑制 Node2Vec 的 RuntimeWarning
warnings.filterwarnings('ignore', category=RuntimeWarning, module='node2vec')

from src.utils.logger import get_logger
from . import config
from . import features
from .model import MLPPolicy
from . import reward
from . import selection

logger = get_logger(__name__)
DEVICE = config.DEVICE
ACCUMULATION_STEPS = 8  # 减少累积步数，避免梯度信号稀释
ENTROPY_COEF = 0.01     # 熵正则化系数，鼓励探索
BASELINE_DECAY = 0.9    # 加快baseline更新，增强advantage信号


class TrainingProgressCallback:
    """训练进度回调，用于实时更新进度"""
    
    def __init__(self, total_epochs: int, total_graphs: int, verbose: bool = True):
        self.total_epochs = total_epochs
        self.total_graphs = total_graphs
        self.verbose = verbose
        self.start_time = time.time()
        self.epoch_rewards = []
        self.best_reward = -float('inf')
        
    def on_epoch_start(self, epoch: int):
        """epoch 开始时调用"""
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, epoch: int, avg_reward: float, graph_count: int):
        """epoch 结束时调用"""
        elapsed = time.time() - self.start_time
        epoch_time = time.time() - self.epoch_start_time
        
        is_best = avg_reward > self.best_reward
        if is_best:
            self.best_reward = avg_reward
        
        self.epoch_rewards.append(avg_reward)
        
        if self.verbose:
            # 计算 ETA
            avg_epoch_time = elapsed / (epoch + 1)
            eta = avg_epoch_time * (self.total_epochs - epoch - 1)
            
            # 进度百分比
            progress = (epoch + 1) / self.total_epochs * 100
            
            # 构建状态字符串
            best_marker = " ★" if is_best else ""
            status = (
                f"\rEpoch {epoch+1:3d}/{self.total_epochs} ({progress:5.1f}%) | "
                f"Reward: {avg_reward:.4f} | Best: {self.best_reward:.4f}{best_marker} | "
                f"Graphs: {graph_count} | "
                f"Time: {elapsed:.0f}s | ETA: {eta:.0f}s"
            )
            print(status, end='', flush=True)
            
            # 每10轮换行
            if (epoch + 1) % 10 == 0:
                print()
    
    def on_training_end(self):
        """训练结束时调用"""
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Total Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Best Reward: {self.best_reward:.4f}")
        print(f"  Final Reward: {self.epoch_rewards[-1]:.4f}" if self.epoch_rewards else "")
        print(f"{'='*60}")


def train_offline_single(G, k, episodes, metric_type='combined', initial_centers=None,
                         use_stepwise_reward=True, use_node2vec=True):
    """单图在线微调，支持 Step-wise Reward。"""
    x, node_list = features.get_node_features(G, use_node2vec=use_node2vec)
    x = x.to(DEVICE)
    nf = x.shape[1]
    model = MLPPolicy(num_features=nf, hidden_dim=64, output_dim=1)
    model = model.to(DEVICE)
    if initial_centers:
        r0 = reward.calculate_reward(G, initial_centers, mode=metric_type, use_stepwise=False)
        best_reward, best_centers = r0, initial_centers.copy()
    else:
        best_reward, best_centers = -1.0, []
    lr = 0.005 if episodes > 50 else 0.01
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(episodes):
        idxs, log_probs, cents = [], [], []
        mask = torch.zeros(len(node_list), dtype=torch.bool, device=DEVICE)
        for step in range(k):
            probs = model(x, mask)
            m = Categorical(probs)
            a = m.sample()
            i = a.item()
            idxs.append(i)
            log_probs.append(m.log_prob(a))
            mask = mask.clone()
            mask[i] = True
            c = node_list[i]
            cents.append(c)
            if use_stepwise_reward:
                r = reward.calculate_stepwise_reward(G, cents[:-1], c, mode=metric_type)
                loss = -log_probs[-1] * r
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
        if not use_stepwise_reward:
            centers = [node_list[i] for i in idxs]
            r = reward.calculate_reward(G, centers, mode=metric_type, use_stepwise=False)
            if r > best_reward:
                best_reward, best_centers = r, centers
            L = sum(-lp * r for lp in log_probs)
            opt.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        else:
            final = reward.calculate_reward(G, cents, mode=metric_type, use_stepwise=False)
            if final > best_reward:
                best_reward, best_centers = final, cents
    if not best_centers and k > 0:
        best_centers = random.sample(list(G.nodes()), k)
    return best_centers


def train_and_select(G, k, episodes=50, metric_type='combined',
                     use_stepwise_reward=True, use_node2vec=True, use_ci=True,
                     test_mode=False):
    """
    预训练 + 在线微调 或 纯在线训练。
    
    Args:
        G: 输入图
        k: 控制器数量
        episodes: 训练轮数
        metric_type: 优化目标类型
        use_stepwise_reward: 是否使用Step-wise Reward
        use_node2vec: 是否使用Node2Vec特征
        use_ci: 是否使用CI算法
        test_mode: 测试模式。当为True时，仅加载已有模型进行推理，不进行任何训练。
                   如果模型不存在则报错。
    """
    model_name = 'rl_agent.pth' if metric_type == 'combined' else f'rl_agent_{metric_type}.pth'
    model_path = os.path.join(config.MODEL_DIR, model_name)
    fallback = os.path.join(config.MODEL_DIR, 'rl_agent.pth')
    
    # 测试模式：仅加载模型进行推理，不进行训练
    if test_mode:
        if os.path.exists(model_path):
            logger.debug(f"[Test Mode] 加载模型: {model_path}")
            return selection.hybrid_rl_select(G, k, model_path=model_path, use_ci=use_ci)
        elif os.path.exists(fallback):
            logger.debug(f"[Test Mode] 加载通用模型: {fallback}")
            return selection.hybrid_rl_select(G, k, model_path=fallback, use_ci=use_ci)
        else:
            logger.warning(f"[Test Mode] 未找到模型文件: {model_path}")
            # 降级为使用CI算法选择
            logger.warning("[Test Mode] 降级为CI算法选择控制器")
            return selection.ci_select_subset(G, list(G.nodes()), k, [], radius=2)
    
    # 训练模式：原有逻辑
    if os.path.exists(model_path):
        if episodes >= 60:
            best_centers, best_r = None, -float('inf')
            pre = selection.hybrid_rl_select(G, k, model_path=model_path, use_ci=use_ci)
            r = reward.calculate_reward(G, pre, mode=metric_type, use_stepwise=False)
            if r > best_r:
                best_r, best_centers = r, pre
            ft = train_offline_single(G, k, episodes, metric_type, initial_centers=pre,
                                     use_stepwise_reward=use_stepwise_reward, use_node2vec=use_node2vec)
            if ft:
                rf = reward.calculate_reward(G, ft, mode=metric_type, use_stepwise=False)
                if rf > best_r:
                    best_r, best_centers = rf, ft
            if episodes >= 100:
                fresh = train_offline_single(G, k, episodes // 2, metric_type,
                                            use_stepwise_reward=use_stepwise_reward, use_node2vec=use_node2vec)
                if fresh:
                    rn = reward.calculate_reward(G, fresh, mode=metric_type, use_stepwise=False)
                    if rn > best_r:
                        best_centers = fresh
            return best_centers or pre
        return selection.hybrid_rl_select(G, k, model_path=model_path, use_ci=use_ci)
    if metric_type != 'combined' and os.path.exists(fallback):
        if episodes >= 60:
            pre = selection.hybrid_rl_select(G, k, model_path=fallback, use_ci=use_ci)
            ft = train_offline_single(G, k, episodes, metric_type, initial_centers=pre,
                                     use_stepwise_reward=use_stepwise_reward, use_node2vec=use_node2vec)
            return ft if ft else pre
        return selection.hybrid_rl_select(G, k, model_path=fallback, use_ci=use_ci)
    return train_offline_single(G, k, max(episodes, 80), metric_type,
                                use_stepwise_reward=use_stepwise_reward, use_node2vec=use_node2vec)


def train_offline_optimized(graphs, epochs=100, save_path=None, mode='combined',
                            use_node2vec=True, use_stepwise_reward=True,
                            resume_from=None, lr=0.001, verbose=True,
                            entropy_coef=ENTROPY_COEF, use_lr_scheduler=True):
    """
    离线批量训练 BiT-HyRL Agent，支持增量训练和进度可视化。
    
    Args:
        graphs: 训练图列表
        epochs: 训练轮数
        save_path: 模型保存路径
        mode: 优化模式 (combined/robustness/csa/entropy/wcp)
        use_node2vec: 是否使用Node2Vec特征
        use_stepwise_reward: 是否使用Step-wise Reward
        resume_from: 增量训练时加载的已有模型路径（为None则从头训练）
        lr: 学习率（增量训练时建议使用较小的学习率如0.00005）
        verbose: 是否显示训练进度（默认True）
        
    Returns:
        dict: 训练结果，包含 history, final_reward, model_path 等
    """
    if not graphs:
        logger.error("No graphs for training")
        return None
        
    if save_path is None:
        save_path = os.path.join(config.MODEL_DIR, 'rl_agent.pth' if mode == 'combined' else f'rl_agent_{mode}.pth')
    
    # 确定统一的特征维度
    # 当使用 Node2Vec 时，特征维度 = node2vec_dim(64) + 5 = 69
    # 不使用 Node2Vec 时，特征维度 = 5
    if use_node2vec:
        nf = 64 + 5  # Node2Vec(64) + 统计特征(5)
    else:
        nf = 5  # 仅统计特征
    
    logger.debug(f"模型特征维度: {nf} ({'Node2Vec + 统计特征' if use_node2vec else '仅统计特征'})")
    
    # 创建或加载模型
    model = MLPPolicy(num_features=nf, hidden_dim=64, output_dim=1).to(DEVICE)
    start_epoch = 0
    prev_history = []
    
    # 增量训练：加载已有模型
    if resume_from and os.path.exists(resume_from):
        try:
            checkpoint = torch.load(resume_from, map_location=DEVICE, weights_only=False)
            # 检查是否为完整的checkpoint（包含optimizer等）还是仅state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                prev_history = checkpoint.get('history', [])
                logger.debug(f"[增量训练] 加载模型: {resume_from}, 从epoch {start_epoch} 继续")
            else:
                # 兼容旧格式（仅state_dict）
                model.load_state_dict(checkpoint)
                logger.debug(f"[增量训练] 加载模型: {resume_from} (旧格式)")
            # 增量训练使用更小的学习率
            lr = lr * 0.5
            logger.debug(f"[增量训练] 使用学习率: {lr}")
        except Exception as e:
            logger.warning(f"加载模型失败: {e}，将从头训练")
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 学习率调度器：余弦退火，帮助跳出局部最优
    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=max(10, epochs // 5), T_mult=2, eta_min=lr * 0.01
        )
    
    history = prev_history.copy()
    moving_avg = 0.0
    best_reward = -float('inf')
    best_model_state = None
    training_start_time = time.time()
    
    # 打印训练配置
    if verbose:
        print(f"\n{'='*60}")
        print(f"BiT-HyRL Training Started")
        print(f"{'='*60}")
        print(f"  Mode: {mode}")
        print(f"  Epochs: {epochs}")
        print(f"  Graphs: {len(graphs)}")
        print(f"  Features: {nf}D ({'Node2Vec' if use_node2vec else 'Stats only'})")
        print(f"  Learning Rate: {lr}")
        print(f"  Device: {DEVICE}")
        print(f"{'='*60}\n")
    
    # 使用 tqdm 进度条
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch", 
                      disable=not verbose, ncols=100)
    
    for epoch in epoch_pbar:
        actual_epoch = start_epoch + epoch + 1
        model.train()
        total_r = 0.0
        graph_count = 0
        random.shuffle(graphs)
        opt.zero_grad()
        total_loss = None
        
        # 图级别进度条（嵌套）
        graph_pbar = tqdm(enumerate(graphs), total=len(graphs), 
                         desc=f"Epoch {actual_epoch}", leave=False,
                         disable=not verbose, ncols=80)
        
        for i, G in graph_pbar:
            if G.number_of_nodes() < 2:
                continue
            
            x, node_list = features.get_node_features(G, use_node2vec=use_node2vec)
            
            # 检查特征维度是否匹配模型
            if x.shape[1] != nf:
                # 如果 Node2Vec 失败导致维度不匹配，跳过此图
                continue
                
            graph_count += 1
            x = x.to(DEVICE)
            k = max(1, min(int(G.number_of_nodes() * 0.1), G.number_of_nodes() - 1))
            log_probs, cents = [], []
            mask = torch.zeros(len(node_list), dtype=torch.bool, device=DEVICE)
            
            for step in range(k):
                probs = model(x, mask)
                m = Categorical(probs)
                a = m.sample()
                idx = a.item()
                log_probs.append(m.log_prob(a))
                mask = mask.clone()
                mask[idx] = True
                c = node_list[idx]
                cents.append(c)
                
                if use_stepwise_reward:
                    r = reward.calculate_stepwise_reward(G, cents[:-1], c, mode=mode)
                    # 使用更快的 baseline 更新
                    moving_avg = r if moving_avg == 0.0 else BASELINE_DECAY * moving_avg + (1 - BASELINE_DECAY) * r
                    adv = r - moving_avg
                    total_r += r
                    
                    # 添加 entropy bonus 鼓励探索
                    entropy = m.entropy()
                    policy_loss = -log_probs[-1] * adv
                    entropy_loss = -entropy_coef * entropy
                    loss = (policy_loss + entropy_loss) / ACCUMULATION_STEPS
                    total_loss = loss if total_loss is None else total_loss + loss
                else:
                    if step < k - 1:
                        continue
                    r = reward.calculate_reward(G, cents, mode=mode, use_stepwise=False)
                    # 使用更快的 baseline 更新
                    moving_avg = r if moving_avg == 0.0 else BASELINE_DECAY * moving_avg + (1 - BASELINE_DECAY) * r
                    adv = r - moving_avg
                    total_r += r
                    
                    # 添加 entropy bonus
                    avg_entropy = sum(Categorical(model(x, None)).entropy() for _ in range(1)) / 1
                    policy_loss = sum(-lp * adv for lp in log_probs)
                    entropy_loss = -entropy_coef * avg_entropy * k
                    loss = (policy_loss + entropy_loss) / ACCUMULATION_STEPS
                    total_loss = loss if total_loss is None else total_loss + loss
                    
            if total_loss is not None:
                total_loss.backward()
                total_loss = None
            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad()
            
            # 更新图级别进度条
            if graph_count > 0:
                current_avg = total_r / graph_count
                graph_pbar.set_postfix({'reward': f'{current_avg:.3f}'})
        
        graph_pbar.close()
                
        if len(graphs) % ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()
        
        # 更新学习率调度器
        if scheduler is not None:
            scheduler.step()
            
        avg_r = total_r / graph_count if graph_count > 0 else 0
        history.append([actual_epoch, avg_r])
        
        # 保存最佳模型
        is_best = avg_r > best_reward
        if is_best:
            best_reward = avg_r
            best_model_state = model.state_dict().copy()
        
        # 更新 epoch 进度条
        epoch_pbar.set_postfix({
            'reward': f'{avg_r:.4f}',
            'best': f'{best_reward:.4f}',
            'graphs': graph_count
        })
        
        # 日志（每10轮或最后一轮）- 使用 DEBUG 级别
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            elapsed = time.time() - training_start_time
            logger.debug(f"Epoch {actual_epoch}/{start_epoch + epochs} | "
                        f"Reward: {avg_r:.4f} | Best: {best_reward:.4f} | "
                        f"Time: {elapsed:.1f}s")
    
    # 训练完成
    total_training_time = time.time() - training_start_time
    
    # 保存模型（完整checkpoint格式，支持增量训练）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'model_state_dict': best_model_state if best_model_state else model.state_dict(),
        'epoch': start_epoch + epochs,
        'history': history,
        'best_reward': best_reward,
        'mode': mode,
        'use_node2vec': use_node2vec,
        'use_stepwise_reward': use_stepwise_reward,
    }
    torch.save(checkpoint, save_path)
    
    # 同时保存纯state_dict版本（兼容旧代码）
    state_dict_path = save_path.replace('.pth', '_state_dict.pth')
    torch.save(best_model_state if best_model_state else model.state_dict(), state_dict_path)
    
    logger.debug(f"Model saved to {save_path}")
    
    # 保存训练历史
    hist_path = save_path.replace('.pth', '_training_history.csv')
    with open(hist_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Epoch', 'Avg_Reward'])
        w.writerows(history)
    logger.debug(f"Training history saved to {hist_path}")
    
    # 打印最终总结
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"  Total Time:    {total_training_time:.1f}s ({total_training_time/60:.1f} min)")
        print(f"  Total Epochs:  {epochs}")
        print(f"  Best Reward:   {best_reward:.4f}")
        print(f"  Final Reward:  {avg_r:.4f}")
        print(f"  Model saved:   {save_path}")
        print(f"  History saved: {hist_path}")
        print(f"{'='*60}\n")
    
    # 返回训练结果
    return {
        'history': history,
        'final_reward': avg_r,
        'best_reward': best_reward,
        'total_epochs': start_epoch + epochs,
        'model_path': save_path,
        'history_path': hist_path,
        'training_time': total_training_time,
    }


def evaluate_model(model_path, test_graphs, mode='combined', use_node2vec=True, use_ci=True, verbose=True):
    """
    评估模型在测试图上的效果（带进度可视化）。
    
    Args:
        model_path: 模型路径
        test_graphs: 测试图列表
        mode: 优化模式
        use_node2vec: 是否使用Node2Vec特征
        use_ci: 是否使用CI算法
        verbose: 是否显示进度
        
    Returns:
        dict: 评估结果
    """
    import networkx as nx
    
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating Model: {os.path.basename(model_path)}")
        print(f"Test Graphs: {len(test_graphs)}")
        print(f"{'='*60}\n")
    
    results = []
    skipped = 0
    
    # 使用 tqdm 进度条
    eval_pbar = tqdm(enumerate(test_graphs), total=len(test_graphs),
                    desc="Evaluating", unit="graph", disable=not verbose, ncols=100)
    
    for i, G in eval_pbar:
        if G.number_of_nodes() < 2:
            skipped += 1
            continue
            
        # 验证图的约束
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        is_connected = nx.is_connected(G)
        
        try:
            # 使用模型选择控制器
            k = max(1, min(int(n_nodes * 0.1), n_nodes - 1))
            centers = selection.hybrid_rl_select(G, k, model_path=model_path, use_ci=use_ci)
            
            # 计算奖励
            r = reward.calculate_reward(G, centers, mode=mode, use_stepwise=False)
            
            results.append({
                'graph_idx': i,
                'nodes': n_nodes,
                'edges': n_edges,
                'is_connected': is_connected,
                'controllers': len(centers),
                'reward': r,
            })
            
            # 更新进度条
            if results:
                avg_so_far = sum(r['reward'] for r in results) / len(results)
                eval_pbar.set_postfix({'avg_reward': f'{avg_so_far:.4f}', 'done': len(results)})
                
        except Exception as e:
            skipped += 1
            logger.warning(f"评估图 {i} 失败: {e}")
            continue
    
    eval_pbar.close()
    
    if skipped > 0:
        logger.debug(f"评估过程中跳过 {skipped} 个图（特征维度不匹配或其他错误）")
        
    if not results:
        return None
        
    # 统计汇总
    rewards = [r['reward'] for r in results]
    avg_reward = sum(rewards) / len(rewards)
    min_reward = min(rewards)
    max_reward = max(rewards)
    
    # 打印评估结果
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluation Results")
        print(f"{'='*60}")
        print(f"  Evaluated Graphs: {len(results)}")
        print(f"  Skipped Graphs:   {skipped}")
        print(f"  Average Reward:   {avg_reward:.4f}")
        print(f"  Min Reward:       {min_reward:.4f}")
        print(f"  Max Reward:       {max_reward:.4f}")
        print(f"{'='*60}\n")
    
    return {
        'num_graphs': len(results),
        'avg_reward': avg_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'details': results,
    }


# ============================================================
# GNN + PPO 训练入口
# ============================================================

def train_gnn_ppo_optimized(
    graphs,
    epochs=100,
    save_path=None,
    mode='gcc',
    lr=3e-4,
    embed_dim=256,          # Node2Vec 嵌入维度
    dre_dim=256,            # DRE 度排名嵌入维度
    hidden_channels=256,    # 隐藏层维度
    heads=12,               # GAT 注意力头数
    num_layers=6,           # GAT 层数（深层）
    k_ratio=0.1,
    collect_per_epoch=60,
    walk_length=20,         # Node2Vec 游走长度
    num_walks=100,          # Node2Vec 每节点游走次数
    use_embedding_cache=True,  # 是否使用嵌入缓存
    verbose=True,
    resume_from=None,       # 增量训练：从此 checkpoint 继续训练
    **reward_kwargs,        # 传递给奖励函数的额外参数（如 adversarial 的 sample_methods）
):
    """
    使用深层 GAT + PPO 训练控制器选择策略
    
    这是 BiT-HyRL 的新架构入口：
    - 使用 Node2Vec 嵌入作为节点特征 (128维)
    - 使用 5层深层 GAT 捕捉高阶结构信息
    - 使用 PPO 算法训练
    - 全部由 GAT+RL 选择控制器，不使用 CI 算法
    
    Args:
        graphs: 训练图列表
        epochs: 训练轮数
        save_path: 模型保存路径
        mode: 奖励模式 ('gcc' 使用 GCC 专注奖励)
        lr: 学习率
        embed_dim: Node2Vec 嵌入维度 (默认128)
        hidden_channels: GAT 隐藏层维度 (默认128)
        heads: GAT 注意力头数 (默认8)
        num_layers: GAT 层数 (默认5)
        k_ratio: 控制器比例
        collect_per_epoch: 每轮收集的轨迹数
        verbose: 是否显示进度
        
    Returns:
        dict: 训练结果
    """
    try:
        from .ppo_trainer import train_gnn_ppo
        from .reward import (
            calculate_gcc_focused_reward,
            calculate_multi_attack_reward,
            calculate_adversarial_reward,
        )
    except ImportError as e:
        logger.error(f"导入 GNN 模块失败: {e}")
        logger.error("请确保已安装 torch-geometric: pip install torch-geometric")
        return None
    
    if not graphs:
        logger.error("没有训练图")
        return None
    
    # 根据模式选择奖励函数
    if mode == 'gcc':
        reward_fn = calculate_gcc_focused_reward
    elif mode == 'multi_attack':
        reward_fn = calculate_multi_attack_reward
    elif mode == 'adversarial':
        # 对抗式多攻击奖励：每次评估从多种 network_dismantling 方法中采样
        if reward_kwargs:
            import functools
            reward_fn = functools.partial(calculate_adversarial_reward, **reward_kwargs)
        else:
            reward_fn = calculate_adversarial_reward
    else:
        # 默认使用 GCC 专注奖励
        reward_fn = calculate_gcc_focused_reward
    
    # 确定保存路径
    if save_path is None:
        save_path = os.path.join(config.MODEL_DIR, 'gnn_ppo_agent.pth')
    
    # 总输入维度 = Node2Vec + DRE
    in_channels = embed_dim + dre_dim
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"BiT-HyRL Deep GAT+PPO Training")
        print(f"{'='*60}")
        print(f"  Mode: {mode}")
        print(f"  Epochs: {epochs}")
        print(f"  Graphs: {len(graphs)}")
        print(f"  Learning Rate: {lr}")
        print(f"  Node2Vec Embed Dim: {embed_dim}")
        print(f"  DRE Embed Dim: {dre_dim}")
        print(f"  Total Input Dim: {in_channels}")
        print(f"  Hidden Channels: {hidden_channels}")
        print(f"  Attention Heads: {heads}")
        print(f"  GAT Layers: {num_layers}")
        print(f"  Controller Ratio: {k_ratio}")
        print(f"  Device: {DEVICE}")
        print(f"{'='*60}\n")
    
    # 调用 PPO 训练（支持增量训练 resume_from）
    result = train_gnn_ppo(
        graphs=graphs,
        epochs=epochs,
        save_path=save_path,
        reward_fn=reward_fn,
        lr=lr,
        in_channels=in_channels,
        embed_dim=embed_dim,
        dre_dim=dre_dim,
        hidden_channels=hidden_channels,
        heads=heads,
        num_layers=num_layers,
        k_ratio=k_ratio,
        collect_per_epoch=collect_per_epoch,
        walk_length=walk_length,
        num_walks=num_walks,
        use_embedding_cache=use_embedding_cache,
        verbose=verbose,
        resume_from=resume_from,
    )
    
    return result


def evaluate_gnn_model(model_path, test_graphs, mode='gcc', k_ratio=0.1, verbose=True):
    """
    评估深层 GNN 模型在测试图上的效果
    
    使用 Node2Vec 嵌入作为节点特征。
    
    Args:
        model_path: GNN 模型路径
        test_graphs: 测试图列表
        mode: 评估模式
        k_ratio: 控制器比例
        verbose: 是否显示进度
        
    Returns:
        dict: 评估结果
    """
    import networkx as nx
    
    try:
        from .gnn_model import GATPolicy, graph_to_pyg_data, get_gnn_node_features
        from .reward import calculate_gcc_focused_reward
    except ImportError as e:
        logger.error(f"导入 GNN 模块失败: {e}")
        return None
    
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None
    
    # 加载模型
    try:
        import torch
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        # 从 checkpoint 获取模型参数
        in_channels = checkpoint.get('in_channels', 128)
        embed_dim = checkpoint.get('embed_dim', in_channels)
        dre_dim = checkpoint.get('dre_dim', 0)
        hidden_channels = checkpoint.get('hidden_channels', 128)
        heads = checkpoint.get('heads', 8)
        num_layers = checkpoint.get('num_layers', 5)
        
        model = GATPolicy(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            heads=heads,
            num_layers=num_layers,
        ).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return None
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating Deep GNN Model: {os.path.basename(model_path)}")
        print(f"  Architecture: {num_layers}-layer GAT, {heads} heads")
        print(f"  Input: {in_channels}D (Node2Vec {embed_dim}D + DRE {dre_dim}D)")
        print(f"Test Graphs: {len(test_graphs)}")
        print(f"{'='*60}\n")
    
    results = []
    skipped = 0
    
    eval_pbar = tqdm(enumerate(test_graphs), total=len(test_graphs),
                    desc="Evaluating Deep GNN", unit="graph", disable=not verbose, ncols=100)
    
    for i, G in eval_pbar:
        if G.number_of_nodes() < 3:
            skipped += 1
            continue
        
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        is_connected = nx.is_connected(G)
        
        try:
            # 获取 DRE + Node2Vec 特征
            x, node_list = get_gnn_node_features(
                G, device=DEVICE, embed_dim=embed_dim, dre_dim=dre_dim, seed=42
            )
            _, edge_index, _ = graph_to_pyg_data(G, device=DEVICE)
            
            # 选择控制器（全部由 GNN 选择）
            k = max(1, int(n_nodes * k_ratio))
            selected_mask = torch.zeros(n_nodes, dtype=torch.bool, device=DEVICE)
            centers = []
            
            with torch.no_grad():
                for _ in range(k):
                    action, _, _, _ = model.get_action(
                        x, edge_index, selected_mask=selected_mask, deterministic=True
                    )
                    action_idx = action.item()
                    centers.append(node_list[action_idx])
                    selected_mask[action_idx] = True
            
            # 计算奖励
            r = calculate_gcc_focused_reward(G, centers)
            
            results.append({
                'graph_idx': i,
                'nodes': n_nodes,
                'edges': n_edges,
                'is_connected': is_connected,
                'controllers': len(centers),
                'reward': r,
            })
            
            if results:
                avg_so_far = sum(r['reward'] for r in results) / len(results)
                eval_pbar.set_postfix({'avg_reward': f'{avg_so_far:.4f}', 'done': len(results)})
                
        except Exception as e:
            skipped += 1
            logger.warning(f"评估图 {i} 失败: {e}")
            continue
    
    eval_pbar.close()
    
    if not results:
        return None
    
    # 统计
    rewards = [r['reward'] for r in results]
    avg_reward = sum(rewards) / len(rewards)
    min_reward = min(rewards)
    max_reward = max(rewards)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Deep GNN Evaluation Results")
        print(f"{'='*60}")
        print(f"  Evaluated Graphs: {len(results)}")
        print(f"  Skipped Graphs:   {skipped}")
        print(f"  Average Reward:   {avg_reward:.4f}")
        print(f"  Min Reward:       {min_reward:.4f}")
        print(f"  Max Reward:       {max_reward:.4f}")
        print(f"{'='*60}\n")
    
    return {
        'num_graphs': len(results),
        'avg_reward': avg_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'details': results,
    }