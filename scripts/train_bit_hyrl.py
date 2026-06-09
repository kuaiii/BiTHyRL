# -*- coding: utf-8 -*-
"""
BiT-HyRL 训练入口：离线批量训练 Agent，支持 K-折交叉验证。

数据集结构:
  - dataset/all/syn/  : 训练数据集（用于 k-折交叉验证）
  - dataset/testdata/   : 最终验证集（独立测试集，不参与训练）

功能:
  1. 从头训练新模型
  2. K-折交叉验证
  3. 增量训练（在已有模型基础上继续训练）
  4. 训练效果评估（使用 testdata 验证集）

用法:
  # 从头训练（默认使用全部训练数据）
  python scripts/train_bit_hyrl.py --epochs 100 --mode robustness
  
  # K-折交叉验证
  python scripts/train_bit_hyrl.py --epochs 100 --kfold 5
  
  # 增量训练
  python scripts/train_bit_hyrl.py --epochs 50 --resume
  
  # 训练后在 testdata 验证集上评估
  python scripts/train_bit_hyrl.py --epochs 100 --evaluate
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import csv
import random
import time
import logging
import numpy as np
import networkx as nx
from tqdm import tqdm
from src.topology.generators import load_graph
from src.topology.reconstruction import create_bimodal_network_exact
from src.bit_hyrl import train_offline_optimized, config
from src.bit_hyrl.training import evaluate_model
from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def _load_training_data(validate=True, verbose=False, use_bimodal=True, hub_ratio=0.15,
                        max_graphs=None, min_nodes=10, max_nodes=500):
    """
    从 dataset/all/syn/ 加载训练数据集。
    
    Args:
        validate: 是否验证约束（节点、边、连通性）
        verbose: 是否输出详细加载信息
        use_bimodal: 是否对训练数据进行双峰拓扑重构（默认True）
        hub_ratio: hub节点比例（默认0.15）
        max_graphs: 最大加载图数量（None表示全部）
        min_nodes: 最小节点数过滤
        max_nodes: 最大节点数过滤（大图训练很慢）
        
    Returns:
        (图列表, 统计信息列表)
    """
    # 训练数据目录
    train_dir = os.path.join(ROOT, 'dataset', 'all', 'syn')
    
    if not os.path.exists(train_dir):
        print(f"错误: 训练数据目录不存在: {train_dir}")
        return [], []
    
    graphs = []
    stats = []
    failed = 0
    constraint_failed = 0
    skipped_size = 0
    
    # 收集所有 .gml 文件
    all_files = []
    for f in os.listdir(train_dir):
        if f.endswith('.gml'):
            all_files.append(os.path.join(train_dir, f))
    
    print(f"从 train/ 发现 {len(all_files)} 个 .gml 文件")
    
    # 随机打乱并限制数量（加速训练）
    if max_graphs and max_graphs < len(all_files):
        random.shuffle(all_files)
        all_files = all_files[:max_graphs]
        print(f"随机采样 {max_graphs} 个图用于训练")
    
    print(f"节点数过滤: {min_nodes} - {max_nodes}")
    if use_bimodal:
        print(f"启用双峰拓扑重构 (hub_ratio={hub_ratio})")
    
    for filepath in tqdm(all_files, desc='Loading training data', ncols=80):
        filename = os.path.basename(filepath)
        name = os.path.splitext(filename)[0]
        
        try:
            G_original, _ = load_graph(filepath, verbose=verbose)
            
            if G_original is None or G_original.number_of_nodes() < 2:
                failed += 1
                continue
            
            original_nodes = G_original.number_of_nodes()
            original_edges = G_original.number_of_edges()
            
            # 节点数过滤（跳过太小或太大的图）
            if original_nodes < min_nodes or original_nodes > max_nodes:
                skipped_size += 1
                continue
            
            original_connected = nx.is_connected(G_original)
            
            # 约束验证（原始图）
            if validate:
                if original_edges < original_nodes - 1:
                    failed += 1
                    continue
                if not original_connected:
                    failed += 1
                    continue
            
            # 双峰拓扑重构
            if use_bimodal:
                G = create_bimodal_network_exact(G_original, hub_ratio=hub_ratio, seed=len(graphs))
                
                final_nodes = G.number_of_nodes()
                final_edges = G.number_of_edges()
                final_connected = nx.is_connected(G)
                
                # 严格检查约束
                if final_nodes != original_nodes or final_edges != original_edges or not final_connected:
                    constraint_failed += 1
                    logger.warning(f"[约束检查] {name}: 重构约束不满足 "
                                  f"(N: {original_nodes}->{final_nodes}, "
                                  f"M: {original_edges}->{final_edges}, "
                                  f"连通: {final_connected})")
                    G = G_original
                    final_nodes = original_nodes
                    final_edges = original_edges
                    final_connected = original_connected
            else:
                G = G_original
                final_nodes = original_nodes
                final_edges = original_edges
                final_connected = original_connected
            
            graphs.append(G)
            stats.append({
                'name': name,
                'nodes': final_nodes,
                'edges': final_edges,
                'connected': final_connected,
                'bimodal': use_bimodal,
            })
            
        except Exception as e:
            failed += 1
            logger.warning(f"加载 {name} 失败: {e}")
            continue
    
    print(f"加载完成: {len(graphs)} 个有效图")
    if skipped_size > 0:
        print(f"  因节点数过滤跳过: {skipped_size} 个")
    if failed > 0:
        print(f"  验证失败: {failed} 个")
    if use_bimodal and constraint_failed > 0:
        print(f"  双峰重构约束失败: {constraint_failed} 个 (已降级为原始图)")
    
    return graphs, stats


def _load_validation_data(validate=False, use_bimodal=True, hub_ratio=0.15):
    """
    从 dataset/testdata 加载最终验证数据集。
    
    Args:
        validate: 是否验证约束
        use_bimodal: 是否进行双峰拓扑重构
        hub_ratio: hub节点比例
        
    Returns:
        (图列表, 统计信息列表)
    """
    test_dir = os.path.join(ROOT, 'dataset', 'testdata')
    
    if not os.path.exists(test_dir):
        print(f"错误: 验证数据目录不存在: {test_dir}")
        return [], []
    
    graphs = []
    stats = []
    
    gml_files = [f for f in os.listdir(test_dir) if f.endswith('.gml')]
    
    for filename in gml_files:
        name = os.path.splitext(filename)[0]
        filepath = os.path.join(test_dir, filename)
        
        try:
            G_original, _ = load_graph(filepath)
            
            if G_original is None or G_original.number_of_nodes() < 2:
                continue
            
            original_nodes = G_original.number_of_nodes()
            original_edges = G_original.number_of_edges()
            original_connected = nx.is_connected(G_original)
            
            # 约束验证
            if validate:
                if original_edges < original_nodes - 1:
                    print(f"  警告: {name} 边数({original_edges})不满足连通性最小要求({original_nodes-1})")
                if not original_connected:
                    print(f"  警告: {name} 图不连通")
            
            # 双峰拓扑重构
            if use_bimodal:
                G = create_bimodal_network_exact(G_original, hub_ratio=hub_ratio, seed=len(graphs))
                final_nodes = G.number_of_nodes()
                final_edges = G.number_of_edges()
                final_connected = nx.is_connected(G)
                
                if final_nodes != original_nodes or final_edges != original_edges or not final_connected:
                    G = G_original
                    final_nodes = original_nodes
                    final_edges = original_edges
                    final_connected = original_connected
            else:
                G = G_original
                final_nodes = original_nodes
                final_edges = original_edges
                final_connected = original_connected
            
            graphs.append(G)
            stats.append({
                'name': name,
                'nodes': final_nodes,
                'edges': final_edges,
                'connected': final_connected,
            })
            print(f"  Loaded {name}: {final_nodes} nodes, {final_edges} edges")
            
        except Exception as e:
            print(f"  加载 {name} 失败: {e}")
            continue
    
    return graphs, stats


def kfold_split(graphs, k=5, seed=42):
    """
    将图列表划分为 K 折。
    
    Args:
        graphs: 图列表
        k: 折数
        seed: 随机种子
        
    Returns:
        list of (train_indices, val_indices) tuples
    """
    n = len(graphs)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    
    fold_size = n // k
    folds = []
    
    for i in range(k):
        start = i * fold_size
        if i == k - 1:
            # 最后一折包含剩余所有数据
            end = n
        else:
            end = start + fold_size
        
        val_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]
        folds.append((train_indices, val_indices))
    
    return folds


def train_with_kfold(graphs, k=5, epochs=100, save_path=None, mode='combined',
                     use_node2vec=True, use_stepwise_reward=True, lr=0.0001,
                     seed=42, verbose=True):
    """
    使用 K-折交叉验证进行训练。
    
    Args:
        graphs: 训练图列表
        k: 折数
        epochs: 每折训练轮数
        save_path: 最终模型保存路径
        mode: 优化模式
        use_node2vec: 是否使用 Node2Vec 特征
        use_stepwise_reward: 是否使用 Step-wise Reward
        lr: 学习率
        seed: 随机种子
        verbose: 是否显示详细信息
        
    Returns:
        dict: 交叉验证结果
    """
    from src.bit_hyrl.training import evaluate_model
    from src.bit_hyrl.reward import calculate_reward
    from src.bit_hyrl.selection import hybrid_rl_select
    
    folds = kfold_split(graphs, k=k, seed=seed)
    
    fold_results = []
    all_val_rewards = []
    best_fold = -1
    best_fold_reward = -float('inf')
    
    print(f"\n{'='*60}")
    print(f"K-折交叉验证 (K={k})")
    print(f"{'='*60}")
    print(f"总图数: {len(graphs)}")
    print(f"每折训练集: ~{len(graphs) * (k-1) // k} 个图")
    print(f"每折验证集: ~{len(graphs) // k} 个图")
    print(f"{'='*60}\n")
    
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{k}")
        print(f"{'='*60}")
        print(f"训练集: {len(train_indices)} 个图")
        print(f"验证集: {len(val_indices)} 个图")
        
        # 获取当前折的训练集和验证集
        train_graphs = [graphs[i] for i in train_indices]
        val_graphs = [graphs[i] for i in val_indices]
        
        # 当前折的模型保存路径
        fold_save_path = save_path.replace('.pth', f'_fold{fold_idx+1}.pth') if save_path else None
        
        # 训练当前折
        t0 = time.time()
        result = train_offline_optimized(
            train_graphs,
            epochs=epochs,
            save_path=fold_save_path,
            mode=mode,
            use_node2vec=use_node2vec,
            use_stepwise_reward=use_stepwise_reward,
            lr=lr,
            verbose=verbose,
        )
        train_time = time.time() - t0
        
        # 在验证集上评估
        print(f"\n评估 Fold {fold_idx + 1} 在验证集上的表现...")
        val_rewards = []
        
        for G in tqdm(val_graphs, desc=f'Fold {fold_idx+1} Validation', ncols=80):
            try:
                n_nodes = G.number_of_nodes()
                k_ctrl = max(1, min(int(n_nodes * 0.1), n_nodes - 1))
                centers = hybrid_rl_select(G, k_ctrl, model_path=fold_save_path, use_ci=True)
                r = calculate_reward(G, centers, mode=mode, use_stepwise=False)
                val_rewards.append(r)
            except Exception as e:
                logger.warning(f"验证评估失败: {e}")
                continue
        
        avg_val_reward = np.mean(val_rewards) if val_rewards else 0.0
        all_val_rewards.extend(val_rewards)
        
        fold_result = {
            'fold': fold_idx + 1,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'train_reward': result['best_reward'] if result else 0.0,
            'val_reward': avg_val_reward,
            'train_time': train_time,
            'model_path': fold_save_path,
        }
        fold_results.append(fold_result)
        
        # 更新最佳折
        if avg_val_reward > best_fold_reward:
            best_fold_reward = avg_val_reward
            best_fold = fold_idx
        
        print(f"\nFold {fold_idx + 1} 结果:")
        print(f"  训练奖励: {result['best_reward']:.4f}" if result else "  训练失败")
        print(f"  验证奖励: {avg_val_reward:.4f}")
        print(f"  训练耗时: {train_time/60:.1f} 分钟")
    
    # 打印交叉验证总结
    print(f"\n{'='*60}")
    print(f"K-折交叉验证总结")
    print(f"{'='*60}")
    
    train_rewards = [r['train_reward'] for r in fold_results]
    val_rewards_per_fold = [r['val_reward'] for r in fold_results]
    
    print(f"各折训练奖励: {['%.4f' % r for r in train_rewards]}")
    print(f"各折验证奖励: {['%.4f' % r for r in val_rewards_per_fold]}")
    print(f"\n平均训练奖励: {np.mean(train_rewards):.4f} ± {np.std(train_rewards):.4f}")
    print(f"平均验证奖励: {np.mean(val_rewards_per_fold):.4f} ± {np.std(val_rewards_per_fold):.4f}")
    print(f"最佳折: Fold {best_fold + 1} (验证奖励: {best_fold_reward:.4f})")
    
    # 复制最佳折的模型作为最终模型
    if save_path and best_fold >= 0:
        best_fold_path = save_path.replace('.pth', f'_fold{best_fold+1}.pth')
        if os.path.exists(best_fold_path):
            import shutil
            shutil.copy(best_fold_path, save_path)
            print(f"\n最佳模型已保存: {save_path}")
    
    print(f"{'='*60}")
    
    return {
        'k': k,
        'fold_results': fold_results,
        'avg_train_reward': np.mean(train_rewards),
        'avg_val_reward': np.mean(val_rewards_per_fold),
        'std_train_reward': np.std(train_rewards),
        'std_val_reward': np.std(val_rewards_per_fold),
        'best_fold': best_fold + 1,
        'best_fold_reward': best_fold_reward,
        'all_val_rewards': all_val_rewards,
    }


def _plot_training_curve(history, save_path):
    """绘制训练曲线。"""
    try:
        import matplotlib.pyplot as plt
        
        epochs = [h[0] for h in history]
        rewards = [h[1] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, rewards, 'b-', linewidth=2, label='Average Reward')
        
        if len(rewards) > 10:
            window = min(10, len(rewards) // 5)
            moving_avg = []
            for i in range(len(rewards)):
                start = max(0, i - window + 1)
                moving_avg.append(sum(rewards[start:i+1]) / (i - start + 1))
            plt.plot(epochs, moving_avg, 'r--', linewidth=1.5, alpha=0.7, label=f'Moving Avg (window={window})')
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.title('BiT-HyRL Training Curve', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        best_idx = rewards.index(max(rewards))
        plt.annotate(f'Best: {rewards[best_idx]:.4f}', 
                     xy=(epochs[best_idx], rewards[best_idx]),
                     xytext=(epochs[best_idx] + len(epochs)*0.05, rewards[best_idx]),
                     arrowprops=dict(arrowstyle='->', color='green'),
                     fontsize=10, color='green')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  训练曲线已保存: {save_path}")
        
    except ImportError:
        print("  警告: matplotlib 未安装，跳过绘图")
    except Exception as e:
        print(f"  绘图失败: {e}")


def _plot_kfold_results(kfold_result, save_path):
    """绘制 K-折交叉验证结果图。"""
    try:
        import matplotlib.pyplot as plt
        
        fold_results = kfold_result['fold_results']
        folds = [r['fold'] for r in fold_results]
        train_rewards = [r['train_reward'] for r in fold_results]
        val_rewards = [r['val_reward'] for r in fold_results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(folds))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_rewards, width, label='Train Reward', color='steelblue')
        bars2 = ax.bar(x + width/2, val_rewards, width, label='Validation Reward', color='coral')
        
        ax.set_xlabel('Fold', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title(f'K-Fold Cross Validation Results (K={kfold_result["k"]})', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {f}' for f in folds])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加平均线
        ax.axhline(y=kfold_result['avg_train_reward'], color='steelblue', linestyle='--', alpha=0.7, 
                   label=f'Avg Train: {kfold_result["avg_train_reward"]:.4f}')
        ax.axhline(y=kfold_result['avg_val_reward'], color='coral', linestyle='--', alpha=0.7,
                   label=f'Avg Val: {kfold_result["avg_val_reward"]:.4f}')
        
        # 标注最佳折
        best_fold_idx = kfold_result['best_fold'] - 1
        ax.annotate(f'Best', xy=(best_fold_idx + width/2, val_rewards[best_fold_idx]),
                    xytext=(best_fold_idx + width/2 + 0.3, val_rewards[best_fold_idx] + 0.05),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=10, color='green')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  K-折验证结果图已保存: {save_path}")
        
    except ImportError:
        print("  警告: matplotlib 未安装，跳过绘图")
    except Exception as e:
        print(f"  绘图失败: {e}")


def _print_training_summary(result, elapsed_time):
    """打印训练效果总结。"""
    print("\n" + "=" * 60)
    print("训练效果总结")
    print("=" * 60)
    
    if result is None:
        print("训练失败，无结果")
        return
        
    print(f"总训练轮数: {result['total_epochs']}")
    print(f"最终奖励值: {result['final_reward']:.4f}")
    print(f"最佳奖励值: {result['best_reward']:.4f}")
    print(f"训练耗时: {elapsed_time/60:.2f} 分钟")
    print(f"模型保存位置: {result['model_path']}")
    print(f"训练历史保存位置: {result['history_path']}")
    
    history = result['history']
    if len(history) >= 10:
        first_10_avg = sum(h[1] for h in history[:10]) / 10
        last_10_avg = sum(h[1] for h in history[-10:]) / 10
        improvement = (last_10_avg - first_10_avg) / max(abs(first_10_avg), 0.001) * 100
        
        print(f"\n训练效果分析:")
        print(f"  前10轮平均奖励: {first_10_avg:.4f}")
        print(f"  后10轮平均奖励: {last_10_avg:.4f}")
        print(f"  提升幅度: {improvement:+.2f}%")
        
        if improvement > 5:
            print("  评估: 训练有效，模型性能有明显提升 ✓")
        elif improvement > 0:
            print("  评估: 训练有效，但提升幅度较小，建议增加训练轮数")
        elif improvement > -5:
            print("  评估: 训练效果持平，可能已收敛")
        else:
            print("  评估: 训练可能存在问题，建议检查参数或数据")
    
    print("=" * 60)


def _print_evaluation_summary(eval_result, model_path):
    """打印评估结果总结。"""
    print("\n" + "=" * 60)
    print("模型评估结果（最终验证集）")
    print("=" * 60)
    
    if eval_result is None:
        print("评估失败，无结果")
        return
        
    print(f"评估模型: {model_path}")
    print(f"测试图数量: {eval_result['num_graphs']}")
    print(f"平均奖励值: {eval_result['avg_reward']:.4f}")
    print(f"最小奖励值: {eval_result['min_reward']:.4f}")
    print(f"最大奖励值: {eval_result['max_reward']:.4f}")
    
    details = eval_result['details']
    connected_count = sum(1 for d in details if d['is_connected'])
    print(f"\n约束满足情况:")
    print(f"  连通图数量: {connected_count}/{len(details)}")
    
    good = sum(1 for d in details if d['reward'] > 0.7)
    medium = sum(1 for d in details if 0.4 <= d['reward'] <= 0.7)
    poor = sum(1 for d in details if d['reward'] < 0.4)
    print(f"\n性能分布:")
    print(f"  优秀 (>0.7): {good} ({good/len(details)*100:.1f}%)")
    print(f"  中等 (0.4-0.7): {medium} ({medium/len(details)*100:.1f}%)")
    print(f"  较差 (<0.4): {poor} ({poor/len(details)*100:.1f}%)")
    
    if eval_result['avg_reward'] > 0.6:
        print("\n总体评估: 模型效果良好，可用于测试 ✓")
    elif eval_result['avg_reward'] > 0.4:
        print("\n总体评估: 模型效果一般，建议继续训练")
    else:
        print("\n总体评估: 模型效果较差，建议重新训练")
    
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(
        description='BiT-HyRL Agent 训练与评估（支持 K-折交叉验证）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
数据集结构:
  - dataset/all/syn/  : 训练数据集（用于 K-折交叉验证）
  - dataset/testdata/   : 最终验证集（独立测试集）

示例:
  # 从头训练（使用全部训练数据）
  python scripts/train_bit_hyrl.py --epochs 100 --mode robustness
  
  # K-折交叉验证
  python scripts/train_bit_hyrl.py --epochs 100 --kfold 5
  
  # 增量训练
  python scripts/train_bit_hyrl.py --epochs 50 --resume
  
  # 训练后在 testdata 验证集上评估
  python scripts/train_bit_hyrl.py --epochs 100 --evaluate
        """
    )
    
    # 训练参数
    ap.add_argument('--mode', type=str, default='combined',
                    choices=['combined', 'robustness', 'csa', 'entropy', 'wcp', 'gcc', 'survival', 'integral'],
                    help='优化目标模式: gcc/survival/integral 推荐用于 GCC 优化')
    ap.add_argument('--epochs', type=int, default=100, help='训练轮数')
    ap.add_argument('--lr', type=float, default=0.001, help='学习率（优化后默认0.001）')
    ap.add_argument('--use_node2vec', action='store_true', default=True)
    ap.add_argument('--no_node2vec', dest='use_node2vec', action='store_false')
    ap.add_argument('--use_stepwise', action='store_true', default=True)
    ap.add_argument('--no_stepwise', dest='use_stepwise', action='store_false')
    
    # K-折交叉验证
    ap.add_argument('--kfold', type=int, default=0, 
                    help='K-折交叉验证的折数（0表示不使用交叉验证，使用全部数据训练）')
    ap.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 增量训练
    ap.add_argument('--resume', action='store_true', help='增量训练：加载已有模型继续训练')
    ap.add_argument('--resume_from', type=str, default=None, help='指定要加载的模型路径')
    
    # 评估
    ap.add_argument('--evaluate', action='store_true', help='训练后在 testdata 验证集上评估')
    ap.add_argument('--no_train', action='store_true', help='不训练，仅评估（需配合--evaluate）')
    
    # 输出
    ap.add_argument('--output_dir', type=str, default=None, help='输出目录')
    ap.add_argument('--no_plot', action='store_true', help='不绘制训练曲线')
    
    # 双峰拓扑重构
    ap.add_argument('--use_bimodal', action='store_true', default=True,
                    help='训练时对数据进行双峰拓扑重构（默认启用）')
    ap.add_argument('--no_bimodal', dest='use_bimodal', action='store_false',
                    help='禁用双峰拓扑重构')
    ap.add_argument('--hub_ratio', type=float, default=0.15, help='Hub节点比例')
    
    # 日志级别
    ap.add_argument('--log_level', type=str, default='WARNING',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    help='日志级别（默认WARNING）')
    
    # 性能优化参数
    ap.add_argument('--max_graphs', type=int, default=None,
                    help='最大训练图数量（采样加速，None表示全部）')
    ap.add_argument('--min_nodes', type=int, default=10,
                    help='最小节点数过滤（默认10）')
    ap.add_argument('--max_nodes', type=int, default=500,
                    help='最大节点数过滤（大图训练慢，默认500）')
    ap.add_argument('--fast', action='store_true',
                    help='快速训练模式：禁用Node2Vec，使用采样')
    
    # ============ GNN + PPO 参数 ============
    ap.add_argument('--use_gnn', action='store_true',
                    help='使用 GNN (GAT) + PPO 架构训练（推荐配合 --mode gcc）')
    ap.add_argument('--hidden_channels', type=int, default=64,
                    help='GNN 隐藏层维度（默认64）')
    ap.add_argument('--heads', type=int, default=4,
                    help='GAT 注意力头数（默认4）')
    ap.add_argument('--collect_per_epoch', type=int, default=20,
                    help='GNN 训练每轮收集的轨迹数（默认20）')
    ap.add_argument('--k_ratio', type=float, default=0.1,
                    help='控制器比例（默认0.1）')
    
    args = ap.parse_args()
    
    # 快速模式预设
    if args.fast:
        args.use_node2vec = False
        if args.max_graphs is None:
            args.max_graphs = 500
        args.max_nodes = 200
        print("启用快速训练模式: 禁用Node2Vec, 采样500图, 最大节点200")
    
    # GNN 模式预设
    if args.use_gnn:
        # GNN 模式推荐配置
        if args.mode == 'combined':
            print("提示: GNN 模式推荐使用 --mode gcc 以获得更好的效果")
        if args.lr == 0.001:
            args.lr = 3e-4  # GNN 推荐学习率
            print(f"GNN 模式：自动调整学习率为 {args.lr}")
    
    # 设置日志级别
    log_level = getattr(logging, args.log_level.upper())
    setup_logger(log_dir='logs', log_filename='training.log', 
                 level=log_level, console_level=log_level)

    print("=" * 60)
    print("BiT-HyRL Agent 训练系统")
    print("=" * 60)
    print("数据集:")
    print("  训练集: dataset/all/syn/")
    print("  验证集: dataset/testdata/")
    if args.kfold > 0:
        print(f"  验证方式: {args.kfold}-折交叉验证")
    else:
        print("  验证方式: 全量训练 + testdata 验证")
    print("=" * 60)
    
    # 确定模型路径
    out_dir = args.output_dir or config.MODEL_DIR
    os.makedirs(out_dir, exist_ok=True)
    
    # 根据训练模式选择模型文件名
    if args.use_gnn:
        model_name = 'gnn_ppo_agent.pth' if args.mode in ['combined', 'gcc'] else f'gnn_ppo_agent_{args.mode}.pth'
    else:
        model_name = 'rl_agent.pth' if args.mode == 'combined' else f'rl_agent_{args.mode}.pth'
    save_path = os.path.join(out_dir, model_name)
    
    # 确定增量训练的源模型
    resume_path = None
    if args.resume or args.resume_from:
        if args.resume_from:
            resume_path = args.resume_from
        elif os.path.exists(save_path):
            resume_path = save_path
        else:
            print(f"警告: 未找到已有模型 {save_path}，将从头训练")
    
    # 加载训练数据
    print("\n加载训练数据...")
    graphs, graph_stats = _load_training_data(
        validate=True, 
        use_bimodal=args.use_bimodal,
        hub_ratio=args.hub_ratio,
        max_graphs=args.max_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes
    )
    
    if not graphs:
        print("无可用训练数据，退出")
        return
    
    # 打印训练数据统计
    print(f"\n训练数据统计:")
    print(f"  总图数: {len(graphs)}")
    if graph_stats:
        nodes_list = [s['nodes'] for s in graph_stats]
        edges_list = [s['edges'] for s in graph_stats]
        print(f"  节点数范围: {min(nodes_list)} - {max(nodes_list)}")
        print(f"  边数范围: {min(edges_list)} - {max(edges_list)}")
        print(f"  平均节点数: {sum(nodes_list)/len(nodes_list):.1f}")
        print(f"  平均边数: {sum(edges_list)/len(edges_list):.1f}")
    
    # 训练
    result = None
    kfold_result = None
    
    if not args.no_train:
        print("\n" + "=" * 60)
        if resume_path:
            print(f"增量训练模式 - 加载: {resume_path}")
        else:
            print("从头训练模式")
        print("=" * 60)
        print(f"训练架构: {'GNN (GAT) + PPO' if args.use_gnn else 'MLP + REINFORCE'}")
        print(f"优化模式: {args.mode}")
        print(f"训练轮数: {args.epochs}")
        print(f"学习率: {args.lr}")
        if args.use_gnn:
            print(f"GAT 隐藏层: {args.hidden_channels}")
            print(f"GAT 注意力头: {args.heads}")
            print(f"控制器比例: {args.k_ratio}")
        else:
            print(f"Node2Vec: {args.use_node2vec}")
            print(f"Step-wise Reward: {args.use_stepwise}")
        print(f"双峰拓扑重构: {args.use_bimodal} (hub_ratio={args.hub_ratio})")
        
        t0 = time.time()
        
        if args.use_gnn:
            # ============ GNN + PPO 训练 ============
            from src.bit_hyrl.training import train_gnn_ppo_optimized, evaluate_gnn_model
            
            result = train_gnn_ppo_optimized(
                graphs,
                epochs=args.epochs,
                save_path=save_path,
                mode=args.mode,
                lr=args.lr,
                hidden_channels=args.hidden_channels,
                heads=args.heads,
                k_ratio=args.k_ratio,
                collect_per_epoch=args.collect_per_epoch,
                verbose=True,
            )
            elapsed = time.time() - t0
            
            # 打印训练总结
            if result:
                print("\n" + "=" * 60)
                print("GNN+PPO 训练效果总结")
                print("=" * 60)
                print(f"训练耗时: {elapsed/60:.2f} 分钟")
                print(f"最佳奖励值: {result['best_reward']:.4f}")
                print(f"最终奖励值: {result['final_reward']:.4f}")
                print(f"模型保存位置: {result['model_path']}")
                print("=" * 60)
            
            # 绘制训练曲线
            if not args.no_plot and result and result['history']:
                curve_path = save_path.replace('.pth', '_training_curve.png')
                _plot_training_curve(result['history'], curve_path)
            # 保存 GNN 训练历史 CSV，便于事后用 visualization 模块重画或对比
            if result and result.get('history'):
                hist_path = save_path.replace('.pth', '_training_history.csv')
                with open(hist_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['Epoch', 'Avg_Reward'])
                    w.writerows(result['history'])
                logger.debug(f"GNN training history saved: {hist_path}")
        
        elif args.kfold > 0:
            # K-折交叉验证
            kfold_result = train_with_kfold(
                graphs,
                k=args.kfold,
                epochs=args.epochs,
                save_path=save_path,
                mode=args.mode,
                use_node2vec=args.use_node2vec,
                use_stepwise_reward=args.use_stepwise,
                lr=args.lr,
                seed=args.seed,
                verbose=True,
            )
            elapsed = time.time() - t0
            
            # 绘制 K-折验证结果图
            if not args.no_plot and kfold_result:
                kfold_plot_path = save_path.replace('.pth', '_kfold_results.png')
                _plot_kfold_results(kfold_result, kfold_plot_path)
        else:
            # 全量训练（传统 MLP + REINFORCE）
            result = train_offline_optimized(
                graphs, 
                epochs=args.epochs, 
                save_path=save_path, 
                mode=args.mode,
                use_node2vec=args.use_node2vec, 
                use_stepwise_reward=args.use_stepwise,
                resume_from=resume_path,
                lr=args.lr,
            )
            elapsed = time.time() - t0
            
            # 打印训练总结
            _print_training_summary(result, elapsed)
            
            # 绘制训练曲线
            if not args.no_plot and result and result['history']:
                curve_path = save_path.replace('.pth', '_training_curve.png')
                _plot_training_curve(result['history'], curve_path)
    
    # 在最终验证集（testdata）上评估
    if args.evaluate:
        print("\n" + "=" * 60)
        print("在最终验证集（dataset/testdata）上评估")
        print("=" * 60)
        
        val_graphs, val_stats = _load_validation_data(
            validate=True,
            use_bimodal=args.use_bimodal,
            hub_ratio=args.hub_ratio
        )
        
        if not val_graphs:
            print("无验证数据，跳过评估")
        else:
            print(f"验证数据: {len(val_graphs)} 个图")
            
            if args.use_gnn:
                # GNN 模型评估
                from src.bit_hyrl.training import evaluate_gnn_model
                eval_result = evaluate_gnn_model(
                    save_path,
                    val_graphs,
                    mode=args.mode,
                    k_ratio=args.k_ratio,
                    verbose=True
                )
                if eval_result:
                    print("\n" + "=" * 60)
                    print("GNN 模型评估结果（最终验证集）")
                    print("=" * 60)
                    print(f"评估模型: {save_path}")
                    print(f"测试图数量: {eval_result['num_graphs']}")
                    print(f"平均奖励值: {eval_result['avg_reward']:.4f}")
                    print(f"最小奖励值: {eval_result['min_reward']:.4f}")
                    print(f"最大奖励值: {eval_result['max_reward']:.4f}")
                    print("=" * 60)
            else:
                eval_result = evaluate_model(
                    save_path, 
                    val_graphs, 
                    mode=args.mode, 
                    use_node2vec=args.use_node2vec
                )
                _print_evaluation_summary(eval_result, save_path)
    
    print("\n完成！")


if __name__ == '__main__':
    main()
