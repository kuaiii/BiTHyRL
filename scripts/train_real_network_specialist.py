# -*- coding: utf-8 -*-
"""
BiT-HyRL 真实网络专家模型训练脚本

专门针对真实网络设计的多阶段训练：
1. 阶段1：小规模图 (10-80节点)，GCC为主，学习基础
2. 阶段2：中等规模 (20-150节点)，GCC + R_GCC
3. 阶段3：全规模 (10-500节点)，完整GCC+R_GCC，多k_ratio
4. 阶段4：真实网络微调，可选数据增强

特点：
- 奖励函数：GCC + R_GCC (曲线下面积)
- 多控制器占比：k_ratio in [0.08, 0.10, 0.12]
- 多种子训练：增强泛化
- 数据增强：真实网络 + 合成网络（仿造节点/边数）
- 训练可视化：奖励曲线、R值对比

用法:
    # 1. 生成数据增强（可选）
    python scripts/generate_real_network_augmentation.py --output dataset/all/real_augmented
    
    # 2. 单种子训练
    python scripts/train_real_network_specialist.py --epochs 400 --seed 42
    
    # 3. 多种子训练（推荐）
    python scripts/train_real_network_specialist.py --epochs 400 --seeds 42,123,456,789,2024
    
    # 4. 使用数据增强
    python scripts/train_real_network_specialist.py --use-augmentation --epochs 400
"""

import os
import sys
import argparse
import glob
import random
import json
import time

import torch
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.bit_hyrl import config
from src.bit_hyrl.gnn_model import DynamicGATPolicy
from src.bit_hyrl.reward import calculate_phase4_enhanced_reward
from src.topology.generators import load_graph as _load_graph_file
from scripts.train_curriculum import (
    load_training_graphs,
    CurriculumPPOTrainer,
)

DEVICE = config.DEVICE


# ============== 数据加载 ==============

def load_combined_training_data(real_dir, aug_dir=None, real_weight=0.7, max_total=2000):
    """加载真实网络 + 增强数据"""
    real_graphs = load_training_graphs(real_dir, min_nodes=10, max_nodes=500)
    
    all_graphs = list(real_graphs)
    
    if aug_dir and os.path.exists(aug_dir):
        aug_graphs = load_training_graphs(aug_dir, min_nodes=10, max_nodes=500, max_graphs=800)
        if aug_graphs:
            n_real = len(real_graphs)
            n_aug = min(len(aug_graphs), max(0, max_total - n_real))
            if n_aug > 0:
                sampled_aug = random.sample(aug_graphs, n_aug)
                all_graphs = real_graphs + sampled_aug
                random.shuffle(all_graphs)
    
    if max_total and len(all_graphs) > max_total:
        all_graphs = random.sample(all_graphs, max_total)
    
    return all_graphs


# ============== 奖励函数（纯GCC + R_GCC）==============

def calculate_gcc_r_reward(G, centers, attack_range=(0, 0.5), num_samples=25):
    """
    纯 GCC 和 R_GCC 奖励
    
    R = 0.5 * 关键点GCC(10%,20%,30%) + 0.5 * R值(0-50%攻击)
    """
    if not centers:
        return 0.0
    
    centers_set = set(centers)
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    
    degrees = dict(G.degree())
    degree_targets = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    
    attack_ratios = np.linspace(attack_range[0], attack_range[1], num_samples)
    gcc_values = []
    
    for ratio in attack_ratios:
        if ratio == 0:
            gcc_values.append(1.0)
            continue
        num_remove = max(1, int(n * ratio))
        remaining_nodes = set(G.nodes()) - set(degree_targets[:num_remove])
        remaining_centers = centers_set & remaining_nodes
        if remaining_nodes and remaining_centers:
            G_sub = G.subgraph(remaining_nodes)
            if G_sub.number_of_nodes() > 0:
                comps = list(nx.connected_components(G_sub))
                controlled = [len(c) for c in comps if not remaining_centers.isdisjoint(c)]
                gcc_ratio = max(controlled) / n if controlled else 0.0
            else:
                gcc_ratio = 0.0
        else:
            gcc_ratio = 0.0
        gcc_values.append(gcc_ratio)
    
    r_value = np.trapz(gcc_values, attack_ratios) / (attack_range[1] - attack_range[0])
    
    # 关键点
    idx_10 = int(0.10 / (attack_range[1] - attack_range[0]) * (num_samples - 1))
    idx_20 = int(0.20 / (attack_range[1] - attack_range[0]) * (num_samples - 1))
    idx_30 = int(0.30 / (attack_range[1] - attack_range[0]) * (num_samples - 1))
    idx_10 = min(idx_10, len(gcc_values) - 1)
    idx_20 = min(idx_20, len(gcc_values) - 1)
    idx_30 = min(idx_30, len(gcc_values) - 1)
    
    key_gcc = (gcc_values[idx_10] * 0.2 + gcc_values[idx_20] * 0.3 + gcc_values[idx_30] * 0.5)
    
    return 0.5 * r_value + 0.5 * key_gcc


# ============== 单阶段训练 ==============

def train_stage(graphs, model, trainer, stage_config, k_ratios=(0.08, 0.10, 0.12), verbose=True):
    """训练单个阶段"""
    epochs = stage_config['epochs']
    reward_fn = stage_config['reward_fn']
    
    history = []
    best_reward = -float('inf')
    best_state = None
    
    pbar = tqdm(range(epochs), desc=f"Stage {stage_config['name']}", disable=not verbose)
    
    for epoch in pbar:
        epoch_rewards = []
        sampled = random.sample(graphs, min(stage_config.get('collect_per_epoch', 35), len(graphs)))
        
        for G in sampled:
            if G.number_of_nodes() < 3:
                continue
            k_ratio = random.choice(k_ratios)
            k = max(1, int(G.number_of_nodes() * k_ratio))
            try:
                _, reward = trainer.collect_trajectory(G, k, reward_fn)
                epoch_rewards.append(reward)
            except Exception:
                continue
        
        stats = trainer.update()
        avg = np.mean(epoch_rewards) if epoch_rewards else 0
        history.append({'epoch': epoch + 1, 'reward': avg, 'loss': stats.get('policy_loss', 0)})
        
        if avg > best_reward:
            best_reward = avg
            best_state = model.state_dict().copy()
        
        pbar.set_postfix({'reward': f'{avg:.4f}', 'best': f'{best_reward:.4f}'})
    
    if best_state:
        model.load_state_dict(best_state)
    
    return history, best_reward


# ============== 可视化 ==============

def plot_training_curves(all_histories, save_dir, seed=None):
    """绘制训练曲线"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, (stage_name, history) in enumerate(all_histories.items()):
        if not history:
            continue
        epochs = [h['epoch'] for h in history]
        rewards = [h['reward'] for h in history]
        losses = [h.get('loss', 0) for h in history]
        
        axes[0, 0].plot(epochs, rewards, label=stage_name, color=colors[i % 10])
        axes[0, 1].plot(epochs, losses, label=stage_name, color=colors[i % 10])
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Policy Loss')
    axes[0, 1].set_title('Policy Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 合并所有阶段的累积奖励曲线
    all_epochs = []
    all_rewards = []
    offset = 0
    for stage_name, history in all_histories.items():
        if not history:
            continue
        for h in history:
            all_epochs.append(offset + h['epoch'])
            all_rewards.append(h['reward'])
        offset += max((e['epoch'] for e in history), default=0)
    
    if all_epochs:
        axes[1, 0].plot(all_epochs, all_rewards, 'b-', alpha=0.7)
        axes[1, 0].set_xlabel('Cumulative Epoch')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_title('Full Training Reward Curve')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 滑动平均
    if len(all_rewards) > 10:
        window = min(20, len(all_rewards) // 5)
        smoothed = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(range(len(smoothed)), smoothed, 'g-', linewidth=2, label='Smoothed')
        axes[1, 1].plot(all_rewards, 'b-', alpha=0.3, label='Raw')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].set_title('Smoothed Reward')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    suffix = f'_seed{seed}' if seed is not None else ''
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_curves{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============== 主训练流程 ==============

def run_training(
    real_dir='dataset/all/real',
    aug_dir='dataset/all/real_augmented',
    use_augmentation=True,
    epochs_per_stage=(80, 100, 120, 100),
    k_ratios=(0.08, 0.10, 0.12),
    lr=3e-4,
    seed=42,
    save_dir='models',
    results_dir='results/training_real_specialist',
    resume_from=None,
    verbose=True,
):
    """运行完整多阶段训练"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 加载数据
    if use_augmentation and aug_dir and os.path.exists(aug_dir):
        graphs = load_combined_training_data(real_dir, aug_dir, real_weight=0.7)
        if verbose:
            print(f"Loaded {len(graphs)} graphs (real + augmented)")
    else:
        graphs = load_training_graphs(real_dir, min_nodes=10, max_nodes=500)
        if verbose:
            print(f"Loaded {len(graphs)} real network graphs")
    
    if not graphs:
        raise ValueError("No graphs found!")
    
    # 模型
    model_config = {
        'in_channels': 64, 'hidden_channels': 96, 'scale_encoding_dim': 8,
        'dynamic_feature_dim': 6, 'heads': 4, 'num_layers': 3, 'dropout': 0.15,
    }
    model = DynamicGATPolicy(**model_config).to(DEVICE)
    
    if resume_from and os.path.exists(resume_from):
        try:
            ck = torch.load(resume_from, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck['model_state_dict'], strict=False)
            if verbose:
                print(f"Resumed from {resume_from}")
        except Exception as e:
            if verbose:
                print(f"Resume failed: {e}")
    
    # 阶段配置
    def _gcc_r_reward(G, centers):
        return calculate_gcc_r_reward(G, centers)
    
    stage_configs = [
        {'name': 'Stage1-Small', 'epochs': epochs_per_stage[0], 'reward_fn': _gcc_r_reward,
         'collect_per_epoch': 30, 'graphs': [g for g in graphs if g.number_of_nodes() <= 80] or graphs},
        {'name': 'Stage2-Medium', 'epochs': epochs_per_stage[1], 'reward_fn': _gcc_r_reward,
         'collect_per_epoch': 35, 'graphs': [g for g in graphs if 20 <= g.number_of_nodes() <= 150] or graphs},
        {'name': 'Stage3-Full', 'epochs': epochs_per_stage[2], 'reward_fn': _gcc_r_reward,
         'collect_per_epoch': 40, 'graphs': graphs},
        {'name': 'Stage4-Finetune', 'epochs': epochs_per_stage[3], 'reward_fn': calculate_phase4_enhanced_reward,
         'collect_per_epoch': 45, 'graphs': graphs},
    ]
    
    trainer = CurriculumPPOTrainer(model, model_type='dynamic_gat', lr=lr)
    
    all_histories = {}
    total_epochs = 0
    
    for sc in stage_configs:
        stage_graphs = sc.get('graphs', graphs)
        if not stage_graphs:
            stage_graphs = graphs
        sc['graphs'] = stage_graphs
        
        history, best_reward = train_stage(
            stage_graphs, model, trainer, sc, k_ratios=k_ratios, verbose=verbose
        )
        all_histories[sc['name']] = history
        total_epochs += sc['epochs']
        
        # 每阶段后降低学习率
        for pg in trainer.optimizer.param_groups:
            pg['lr'] *= 0.8
    
    # 保存
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'real_network_specialist_seed{seed}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'seed': seed,
        'epochs': total_epochs,
        'histories': all_histories,
        'k_ratios': list(k_ratios),
    }, save_path)
    
    # 可视化
    os.makedirs(results_dir, exist_ok=True)
    plot_training_curves(all_histories, results_dir, seed)
    
    # 保存历史 JSON
    with open(os.path.join(results_dir, f'training_history_seed{seed}.json'), 'w') as f:
        json.dump(all_histories, f, indent=2)
    
    if verbose:
        print(f"\nModel saved: {save_path}")
        print(f"Training curves: {results_dir}/training_curves_seed{seed}.png")
    
    return save_path, all_histories


# ============== 入口 ==============

def main():
    parser = argparse.ArgumentParser(description='Train Real Network Specialist Model')
    parser.add_argument('--real-dir', type=str, default='dataset/all/real')
    parser.add_argument('--aug-dir', type=str, default='dataset/all/real_augmented')
    parser.add_argument('--use-augmentation', action='store_true', default=True,
                        help='Use augmented synthetic data')
    parser.add_argument('--no-augmentation', action='store_false', dest='use_augmentation')
    parser.add_argument('--epochs', type=int, default=400,
                        help='Total epochs (split across stages)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds for multi-seed training, e.g. 42,123,456')
    parser.add_argument('--k-ratios', type=str, default='0.08,0.10,0.12')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--results-dir', type=str, default='results/training_real_specialist')
    parser.add_argument('--verbose', action='store_true', default=True)
    
    args = parser.parse_args()
    
    k_ratios = tuple(float(x) for x in args.k_ratios.split(','))
    epochs_per_stage = (
        args.epochs // 4,
        args.epochs // 4,
        args.epochs // 4,
        args.epochs - 3 * (args.epochs // 4),
    )
    
    seeds = [args.seed]
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(',')]
    
    print("="*60)
    print("BiT-HyRL Real Network Specialist Training")
    print("="*60)
    print(f"Seeds: {seeds}")
    print(f"K ratios: {k_ratios}")
    print(f"Epochs per stage: {epochs_per_stage}")
    print(f"Use augmentation: {args.use_augmentation}")
    print("="*60)
    
    paths = []
    for seed in seeds:
        run_training(
            real_dir=args.real_dir,
            aug_dir=args.aug_dir,
            use_augmentation=args.use_augmentation,
            epochs_per_stage=epochs_per_stage,
            k_ratios=k_ratios,
            lr=args.lr,
            seed=seed,
            save_dir=args.save_dir,
            results_dir=args.results_dir,
            resume_from=args.resume if seed == seeds[0] else None,
            verbose=args.verbose,
        )
        paths.append(os.path.join(args.save_dir, f'real_network_specialist_seed{seed}.pth'))
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    for p in paths:
        print(f"  {p}")
    print("\nEvaluate on Colt/Chinanet/UsCarrier:")
    print(f"  python main.py -d Colt -m {paths[0]}")
    print(f"  python main.py -d Chinanet -m {paths[0]}")
    print(f"  python main.py -d UsCarrier -m {paths[0]}")
    print("="*60)


if __name__ == '__main__':
    main()
