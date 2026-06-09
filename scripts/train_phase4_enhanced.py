# -*- coding: utf-8 -*-
"""
BiT-HyRL 阶段4增强训练脚本

专门针对真实网络设计的激进优化策略，完全聚焦于GCC和R值。

设计特点：
1. 使用phase 5奖励（完整0-100%攻击的R值积分）
2. 更低的学习率（3e-5）进行精细微调
3. 更多的epoch和trajectory采样
4. 数据增强：真实网络 + 部分合成网络
5. 自适应学习率调度
6. 早停机制

Usage:
    # 基础增强训练
    python scripts/train_phase4_enhanced.py --epochs 150
    
    # 使用更多trajectory
    python scripts/train_phase4_enhanced.py --epochs 200 --collect-per-epoch 50
    
    # 从特定模型继续
    python scripts/train_phase4_enhanced.py --resume src/train/v1/checkpoints/curriculum_phase4_dynamic_gat.pth --epochs 100
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

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.bit_hyrl import config
from src.bit_hyrl.gnn_model import (
    DynamicGATPolicy,
    GraphTransformerPolicy,
    UnifiedGATPolicy,
)
from src.bit_hyrl.reward import calculate_phase4_enhanced_reward
from src.topology.generators import load_graph as _load_graph_file

# 导入课程训练的工具函数
from scripts.train_curriculum import (
    load_training_graphs,
    filter_graphs_by_size,
    CurriculumPPOTrainer,
)

DEVICE = config.DEVICE


def load_mixed_training_data(real_network_dir, synthetic_dir=None, mix_ratio=0.7):
    """
    加载混合训练数据：真实网络 + 合成网络
    
    Args:
        real_network_dir: 真实网络目录
        synthetic_dir: 合成网络目录（可选）
        mix_ratio: 真实网络占比（默认70%）
        
    Returns:
        graphs: 混合图列表
    """
    print(f"Loading mixed training data...")
    
    # 加载真实网络（10-1000节点）
    real_graphs = load_training_graphs(real_network_dir, min_nodes=10, max_nodes=1000)
    print(f"  Loaded {len(real_graphs)} real network graphs")
    
    # 可选：加载部分合成网络（相似规模）
    synthetic_graphs = []
    if synthetic_dir and os.path.exists(synthetic_dir):
        synthetic_graphs = load_training_graphs(synthetic_dir, min_nodes=10, max_nodes=1000, max_graphs=500)
        print(f"  Loaded {len(synthetic_graphs)} synthetic graphs")
    
    # 混合数据
    if synthetic_graphs:
        num_real = int(len(real_graphs) * mix_ratio)
        num_synthetic = len(real_graphs) - num_real
        
        mixed_graphs = (
            random.sample(real_graphs, min(num_real, len(real_graphs))) +
            random.sample(synthetic_graphs, min(num_synthetic, len(synthetic_graphs)))
        )
        random.shuffle(mixed_graphs)
        print(f"  Mixed: {num_real} real + {num_synthetic} synthetic = {len(mixed_graphs)} total")
    else:
        mixed_graphs = real_graphs
        print(f"  Using {len(mixed_graphs)} real network graphs only")
    
    return mixed_graphs


def train_phase4_enhanced(
    graphs,
    epochs=150,
    save_path='src/train/v1/checkpoints/curriculum_phase4_enhanced_dynamic_gat.pth',
    model_type='dynamic_gat',
    resume_from=None,
    lr=3e-5,
    k_ratio=0.1,
    collect_per_epoch=40,
    use_early_stopping=True,
    patience=30,
    verbose=True,
):
    """
    阶段4增强训练
    
    Args:
        graphs: 训练图列表
        epochs: 训练轮数
        save_path: 模型保存路径
        model_type: 模型类型
        resume_from: 继续训练的模型路径
        lr: 学习率（默认3e-5，更低以精细微调）
        k_ratio: 控制器比例
        collect_per_epoch: 每轮采样图数量（增加以获得更多数据）
        use_early_stopping: 是否使用早停
        patience: 早停耐心值
        verbose: 是否显示进度
        
    Returns:
        dict: 训练结果
    """
    # 使用phase 5奖励（增强版）
    reward_fn = calculate_phase4_enhanced_reward
    
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
    
    # 创建模型
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
    epochs_no_improve = 0
    
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
    
    # 创建训练器（使用phase 5）
    trainer = CurriculumPPOTrainer(model, model_type=model_type, lr=lr)
    
    # 自适应学习率调度器（PyTorch 较旧版本可能不支持 verbose 参数）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-6
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 4 Enhanced Training")
        print(f"{'='*60}")
        print(f"  Model: {model_type}")
        print(f"  Epochs: {epochs}")
        print(f"  Graphs: {len(graphs)}")
        print(f"  Learning rate: {lr}")
        print(f"  Collect per epoch: {collect_per_epoch}")
        print(f"  Reward: Phase 4 Enhanced (Full R-value + Key GCC)")
        print(f"  Early stopping: {use_early_stopping} (patience={patience})")
        print(f"  Device: {DEVICE}")
        print(f"{'='*60}\n")
    
    # 训练循环
    epoch_pbar = tqdm(range(epochs), desc=f"Phase 4 Enhanced", disable=not verbose)
    
    for epoch in epoch_pbar:
        epoch_rewards = []
        
        # 采样图（每个epoch从不同规模采样）
        sampled_graphs = random.sample(graphs, min(collect_per_epoch, len(graphs)))
        
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
        current_lr = trainer.optimizer.param_groups[0]['lr']
        history.append([start_epoch + epoch + 1, avg_reward, current_lr])
        
        # 保存最佳模型
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # 学习率调度
        scheduler.step(avg_reward)
        
        # 更新进度条
        epoch_pbar.set_postfix({
            'reward': f'{avg_reward:.4f}',
            'best': f'{best_reward:.4f}',
            'lr': f'{current_lr:.1e}',
            'no_improve': epochs_no_improve,
        })
        
        # 早停检查
        if use_early_stopping and epochs_no_improve >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {start_epoch + epoch + 1}")
                print(f"No improvement for {patience} epochs")
            break
    
    # 保存模型
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        checkpoint = {
            'model_state_dict': best_model_state if best_model_state else model.state_dict(),
            'model_type': model_type,
            'model_config': model_config,
            'phase': 5,  # Phase 4 Enhanced
            'best_reward': best_reward,
            'history': history,
        }
        torch.save(checkpoint, save_path)
        
        if verbose:
            print(f"\nModel saved to {save_path}")
            print(f"Best reward: {best_reward:.4f}")
            print(f"Total epochs: {len(history)}")
    
    return {
        'history': history,
        'best_reward': best_reward,
        'final_reward': history[-1][1] if history else 0,
        'model_path': save_path,
    }


def main():
    parser = argparse.ArgumentParser(description='BiT-HyRL Phase 4 Enhanced Training')
    
    # 数据参数
    parser.add_argument('--real-network-dir', type=str, default='dataset/all/real',
                        help='Real network data directory')
    parser.add_argument('--synthetic-dir', type=str, default='dataset/all/syn',
                        help='Synthetic network directory (optional, for data augmentation)')
    parser.add_argument('--use-synthetic', action='store_true',
                        help='Include synthetic networks for data augmentation')
    parser.add_argument('--mix-ratio', type=float, default=0.7,
                        help='Ratio of real networks in mixed data (if using synthetic)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate (default: 3e-5 for fine-tuning)')
    parser.add_argument('--k-ratio', type=float, default=0.1,
                        help='Controller ratio')
    parser.add_argument('--collect-per-epoch', type=int, default=40,
                        help='Graphs to sample per epoch')
    
    # 早停参数
    parser.add_argument('--early-stopping', action='store_true', default=True,
                        help='Use early stopping')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    
    # 模型参数
    parser.add_argument('--model-type', type=str, default='dynamic_gat',
                        choices=['dynamic_gat', 'graph_transformer', 'unified_gat'],
                        help='Model architecture')
    
    # 保存/加载
    parser.add_argument('--save-path', type=str, default='src/train/v1/checkpoints/curriculum_phase4_enhanced_dynamic_gat.pth',
                        help='Model save path')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (default: auto-detect phase 4 or phase 3)')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Save directory')
    
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
    
    # 自动检测resume模型
    if args.resume is None:
        # 优先尝试phase 4模型
        phase4_path = os.path.join(args.save_dir, f'curriculum_phase4_{args.model_type}.pth')
        if os.path.exists(phase4_path):
            args.resume = phase4_path
            print(f"Auto-detected phase 4 model: {args.resume}")
        else:
            # 尝试phase 3模型
            phase3_path = os.path.join(args.save_dir, f'curriculum_phase3_{args.model_type}.pth')
            if os.path.exists(phase3_path):
                args.resume = phase3_path
                print(f"Auto-detected phase 3 model: {args.resume}")
            else:
                # 尝试final模型
                final_path = os.path.join(args.save_dir, f'curriculum_final_{args.model_type}.pth')
                if os.path.exists(final_path):
                    args.resume = final_path
                    print(f"Auto-detected final model: {args.resume}")
    
    # 加载训练数据
    if args.use_synthetic:
        graphs = load_mixed_training_data(
            args.real_network_dir,
            args.synthetic_dir,
            args.mix_ratio
        )
    else:
        graphs = load_training_graphs(args.real_network_dir, min_nodes=10, max_nodes=1000)
    
    if not graphs:
        print(f"Error: No graphs found in {args.real_network_dir}")
        return
    
    print(f"\nStarting Phase 4 Enhanced Training...")
    print(f"Total graphs: {len(graphs)}")
    
    # 训练
    result = train_phase4_enhanced(
        graphs=graphs,
        epochs=args.epochs,
        save_path=args.save_path,
        model_type=args.model_type,
        resume_from=args.resume,
        lr=args.lr,
        k_ratio=args.k_ratio,
        collect_per_epoch=args.collect_per_epoch,
        use_early_stopping=args.early_stopping,
        patience=args.patience,
        verbose=args.verbose,
    )
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best reward: {result['best_reward']:.4f}")
    print(f"Final reward: {result['final_reward']:.4f}")
    print(f"Model saved: {result['model_path']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
