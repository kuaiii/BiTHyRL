# -*- coding: utf-8 -*-
"""
深层 GAT + PPO 训练脚本

特性:
- 使用 Node2Vec 嵌入作为节点特征 (128维)
- 5层深层 GAT 捕捉高阶结构信息
- 全部由 GAT+RL 选择控制器，不使用 CI 算法
- 支持残差连接和跳跃连接

用法:
  # 训练新模型
  python scripts/train_deep_gat.py --epochs 200 --mode gcc
  
  # 自定义参数
  python scripts/train_deep_gat.py --epochs 300 --embed_dim 128 --hidden 128 --layers 5 --heads 8
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
from tqdm import tqdm
from src.topology.generators import load_graph
from src.bit_hyrl import config
from src.bit_hyrl.training import train_gnn_ppo_optimized, evaluate_gnn_model
from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def load_training_data(train_dir, max_graphs=None, min_nodes=10, max_nodes=500, verbose=True):
    """
    从训练目录加载图数据
    
    Args:
        train_dir: 训练数据目录
        max_graphs: 最大加载图数量
        min_nodes: 最小节点数
        max_nodes: 最大节点数
        verbose: 是否显示详细信息
        
    Returns:
        graphs: 图列表
    """
    if not os.path.exists(train_dir):
        print(f"错误: 训练数据目录不存在: {train_dir}")
        return []
    
    graphs = []
    failed = 0
    skipped_size = 0
    
    # 收集所有 .gml 文件
    all_files = []
    for f in os.listdir(train_dir):
        if f.endswith('.gml'):
            all_files.append(os.path.join(train_dir, f))
    
    print(f"发现 {len(all_files)} 个 .gml 文件")
    
    # 随机打乱并限制数量
    if max_graphs and max_graphs < len(all_files):
        random.shuffle(all_files)
        all_files = all_files[:max_graphs]
        print(f"随机采样 {max_graphs} 个图用于训练")
    
    print(f"节点数过滤: {min_nodes} - {max_nodes}")
    
    for filepath in tqdm(all_files, desc='Loading training data', ncols=80, disable=not verbose):
        try:
            G, _ = load_graph(filepath, verbose=False)
            
            if G is None or G.number_of_nodes() < 2:
                failed += 1
                continue
            
            n_nodes = G.number_of_nodes()
            
            # 节点数过滤
            if n_nodes < min_nodes or n_nodes > max_nodes:
                skipped_size += 1
                continue
            
            # 确保图连通
            if not nx.is_connected(G):
                # 取最大连通分量
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
            
            graphs.append(G)
            
        except Exception as e:
            failed += 1
            continue
    
    print(f"加载完成: {len(graphs)} 个有效图")
    if skipped_size > 0:
        print(f"  因节点数过滤跳过: {skipped_size} 个")
    if failed > 0:
        print(f"  加载失败: {failed} 个")
    
    return graphs


def main():
    parser = argparse.ArgumentParser(
        description='深层 GAT + PPO 训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 数据参数
    parser.add_argument('--train_dir', type=str, default='dataset/all/syn',
                        help='训练数据目录')
    parser.add_argument('--max_graphs', type=int, default=None,
                        help='最大训练图数量')
    parser.add_argument('--min_nodes', type=int, default=10,
                        help='最小节点数')
    parser.add_argument('--max_nodes', type=int, default=300,
                        help='最大节点数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='学习率')
    parser.add_argument('--mode', type=str, default='gcc',
                        choices=['gcc', 'multi_attack', 'survival'],
                        help='奖励模式')
    parser.add_argument('--k_ratio', type=float, default=0.1,
                        help='控制器比例')
    parser.add_argument('--collect_per_epoch', type=int, default=30,
                        help='每轮收集的轨迹数')
    
    # 模型架构参数
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Node2Vec 嵌入维度')
    parser.add_argument('--hidden', type=int, default=128,
                        help='隐藏层维度')
    parser.add_argument('--heads', type=int, default=8,
                        help='GAT 注意力头数')
    parser.add_argument('--layers', type=int, default=5,
                        help='GAT 层数')
    
    # 输出与增量训练
    parser.add_argument('--output', type=str, default=None,
                        help='模型保存路径')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='增量训练：从此 checkpoint 继续训练（如 src/train/v1/checkpoints/deep_gat_gcc_L5_H8_D128.pth）')
    parser.add_argument('--evaluate', action='store_true',
                        help='训练后评估')

    # 日志
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.WARNING
    setup_logger(log_dir='logs', log_filename='deep_gat_training.log',
                 level=log_level, console_level=log_level)
    
    print("=" * 70)
    print("深层 GAT + PPO 训练")
    print("=" * 70)
    print(f"模型架构:")
    print(f"  - Node2Vec 嵌入: {args.embed_dim}维")
    print(f"  - GAT 层数: {args.layers}")
    print(f"  - 注意力头数: {args.heads}")
    print(f"  - 隐藏层维度: {args.hidden}")
    print(f"  - 全部由 GAT+RL 选择控制器（无 CI 算法）")
    print("=" * 70)
    
    # 加载训练数据
    print("\n加载训练数据...")
    graphs = load_training_data(
        args.train_dir,
        max_graphs=args.max_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
    )
    
    if not graphs:
        print("无可用训练数据，退出")
        return
    
    # 打印数据统计
    nodes_list = [G.number_of_nodes() for G in graphs]
    print(f"\n训练数据统计:")
    print(f"  总图数: {len(graphs)}")
    print(f"  节点数范围: {min(nodes_list)} - {max(nodes_list)}")
    print(f"  平均节点数: {np.mean(nodes_list):.1f}")
    
    # 确定保存路径
    if args.output:
        save_path = args.output
    else:
        save_path = os.path.join(
            config.MODEL_DIR, 
            f'deep_gat_{args.mode}_L{args.layers}_H{args.heads}_D{args.embed_dim}.pth'
        )
    
    # 训练（支持增量训练 --resume_from）
    if args.resume_from:
        print(f"\n增量训练: 从 {args.resume_from} 加载")
    print(f"\n开始训练...")
    t0 = time.time()

    result = train_gnn_ppo_optimized(
        graphs=graphs,
        epochs=args.epochs,
        save_path=save_path,
        mode=args.mode,
        lr=args.lr,
        embed_dim=args.embed_dim,
        hidden_channels=args.hidden,
        heads=args.heads,
        num_layers=args.layers,
        k_ratio=args.k_ratio,
        collect_per_epoch=args.collect_per_epoch,
        verbose=True,
        resume_from=args.resume_from,
    )
    
    elapsed = time.time() - t0
    
    # 打印训练结果
    if result:
        print("\n" + "=" * 70)
        print("训练完成!")
        print("=" * 70)
        print(f"  训练耗时: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  最佳奖励: {result['best_reward']:.4f}")
        print(f"  最终奖励: {result['final_reward']:.4f}")
        print(f"  模型保存: {result['model_path']}")
        print("=" * 70)
    
    # 评估
    if args.evaluate and result:
        print("\n评估模型...")
        eval_result = evaluate_gnn_model(
            result['model_path'],
            graphs[:min(50, len(graphs))],  # 使用部分数据评估
            mode=args.mode,
            k_ratio=args.k_ratio,
            verbose=True,
        )
    
    print("\n完成!")


if __name__ == '__main__':
    main()
