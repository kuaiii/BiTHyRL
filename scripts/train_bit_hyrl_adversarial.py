#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BiT-HyRL 对抗式多攻击训练入口。

使用 network_dismantling 中多种拆解方法生成多样化攻击序列，
训练控制器选择策略的泛化能力。

数据集:
  - dataset/all/syn/   : 训练数据集
  - dataset/testdata/  : 最终验证集

用法:
  python scripts/train_bit_hyrl_adversarial.py --epochs 100 --lr 3e-4
  python scripts/train_bit_hyrl_adversarial.py --epochs 50 --max-graphs 200
  python scripts/train_bit_hyrl_adversarial.py --resume src/train/v1/checkpoints/gnn_ppo_agent_adversarial.pth
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
import numpy as np
import networkx as nx
from tqdm import tqdm

from src.topology.generators import load_graph
from src.topology.reconstruction import create_bimodal_network_exact
from src.bit_hyrl import config
from src.bit_hyrl.training import train_gnn_ppo_optimized
from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def load_training_graphs(
    data_dir=None,
    max_graphs=None,
    min_nodes=20,
    max_nodes=500,
    use_bimodal=True,
    hub_ratio=0.15,
    seed=42,
):
    """从 dataset/all/syn/ 加载训练图列表。"""
    if data_dir is None:
        data_dir = os.path.join(ROOT, 'dataset', 'all', 'syn')

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"训练数据目录不存在: {data_dir}")

    gml_files = [f for f in os.listdir(data_dir) if f.endswith('.gml')]
    gml_files.sort()
    random.seed(seed)
    random.shuffle(gml_files)

    if max_graphs and max_graphs < len(gml_files):
        gml_files = gml_files[:max_graphs]

    graphs = []
    stats = []
    skipped = 0
    failed = 0
    bimodal_failed = 0

    for filename in tqdm(gml_files, desc='Loading graphs', ncols=80):
        filepath = os.path.join(data_dir, filename)
        name = os.path.splitext(filename)[0]
        try:
            G_original, _ = load_graph(filepath, verbose=False)
            if G_original is None or G_original.number_of_nodes() < min_nodes:
                skipped += 1
                continue
            if G_original.number_of_nodes() > max_nodes:
                skipped += 1
                continue
            if not nx.is_connected(G_original):
                skipped += 1
                continue

            original_nodes = G_original.number_of_nodes()
            original_edges = G_original.number_of_edges()

            if use_bimodal:
                hub_num = max(1, int(G_original.number_of_nodes() * hub_ratio))
                G = create_bimodal_network_exact(
                    G_original, hub_num=hub_num, seed=len(graphs)
                )
                if (
                    G.number_of_nodes() != original_nodes
                    or G.number_of_edges() != original_edges
                    or not nx.is_connected(G)
                ):
                    bimodal_failed += 1
                    G = G_original
            else:
                G = G_original

            graphs.append(G)
            stats.append(
                {
                    'name': name,
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges(),
                }
            )
        except Exception as e:
            failed += 1
            logger.warning(f"加载 {name} 失败: {e}")

    print(f"加载完成: {len(graphs)} 个有效图")
    if skipped:
        print(f"  跳过: {skipped} 个")
    if failed:
        print(f"  失败: {failed} 个")
    if bimodal_failed:
        print(f"  双峰重构失败(已降级): {bimodal_failed} 个")
    return graphs, stats


def save_training_history(history, save_path):
    """保存训练历史到 CSV。"""
    hist_path = save_path.replace('.pth', '_training_history.csv')
    with open(hist_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Avg_Reward'])
        writer.writerows(history)
    print(f"训练历史已保存: {hist_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='BiT-HyRL Adversarial Multi-Attack Training'
    )
    parser.add_argument('--data-dir', type=str, default=None,
                        help='训练数据目录 (默认: dataset/all/syn)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数 (默认: 50)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='学习率 (默认: 3e-4)')
    parser.add_argument('--k-ratio', type=float, default=0.1,
                        help='控制器比例 (默认: 0.1)')
    parser.add_argument('--hidden-channels', type=int, default=128,
                        help='GAT 隐藏层维度 (默认: 128)')
    parser.add_argument('--heads', type=int, default=8,
                        help='GAT 注意力头数 (默认: 8)')
    parser.add_argument('--num-layers', type=int, default=5,
                        help='GAT 层数 (默认: 5)')
    parser.add_argument('--embed-dim', type=int, default=128,
                        help='Node2Vec 嵌入维度 (默认: 128)')
    parser.add_argument('--collect-per-epoch', type=int, default=20,
                        help='每轮收集轨迹数 (默认: 20)')
    parser.add_argument('--max-graphs', type=int, default=None,
                        help='最大使用图数量 (None 表示全部)')
    parser.add_argument('--min-nodes', type=int, default=20,
                        help='最小节点数过滤 (默认: 20)')
    parser.add_argument('--max-nodes', type=int, default=500,
                        help='最大节点数过滤 (默认: 500)')
    parser.add_argument('--use-bimodal', action='store_true', default=True,
                        help='启用双峰拓扑重构 (默认: True)')
    parser.add_argument('--hub-ratio', type=float, default=0.15,
                        help='双峰拓扑 hub 比例 (默认: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='模型保存路径 (默认: src/train/v1/checkpoints/gnn_ppo_agent_adversarial.pth)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从已有 checkpoint 恢复训练')
    parser.add_argument('--no-plot', action='store_true',
                        help='不绘制训练曲线')
    parser.add_argument('--attack-ratio', type=float, default=0.1,
                        help='奖励函数攻击比例 (默认: 0.1)')
    parser.add_argument('--aggregation', type=str, default='min',
                        choices=['min', 'mean', 'max'],
                        help='多攻击奖励聚合方式 (默认: min，即最脆弱攻击)')
    parser.add_argument('--sample-methods', type=int, default=3,
                        help='每次奖励计算随机采样的攻击方法数 (默认: 3, 0 表示使用全部)')
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.save_path is None:
        args.save_path = os.path.join(
            config.MODEL_DIR, 'gnn_ppo_agent_adversarial.pth'
        )

    print("=" * 60)
    print("BiT-HyRL Adversarial Multi-Attack Training")
    print("=" * 60)
    print(f"Mode: adversarial")
    print(f"Aggregation: {args.aggregation}")
    print(f"Sample methods: {args.sample_methods}")
    print(f"Attack ratio: {args.attack_ratio}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"GAT layers: {args.num_layers}")
    print(f"Hidden channels: {args.hidden_channels}")
    print(f"Attention heads: {args.heads}")
    print(f"Model save path: {args.save_path}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    print("=" * 60)

    # 加载训练数据
    graphs, stats = load_training_graphs(
        data_dir=args.data_dir,
        max_graphs=args.max_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        use_bimodal=args.use_bimodal,
        hub_ratio=args.hub_ratio,
        seed=args.seed,
    )

    if not graphs:
        print("错误: 没有加载到有效的训练图")
        return

    print(f"\n开始训练，共 {len(graphs)} 个图...")
    t0 = time.time()

    # 组装传递给奖励函数的额外参数
    reward_kwargs = {'aggregation': args.aggregation}
    if args.sample_methods > 0:
        reward_kwargs['sample_methods'] = args.sample_methods

    result = train_gnn_ppo_optimized(
        graphs,
        epochs=args.epochs,
        save_path=args.save_path,
        mode='adversarial',
        lr=args.lr,
        embed_dim=args.embed_dim,
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        num_layers=args.num_layers,
        k_ratio=args.k_ratio,
        collect_per_epoch=args.collect_per_epoch,
        verbose=True,
        resume_from=args.resume,
        **reward_kwargs,
    )

    elapsed = time.time() - t0

    if result:
        print("\n" + "=" * 60)
        print("对抗式训练完成")
        print("=" * 60)
        print(f"训练耗时: {elapsed / 60:.2f} 分钟")
        print(f"最佳奖励: {result['best_reward']:.4f}")
        print(f"最终奖励: {result['final_reward']:.4f}")
        print(f"模型保存: {result['model_path']}")
        print("=" * 60)

        if result.get('history'):
            save_training_history(result['history'], args.save_path)
    else:
        print("训练失败，请检查日志。")


if __name__ == '__main__':
    main()
