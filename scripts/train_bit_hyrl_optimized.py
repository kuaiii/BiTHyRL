#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BiT-HyRL 优化研究训练脚本 (Optimized Training)
================================================

研究如何改进 BiT-HyRL 算法本身，不只是 scale-up。

核心改进：
  1. 课程式攻击采样 (Curriculum)：早期简单攻击，后期逐步加入复杂攻击
  2. 自适应奖励聚合 (Adaptive Aggregation)：早期 mean 鼓励泛化，后期 min 强化鲁棒性
  3. 完整攻击池：使用全部可用 network_dismantling 方法
  4. 增大 collect_per_epoch 以收集更多轨迹

数据集:
  - dataset/all/syn/   : 扩展后的训练数据集 (~12000 图)
  - dataset/testdata/  : 最终验证集

用法:
  python scripts/train_bit_hyrl_optimized.py --epochs 200
  python scripts/train_bit_hyrl_optimized.py --resume src/train/v1/checkpoints/gnn_ppo_agent_optimized.pth
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 必须在导入 tf 前设置：0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

import argparse
import csv
import random
import time
import datetime
import logging
import numpy as np
import networkx as nx
from tqdm import tqdm

# ===== 抑制第三方库的冗长日志 =====
# gensim / node2vec
logging.getLogger('gensim').setLevel(logging.WARNING)
logging.getLogger('gensim.models').setLevel(logging.WARNING)
logging.getLogger('gensim.models.word2vec').setLevel(logging.WARNING)
logging.getLogger('gensim.utils').setLevel(logging.WARNING)
# tensorflow
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# node2vec 包本身
logging.getLogger('node2vec').setLevel(logging.WARNING)

from src.topology.generators import load_graph
from src.topology.reconstruction import create_bimodal_network_exact
from src.bit_hyrl import config
from src.bit_hyrl.training import train_gnn_ppo_optimized
from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


class TeeOutput:
    """同时将输出写入文件和控制台的封装器"""
    def __init__(self, filepath, stream):
        self.file = open(filepath, 'a', encoding='utf-8')
        self.stream = stream
        
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stream.write(data)
        self.stream.flush()
        
    def flush(self):
        self.file.flush()
        self.stream.flush()
        
    def close(self):
        self.file.close()
    
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        self.close()


def init_training_logger():
    """初始化训练日志：同时输出到控制台和 logs/ 文件夹"""
    import logging
    log_dir = os.path.join(ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'train_optimized_{timestamp}.log'
    log_path = os.path.join(log_dir, log_filename)
    
    # 配置 logging（INFO 级别）
    setup_logger(
        log_dir=log_dir,
        log_filename=log_filename,
        level=logging.INFO,
        console_level=logging.INFO
    )
    
    # 同时重定向 stdout 和 stderr 到日志文件
    tee_out = TeeOutput(log_path, sys.stdout)
    tee_err = TeeOutput(log_path, sys.stderr)
    sys.stdout = tee_out
    sys.stderr = tee_err
    
    logger.info(f"训练日志文件: {log_path}")
    return log_path


def _load_single_graph(args):
    """单图加载（用于多进程）"""
    filepath, name, min_nodes, max_nodes, use_bimodal, hub_ratio = args
    try:
        G_original, _ = load_graph(filepath, verbose=False)
        if G_original is None or G_original.number_of_nodes() < min_nodes:
            return None, 'skipped'
        if G_original.number_of_nodes() > max_nodes:
            return None, 'skipped'
        if not nx.is_connected(G_original):
            return None, 'skipped'

        original_nodes = G_original.number_of_nodes()
        original_edges = G_original.number_of_edges()

        if use_bimodal:
            hub_num = max(1, int(G_original.number_of_nodes() * hub_ratio))
            G = create_bimodal_network_exact(
                G_original, hub_num=hub_num, seed=0
            )
            if (
                G.number_of_nodes() != original_nodes
                or G.number_of_edges() != original_edges
                or not nx.is_connected(G)
            ):
                G = G_original
        else:
            G = G_original

        return {
            'graph': G,
            'name': name,
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
        }, 'ok'
    except Exception as e:
        return None, f'failed:{e}'


def load_training_graphs(
    data_dir=None,
    max_graphs=None,
    min_nodes=20,
    max_nodes=1000,
    use_bimodal=True,
    hub_ratio=0.15,
    seed=42,
    num_workers=8,
):
    """从 dataset/all/syn/ 加载训练图列表（支持多进程并行加载）。"""
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

    # 构建参数列表
    args_list = [
        (os.path.join(data_dir, f), os.path.splitext(f)[0], min_nodes, max_nodes, use_bimodal, hub_ratio)
        for f in gml_files
    ]

    graphs = []
    stats = []
    skipped = 0
    failed = 0

    # 多进程并行加载
    from multiprocessing import Pool
    with Pool(processes=num_workers) as pool:
        for result, status in tqdm(
            pool.imap(_load_single_graph, args_list),
            total=len(args_list),
            desc='Loading graphs',
            ncols=80
        ):
            if status == 'ok':
                graphs.append(result['graph'])
                stats.append({
                    'name': result['name'],
                    'nodes': result['nodes'],
                    'edges': result['edges'],
                })
            elif status == 'skipped':
                skipped += 1
            elif status.startswith('failed'):
                failed += 1
                logger.warning(f"加载失败: {status}")

    print(f"加载完成: {len(graphs)} 个有效图")
    if skipped:
        print(f"  跳过: {skipped} 个")
    if failed:
        print(f"  失败: {failed} 个")
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
        description='BiT-HyRL Optimized Training (Curriculum + Adaptive Aggregation)'
    )
    parser.add_argument('--data-dir', type=str, default=None,
                        help='训练数据目录 (默认: dataset/all/syn)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数 (默认: 200)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='学习率 (默认: 3e-4)')
    parser.add_argument('--k-ratio', type=float, default=0.1,
                        help='控制器比例 (默认: 0.1)')
    # ===== Scale-up 模型参数 (与 scaled 一致) =====
    parser.add_argument('--hidden-channels', type=int, default=256,
                        help='GAT 隐藏层维度 (默认: 256)')
    parser.add_argument('--heads', type=int, default=12,
                        help='GAT 注意力头数 (默认: 12)')
    parser.add_argument('--num-layers', type=int, default=6,
                        help='GAT 层数 (默认: 6)')
    parser.add_argument('--embed-dim', type=int, default=256,
                        help='Node2Vec 嵌入维度 (默认: 256)')
    parser.add_argument('--dre-dim', type=int, default=256,
                        help='DRE 度排名嵌入维度 (默认: 256)')
    parser.add_argument('--node2vec-walk-length', type=int, default=20,
                        help='Node2Vec 游走长度 (默认: 20，不建议减小)')
    parser.add_argument('--node2vec-num-walks', type=int, default=100,
                        help='Node2Vec 每节点游走次数 (默认: 100，不建议减小)')
    parser.add_argument('--no-embedding-cache', action='store_true',
                        help='禁用嵌入缓存（默认启用缓存加速）')
    # ==============================================
    parser.add_argument('--collect-per-epoch', type=int, default=60,
                        help='每轮收集轨迹数 (默认: 60，可酌情减小以加速)')
    parser.add_argument('--max-graphs', type=int, default=None,
                        help='最大使用图数量 (None 表示全部)')
    parser.add_argument('--min-nodes', type=int, default=20,
                        help='最小节点数过滤 (默认: 20)')
    parser.add_argument('--max-nodes', type=int, default=1000,
                        help='最大节点数过滤 (默认: 1000)')
    parser.add_argument('--use-bimodal', action='store_true', default=True,
                        help='启用双峰拓扑重构 (默认: True)')
    parser.add_argument('--hub-ratio', type=float, default=0.15,
                        help='双峰拓扑 hub 比例 (默认: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='模型保存路径 (默认: src/train/v1/checkpoints/gnn_ppo_agent_optimized.pth)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从已有 checkpoint 恢复训练')
    parser.add_argument('--attack-ratio', type=float, default=0.15,
                        help='奖励函数攻击比例 (默认: 0.15)')
    parser.add_argument('--sample-methods', type=int, default=0,
                        help='每次奖励计算随机采样的攻击方法数 (默认: 0, 表示使用全部)')
    parser.add_argument('--attack-methods', type=str, default=None,
                        help='指定攻击方法池，逗号分隔 (默认使用全部可用方法，如: degree,random)')
    # ===== 优化特有参数 =====
    parser.add_argument('--no-curriculum', action='store_true',
                        help='禁用课程式攻击采样')
    parser.add_argument('--no-adaptive-aggregation', action='store_true',
                        help='禁用自适应聚合策略')
    parser.add_argument('--aggregation', type=str, default='min',
                        choices=['min', 'mean', 'max'],
                        help='基础聚合方式 (默认: min)')
    parser.add_argument('--show-node-stats', action='store_true',
                        help='加载完成后显示节点数量统计信息')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 初始化日志（同时输出到控制台和 logs/ 文件夹）
    log_path = init_training_logger()
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.save_path is None:
        args.save_path = os.path.join(
            config.MODEL_DIR, 'gnn_ppo_agent_optimized.pth'
        )

    print("=" * 70)
    print("BiT-HyRL Optimized Training")
    print("=" * 70)
    print(f"Mode: adversarial (optimized)")
    print(f"Curriculum sampling: {not args.no_curriculum}")
    print(f"Adaptive aggregation: {not args.no_adaptive_aggregation}")
    print(f"Base aggregation: {args.aggregation}")
    print(f"Sample methods: {args.sample_methods} (0 = all available)")
    print(f"Attack ratio: {args.attack_ratio}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"GAT layers: {args.num_layers}")
    print(f"Hidden channels: {args.hidden_channels}")
    print(f"Attention heads: {args.heads}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"DRE dim: {args.dre_dim}")
    print(f"Total input dim: {args.embed_dim + args.dre_dim}")
    print(f"Model save path: {args.save_path}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    print("=" * 70)

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

    if args.show_node_stats and stats:
        print("\n" + "=" * 70)
        print("节点数量统计")
        print("=" * 70)
        node_counts = [s['nodes'] for s in stats]
        print(f"  总图数   : {len(node_counts)}")
        print(f"  最小节点数: {min(node_counts)}")
        print(f"  最大节点数: {max(node_counts)}")
        print(f"  平均节点数: {sum(node_counts) / len(node_counts):.1f}")

        # 分布区间
        bins = [(0, 50), (50, 100), (100, 200), (200, 500), (500, float('inf'))]
        labels = ['0-50', '50-100', '100-200', '200-500', '500+']
        dist = {label: 0 for label in labels}
        for n in node_counts:
            for (low, high), label in zip(bins, labels):
                if low <= n < high:
                    dist[label] += 1
                    break
        print("  分布:")
        for label in labels:
            print(f"    {label:8s}: {dist[label]}")
        print("=" * 70)

    print(f"\n开始训练，共 {len(graphs)} 个图...")
    t0 = time.time()

    # 组装传递给奖励函数的额外参数
    reward_kwargs = {
        'aggregation': args.aggregation,
        'use_curriculum': not args.no_curriculum,
        'use_adaptive_aggregation': not args.no_adaptive_aggregation,
    }
    if args.sample_methods > 0:
        reward_kwargs['sample_methods'] = args.sample_methods
    if args.attack_methods:
        reward_kwargs['attack_methods'] = [m.strip() for m in args.attack_methods.split(',')]

    result = train_gnn_ppo_optimized(
        graphs,
        epochs=args.epochs,
        save_path=args.save_path,
        mode='adversarial',
        lr=args.lr,
        embed_dim=args.embed_dim,
        dre_dim=args.dre_dim,
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        num_layers=args.num_layers,
        k_ratio=args.k_ratio,
        collect_per_epoch=args.collect_per_epoch,
        walk_length=args.node2vec_walk_length,
        num_walks=args.node2vec_num_walks,
        use_embedding_cache=not args.no_embedding_cache,
        verbose=True,
        resume_from=args.resume,
        **reward_kwargs,
    )

    elapsed = time.time() - t0

    if result:
        print("\n" + "=" * 70)
        print("Optimized 训练完成")
        print("=" * 70)
        print(f"训练耗时: {elapsed / 60:.2f} 分钟")
        print(f"最佳奖励: {result['best_reward']:.4f}")
        print(f"最终奖励: {result['final_reward']:.4f}")
        print(f"模型保存: {result['model_path']}")
        print("=" * 70)

        if result.get('history'):
            save_training_history(result['history'], args.save_path)
    else:
        print("训练失败，请检查日志。")


if __name__ == '__main__':
    main()
