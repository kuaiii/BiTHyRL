#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络构造算法横向对比基准测试
====================================

在相同数据集和攻击方式下，横向对比所有 network_construction 注册算法的性能。

对比维度：
  - 拓扑鲁棒性：GCC AUC、Collapse Point
  - 控制效果：CSA、CCE、WCP（统一使用 degree 最高 k 节点部署控制器）

额外对比 BiT-HyRL GNN 控制器部署（adversarial / baseline）。

用法:
  python scripts/cross_algorithm_benchmark.py --output results/cross_benchmark.csv
  python scripts/cross_algorithm_benchmark.py \
      --datasets BA-200 BA-500 Chinanet Colt \
      --attacks degree betweenness CI_L1 CoreHD GND \
      --output results/cross_benchmark.csv
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import csv
import time
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict

from src.topology.generators import load_graph
from src.utils.logger import get_logger
import network_construction as nc
import network_dismantling as nd
import network_metrics as nm

logger = get_logger(__name__)

# 默认对比算法（排除 BiT-HyRL，因为它在 construction 中是拓扑重构，
# 但我们会额外加入 BiT-HyRL GNN 部署对比）
DEFAULT_ALGORITHMS = [
    'baseline', 'UNITY', 'FRED_ABL', 'BiT-HyRL',
    # 以下算法构造可能较慢或在部分图上卡住，可通过 --algorithms 参数显式添加
    # 'GA', 'Onion', 'ROMEN', 'QDLM', 'Q_Robust', 'SmartTRO', 'TEAM',
]

# BiT-HyRL GNN 模型路径
BIT_HYRL_BASELINE_MODEL = os.path.join(ROOT, 'src', 'train', 'v1', 'checkpoints', 'deep_gat_gcc_L5_H8_D128.pth')
BIT_HYRL_ADV_MODEL = os.path.join(ROOT, 'src', 'train', 'v1', 'checkpoints', 'gnn_ppo_agent_adversarial.pth')

DEFAULT_ATTACK_METHODS = [
    'degree', 'betweenness', 'CI_L1', 'CoreHD', 'GND', 'eigenvector', 'random',
]


def load_test_graphs(data_dir=None, min_nodes=10):
    """从 dataset/testdata/ 加载测试图。"""
    if data_dir is None:
        data_dir = os.path.join(ROOT, 'dataset', 'testdata')
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"测试数据目录不存在: {data_dir}")

    gml_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.gml')])
    graphs, names = [], []
    for filename in gml_files:
        filepath = os.path.join(data_dir, filename)
        name = os.path.splitext(filename)[0]
        try:
            G, _ = load_graph(filepath, verbose=False)
            if G is None or G.number_of_nodes() < min_nodes:
                continue
            if not nx.is_connected(G):
                continue
            graphs.append(G)
            names.append(name)
        except Exception as e:
            logger.warning(f"加载 {name} 失败: {e}")
    return graphs, names


def select_controllers_degree(G, k):
    """按 degree 选择前 k 个节点作为控制器。"""
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in degrees[:k]]


def evaluate_topology(G, centers, attack_method, attack_ratio=0.3):
    """
    评估给定拓扑和控制器部署在指定攻击下的指标。

    Returns:
        dict: {'gcc_auc': float, 'collapse_ratio': float,
               'csa': float, 'cce': float, 'wcp': float}
        或 None（失败时）
    """
    total_nodes = G.number_of_nodes()
    if total_nodes == 0:
        return None

    try:
        sequence = nd.dismantle(G, method=attack_method)
    except Exception as e:
        logger.warning(f"攻击方法 {attack_method} 失败: {e}")
        return None

    num_remove = max(1, int(total_nodes * attack_ratio))
    attack_sequence = sequence[:num_remove]

    try:
        # GCC 轨迹
        gcc_traj = nm.compute('attack_trajectory', G, attack_sequence=attack_sequence)
        x_curve = [p[1] for p in gcc_traj]
        y_curve = [p[2] / total_nodes for p in gcc_traj]
        gcc_auc = nm.compute('r_value_interpolated', x_curve, y_curve, num_points=101)

        # 崩溃点
        collapse_ratio = None
        for p in gcc_traj:
            if p[2] <= total_nodes * 0.2:
                collapse_ratio = p[1]
                break
        if collapse_ratio is None and gcc_traj:
            collapse_ratio = gcc_traj[-1][1]

        # 功能指标（攻击后网络）
        G_attacked = G.copy()
        alive_centers = set(centers)
        for node in attack_sequence:
            if node in G_attacked:
                G_attacked.remove_node(node)
            if node in alive_centers:
                alive_centers.remove(node)

        if G_attacked.number_of_nodes() == 0 or not alive_centers:
            csa = cce = wcp = 0.0
        else:
            csa = nm.compute('csa', G_attacked, centers=list(alive_centers))
            cce = nm.compute('cce', G_attacked, centers=list(alive_centers))
            wcp = nm.compute('wcp', G_attacked, centers=list(alive_centers))

        return {
            'gcc_auc': gcc_auc,
            'collapse_ratio': collapse_ratio,
            'csa': csa,
            'cce': cce,
            'wcp': wcp,
        }
    except Exception as e:
        logger.warning(f"评估失败: {e}")
        return None


def construct_topology(G, algorithm, seed=42):
    """使用指定算法构造拓扑，返回 (G_constructed, success)。"""
    try:
        if algorithm == 'GA':
            G2 = nc.construct(G, algorithm='GA')
        elif algorithm == 'Onion':
            G2 = nc.construct(G, algorithm='Onion', max_iter=300)
        elif algorithm == 'ROMEN':
            G2 = nc.construct(G, algorithm='ROMEN', seed=seed)
        elif algorithm == 'UNITY':
            G2 = nc.construct(G, algorithm='UNITY')
        elif algorithm == 'FRED_ABL':
            G2 = nc.construct(G, algorithm='FRED_ABL', seed=seed, iterations=15,
                              initial_samples=3, verbose=False)
        elif algorithm == 'QDLM':
            G2 = nc.construct(G, algorithm='QDLM', seed=seed, pop_size=20,
                              max_iterations=25, verbose=False)
        elif algorithm == 'Q_Robust':
            G2 = nc.construct(G, algorithm='Q_Robust')
        elif algorithm == 'SmartTRO':
            G2 = nc.construct(G, algorithm='SmartTRO')
        elif algorithm == 'TEAM':
            G2 = nc.construct(G, algorithm='TEAM', seed=seed, verbose=False)
        elif algorithm == 'BiT-HyRL':
            G2 = nc.construct(G, algorithm='BiT-HyRL', num_hubs=max(1, G.number_of_nodes() // 10))
        else:
            # baseline
            G2 = nc.construct(G, algorithm='baseline')
        return G2, True
    except Exception as e:
        logger.warning(f"构造算法 {algorithm} 失败: {e}")
        return None, False


def run_benchmark(datasets=None, algorithms=None, attack_methods=None,
                  attack_ratio=0.3, k_ratio=0.1, data_dir=None,
                  include_bit_hyrl_gnn=True):
    """
    运行横向对比基准测试。

    Returns:
        list[dict]: 每条记录包含 algorithm, dataset, attack_method, gcc_auc, collapse_ratio, csa, cce, wcp
    """
    if algorithms is None:
        algorithms = DEFAULT_ALGORITHMS.copy()
    if attack_methods is None:
        attack_methods = DEFAULT_ATTACK_METHODS.copy()

    graphs, names = load_test_graphs(data_dir)
    if datasets:
        # 过滤指定数据集
        filtered = [(g, n) for g, n in zip(graphs, names) if n in datasets]
        if not filtered:
            raise ValueError(f"指定的数据集未找到: {datasets}")
        graphs, names = zip(*filtered) if filtered else ([], [])
        graphs, names = list(graphs), list(names)

    # 过滤可用攻击方法
    available_methods = [m for m in attack_methods if m in nd.list_methods()]
    if not available_methods:
        raise RuntimeError("没有可用的攻击方法")
    print(f"可用攻击方法 ({len(available_methods)}): {', '.join(available_methods)}")

    results = []

    # 额外加载 BiT-HyRL GNN 模型（如果需要）
    bit_hyrl_models = {}
    if include_bit_hyrl_gnn:
        try:
            from src.bit_hyrl.selection import load_gnn_model, gnn_predict
            for label, path in [('BiT-HyRL-GNN-Baseline', BIT_HYRL_BASELINE_MODEL),
                                ('BiT-HyRL-GNN-Adv', BIT_HYRL_ADV_MODEL)]:
                if os.path.exists(path):
                    model, ckpt, mtype = load_gnn_model(path)
                    in_ch = ckpt.get('in_channels', 128)
                    bit_hyrl_models[label] = (model, in_ch)
                    print(f"已加载 {label}: {path}")
                else:
                    print(f"模型不存在，跳过 {label}: {path}")
        except Exception as e:
            logger.warning(f"加载 BiT-HyRL GNN 模型失败: {e}")

    total_evals = len(graphs) * (len(algorithms) + len(bit_hyrl_models)) * len(available_methods)
    pbar = tqdm(total=total_evals, desc='Cross-algorithm benchmark', ncols=100)

    for G, name in zip(graphs, names):
        n_nodes = G.number_of_nodes()
        k = max(1, int(n_nodes * k_ratio))
        print(f"\n数据集: {name} ({n_nodes} nodes)")

        # 1) 拓扑重构算法
        for algo in algorithms:
            t0 = time.time()
            G2, ok = construct_topology(G, algo, seed=42)
            construct_time = time.time() - t0
            if not ok or G2 is None:
                logger.warning(f"  {algo}: 构造失败，跳过")
                pbar.update(len(available_methods))
                continue

            # 确保图连通
            if not nx.is_connected(G2):
                largest_cc = max(nx.connected_components(G2), key=len)
                G2 = G2.subgraph(largest_cc).copy()

            centers = select_controllers_degree(G2, k)
            for method in available_methods:
                metrics = evaluate_topology(G2, centers, method, attack_ratio=attack_ratio)
                if metrics:
                    results.append({
                        'algorithm': algo,
                        'dataset': name,
                        'nodes': n_nodes,
                        'controllers': len(centers),
                        'attack_method': method,
                        'construct_time': round(construct_time, 2),
                        **metrics,
                    })
                pbar.update(1)

        # 2) BiT-HyRL GNN 控制器部署（在原图上，不改拓扑）
        for label, (model, in_ch) in bit_hyrl_models.items():
            try:
                centers = gnn_predict(G, k, model=model, deterministic=True, embed_dim=in_ch)
            except Exception as e:
                logger.warning(f"  {label}: GNN 预测失败 ({e})")
                pbar.update(len(available_methods))
                continue
            for method in available_methods:
                metrics = evaluate_topology(G, centers, method, attack_ratio=attack_ratio)
                if metrics:
                    results.append({
                        'algorithm': label,
                        'dataset': name,
                        'nodes': n_nodes,
                        'controllers': len(centers),
                        'attack_method': method,
                        'construct_time': 0.0,
                        **metrics,
                    })
                pbar.update(1)

    pbar.close()
    return results


def save_results(results, output_path):
    """保存结果到 CSV。"""
    if not results:
        print("无结果可保存")
        return
    keys = ['algorithm', 'dataset', 'nodes', 'controllers', 'attack_method',
            'gcc_auc', 'collapse_ratio', 'csa', 'cce', 'wcp', 'construct_time']
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in keys})
    print(f"\n详细结果已保存: {output_path}")


def print_summary(results):
    """按算法和攻击方法打印汇总表格。"""
    if not results:
        print("无结果")
        return

    # 按 (algorithm, attack_method) 聚合
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r['algorithm'], r['attack_method'])
        for metric in ['gcc_auc', 'collapse_ratio', 'csa', 'cce', 'wcp']:
            agg[key][metric].append(r[metric])

    # 按算法分组
    algo_groups = defaultdict(list)
    for (algo, method), metrics in agg.items():
        algo_groups[algo].append((method, metrics))

    print("\n" + "=" * 110)
    print("横向对比汇总：按算法 × 攻击方法（跨数据集平均）")
    print("=" * 110)
    header = f"{'Algorithm':<22} {'Attack':<14} {'GCC AUC':>9} {'Collapse':>9} {'CSA':>9} {'CCE':>9} {'WCP':>9} {'Count':>6}"
    print(header)
    print("-" * 110)

    for algo in sorted(algo_groups.keys()):
        for method, metrics in sorted(algo_groups[algo]):
            count = len(metrics['gcc_auc'])
            vals = {m: np.mean(metrics[m]) for m in ['gcc_auc', 'collapse_ratio', 'csa', 'cce', 'wcp']}
            print(f"{algo:<22} {method:<14} {vals['gcc_auc']:>9.4f} {vals['collapse_ratio']:>9.4f} "
                  f"{vals['csa']:>9.4f} {vals['cce']:>9.4f} {vals['wcp']:>9.4f} {count:>6}")
    print("=" * 110)

    # 按算法打印平均（跨所有攻击方法）
    print("\n" + "=" * 90)
    print("算法综合排名（跨攻击方法平均）")
    print("=" * 90)
    print(f"{'Algorithm':<22} {'GCC AUC':>9} {'Collapse':>9} {'CSA':>9} {'CCE':>9} {'WCP':>9}")
    print("-" * 90)
    algo_avg = {}
    for algo in sorted(algo_groups.keys()):
        all_vals = defaultdict(list)
        for _, metrics in algo_groups[algo]:
            for m in ['gcc_auc', 'collapse_ratio', 'csa', 'cce', 'wcp']:
                all_vals[m].extend(metrics[m])
        avg = {m: np.mean(all_vals[m]) for m in ['gcc_auc', 'collapse_ratio', 'csa', 'cce', 'wcp']}
        algo_avg[algo] = avg
        print(f"{algo:<22} {avg['gcc_auc']:>9.4f} {avg['collapse_ratio']:>9.4f} "
              f"{avg['csa']:>9.4f} {avg['cce']:>9.4f} {avg['wcp']:>9.4f}")
    print("=" * 90)

    # 打印每项指标的最佳算法
    print("\n" + "=" * 60)
    print("各指标最佳算法")
    print("=" * 60)
    for metric, direction in [('gcc_auc', 'max'), ('collapse_ratio', 'max'),
                               ('csa', 'max'), ('cce', 'max'), ('wcp', 'max')]:
        best_algo = max(algo_avg, key=lambda a: algo_avg[a][metric]) if direction == 'max' else \
                    min(algo_avg, key=lambda a: algo_avg[a][metric])
        best_val = algo_avg[best_algo][metric]
        print(f"  {metric:<18} -> {best_algo:<22} ({best_val:.4f})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='网络构造算法横向对比基准测试')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='指定数据集名称（默认全部）')
    parser.add_argument('--algorithms', nargs='+', default=None,
                        help=f'指定拓扑重构算法（默认: {DEFAULT_ALGORITHMS}）')
    parser.add_argument('--attacks', nargs='+', default=None,
                        help=f'指定攻击方法（默认: {DEFAULT_ATTACK_METHODS}）')
    parser.add_argument('--attack-ratio', type=float, default=0.3,
                        help='攻击比例（默认 0.3）')
    parser.add_argument('--k-ratio', type=float, default=0.1,
                        help='控制器比例（默认 0.1）')
    parser.add_argument('--data-dir', default=None,
                        help='数据集目录（默认 dataset/testdata）')
    parser.add_argument('--output', default='results/cross_benchmark.csv',
                        help='输出 CSV 路径')
    parser.add_argument('--no-gnn', action='store_true',
                        help='跳过 BiT-HyRL GNN 控制器部署对比')
    args = parser.parse_args()

    results = run_benchmark(
        datasets=args.datasets,
        algorithms=args.algorithms,
        attack_methods=args.attacks,
        attack_ratio=args.attack_ratio,
        k_ratio=args.k_ratio,
        data_dir=args.data_dir,
        include_bit_hyrl_gnn=not args.no_gnn,
    )

    save_results(results, args.output)
    print_summary(results)


if __name__ == '__main__':
    main()
