# -*- coding: utf-8 -*-
"""
测试：逐步增大 hub 节点数量，观察双峰单层网络的韧性值（R 值）变化。

在固定 N、M 下，对 num_hubs = 1, 2, ..., max_hubs 分别构造双峰网络，
进行度攻击仿真，计算 R 值（GCC 曲线下面积），并绘制 num_hubs vs R 曲线。

用法:
    python scripts/test_bimodal_hubs_robustness.py
    python scripts/test_bimodal_hubs_robustness.py --dataset Colt
    python scripts/test_bimodal_hubs_robustness.py -n 153 -m 177 --max-hubs 30
    python scripts/test_bimodal_hubs_robustness.py --metric random    # 随机攻击下的 R 值
    python scripts/test_bimodal_hubs_robustness.py --metric degree    # 度攻击下的 R 值（默认）
    python scripts/test_bimodal_hubs_robustness.py --metric combined  # 0.5*R_random + R_degree
"""

import os
import sys
import argparse
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.topology.reconstruction import create_bimodal_with_num_hubs
from network_metrics import compute as _nm_compute


def degree_attack_sequence(G):
    """按度数降序返回攻击序列（节点列表）"""
    return sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)


def random_attack_sequence(G, seed=None):
    """随机打乱返回攻击序列（节点列表）"""
    nodes = list(G.nodes())
    rng = np.random.RandomState(seed)
    rng.shuffle(nodes)
    return nodes


def simulate_degree_attack_gcc_curve(G, attack_sequence=None):
    """
    度攻击仿真：无控制器，仅记录最大连通分量占比随移除比例的变化。
    用于单层网络韧性（不涉及控制器覆盖）。
    
    Returns:
        x_curve: 已移除节点比例 [0, 1]
        y_curve: 最大连通分量节点数 / 初始节点数
    """
    n0 = G.number_of_nodes()
    if n0 == 0:
        return [0.0], [0.0]
    if attack_sequence is None:
        attack_sequence = degree_attack_sequence(G)
    
    x_curve = [0.0]
    y_curve = [1.0]
    G_sim = G.copy()
    
    for step, node in enumerate(attack_sequence):
        if not G_sim.has_node(node):
            continue
        G_sim.remove_node(node)
        if G_sim.number_of_nodes() == 0:
            x_curve.append((step + 1) / n0)
            y_curve.append(0.0)
            break
        comps = list(nx.connected_components(G_sim))
        largest_cc_size = max(len(c) for c in comps) if comps else 0
        x_curve.append((step + 1) / n0)
        y_curve.append(largest_cc_size / n0)
    
    if x_curve[-1] < 1.0:
        x_curve.append(1.0)
        y_curve.append(y_curve[-1] if y_curve else 0.0)
    return x_curve, y_curve


def compute_r_value_with_attack(G, attack_sequence, num_points=101):
    """计算图在给定攻击序列下的韧性 R 值（GCC 曲线下面积）"""
    x_curve, y_curve = simulate_degree_attack_gcc_curve(G, attack_sequence=attack_sequence)
    return _nm_compute('r_value_interpolated', x_curve, y_curve, num_points=num_points)


def compute_r_value_by_metric(G, metric="degree", seed=None, num_points=101):
    """
    根据指定指标计算韧性 R 值。

    Args:
        G: 图
        metric: 评估指标
            - "random": 随机攻击下的 R 值
            - "degree": 度攻击下的 R 值
            - "combined": 0.5 * R_random + R_degree
        seed: 随机种子（用于 random 和 combined）
        num_points: 插值点数

    Returns:
        float: R 值
    """
    if metric == "degree":
        seq = degree_attack_sequence(G)
        return compute_r_value_with_attack(G, seq, num_points=num_points)
    elif metric == "random":
        seq = random_attack_sequence(G, seed=seed)
        return compute_r_value_with_attack(G, seq, num_points=num_points)
    elif metric == "combined":
        r_random = compute_r_value_by_metric(G, "random", seed=seed, num_points=num_points)
        r_degree = compute_r_value_by_metric(G, "degree", seed=None, num_points=num_points)
        return 0.5 * r_random + r_degree
    else:
        raise ValueError(f"不支持的 metric: {metric}，可选: random, degree, combined")


def test_bimodal_hubs_robustness(n, m, max_hubs=None, seed=42, metric="degree", verbose=True):
    """
    逐步增大 hub 数量，记录每个双峰网络的 R 值。
    
    Args:
        n: 节点数
        m: 边数
        max_hubs: 最大 hub 数（默认 min(50, n//4)）
        seed: 随机种子
        metric: 评估指标，可选 "random", "degree", "combined"（0.5*random+degree）
        verbose: 是否打印进度
        
    Returns:
        list of (num_hubs, r_value, success)
    """
    if max_hubs is None:
        max_hubs = min(50, max(1, n // 4))
    max_hubs = max(1, min(max_hubs, n - 1))
    
    results = []
    iterator = range(1, max_hubs + 1)
    if verbose:
        iterator = tqdm(iterator, desc=f"num_hubs (metric={metric})")
    
    for num_hubs in iterator:
        G = create_bimodal_with_num_hubs(n, m, num_hubs, seed=seed)
        if G is None:
            results.append((num_hubs, float('nan'), False))
            continue
        r_val = compute_r_value_by_metric(G, metric=metric, seed=seed, num_points=101)
        results.append((num_hubs, r_val, True))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="双峰网络 hub 数量与韧性 R 值关系测试")
    parser.add_argument("--dataset", "-d", type=str, default=None,
                        help="数据集名称（如 Colt），从 dataset/testdata 加载 N、M")
    parser.add_argument("-n", "--nodes", type=int, default=None, help="节点数（与 -m 一起使用）")
    parser.add_argument("-m", "--edges", type=int, default=None, help="边数")
    parser.add_argument("--max-hubs", type=int, default=None,
                        help="最大 hub 数量（默认 min(50, N//4)）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", "-M", type=str, default="degree",
                        choices=["random", "degree", "combined"],
                        help="评估指标: random(R随机), degree(R度攻击), combined(0.5*random+degree)")
    parser.add_argument("--output", "-o", type=str, default="results/bimodal_hubs_robustness",
                        help="结果与图表输出目录")
    parser.add_argument("--no-plot", action="store_true", help="不绘制曲线图")
    args = parser.parse_args()
    
    if args.dataset:
        from src.topology.generators import load_graph
        gml_path = os.path.join(PROJECT_ROOT, "dataset", "testdata", f"{args.dataset}.gml")
        if not os.path.exists(gml_path):
            gml_path = os.path.join(PROJECT_ROOT, "dataset", "all", "real", f"{args.dataset}.gml")
        if not os.path.exists(gml_path):
            print(f"未找到图: {args.dataset}.gml")
            return
        G0, _ = load_graph(gml_path, verbose=False)
        if G0 is None:
            print("加载图失败")
            return
        if not nx.is_connected(G0):
            G0 = G0.subgraph(max(nx.connected_components(G0), key=len)).copy()
        n, m = G0.number_of_nodes(), G0.number_of_edges()
        print(f"数据集: {args.dataset}, N={n}, M={m}")
    elif args.nodes is not None and args.edges is not None:
        n, m = args.nodes, args.edges
        print(f"使用指定 N={n}, M={m}")
    else:
        n, m = 153, 177
        print(f"使用默认 N={n}, M={m} (Colt)")
    
    results = test_bimodal_hubs_robustness(
        n, m, max_hubs=args.max_hubs, seed=args.seed, metric=args.metric, verbose=True
    )
    
    num_hubs_list = [r[0] for r in results]
    r_values = [r[1] for r in results]
    success_list = [r[2] for r in results]
    
    metric_label = {"random": "R(random)", "degree": "R(degree)", "combined": "R(0.5*random+degree)"}[args.metric]
    print("\n" + "=" * 60)
    print(f"num_hubs  vs  {metric_label}（韧性）")
    print("=" * 60)
    for (num_hubs, r_val, ok) in results:
        status = "" if ok else " (构造失败)"
        print(f"  num_hubs = {num_hubs:3d}  ->  R = {r_val:.4f}{status}")
    print("=" * 60)
    
    valid = [(h, r) for h, r, ok in results if ok and not np.isnan(r)]
    if not valid:
        print("无有效数据，跳过绘图")
        return
    
    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, f"hubs_vs_r_{args.metric}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("num_hubs,r_value,metric,success\n")
        for (h, r, ok) in results:
            f.write(f"{h},{r:.6f},{args.metric},{ok}\n")
    print(f"数据已保存: {csv_path}")
    
    if not args.no_plot and valid:
        hubs_plot = [x[0] for x in valid]
        r_plot = [x[1] for x in valid]
        plt.figure(figsize=(8, 5))
        plt.plot(hubs_plot, r_plot, "b-o", markersize=4)
        plt.xlabel("num_hubs (Hub count)")
        plt.ylabel(f"R ({metric_label}, GCC area)")
        plt.title(f"Bimodal single-layer: num_hubs vs {metric_label} (N={n}, M={m})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(args.output, f"bimodal_hubs_vs_R_{args.metric}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"曲线图已保存: {fig_path}")
    
    best_idx = np.nanargmax(r_values) if r_values else 0
    best_hubs = num_hubs_list[best_idx]
    best_r = r_values[best_idx]
    print(f"\n当前参数下 {metric_label} 最大: num_hubs = {best_hubs}, {metric_label} = {best_r:.4f}")


if __name__ == "__main__":
    main()
