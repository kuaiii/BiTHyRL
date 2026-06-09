# -*- coding: utf-8 -*-
"""
在 Colt 数据集上，根据双峰网络构造方法计算并输出度分布序列的特征与参数。

用法:
    python scripts/inspect_bimodal_colt.py
    python scripts/inspect_bimodal_colt.py --dataset Chinanet
"""

import os
import sys
import argparse
import numpy as np
import networkx as nx
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.topology.generators import load_graph
from src.topology.reconstruction import (
    create_bimodal_theoretical,
    create_bimodal_network_exact,
    _is_low_density,
    _bimodal_degree_sequence_sparse,
    SPARSE_AVG_DEGREE_THRESHOLD,
    SPARSE_DENSITY_THRESHOLD,
)


def compute_bimodal_params_theoretical(n, m, seed=42):
    """
    按双峰理论公式计算度序列参数（与 create_bimodal_theoretical 一致）。
    返回 (use_sparse, params_dict, degree_seq)。
    """
    avg_k = 2 * m / n
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0
    use_sparse = _is_low_density(n, m)

    params = {
        "n": n,
        "m": m,
        "avg_k": avg_k,
        "density": density,
        "use_sparse": use_sparse,
        "sparse_threshold_avg_k": SPARSE_AVG_DEGREE_THRESHOLD,
        "sparse_threshold_density": SPARSE_DENSITY_THRESHOLD,
    }

    if use_sparse:
        degree_seq = _bimodal_degree_sequence_sparse(n, m, seed=seed)
        degree_seq_sorted = sorted(degree_seq, reverse=True)
        params["num_hubs"] = 1
        params["k_max"] = degree_seq_sorted[0]
        params["k_base"] = degree_seq_sorted[-1] if degree_seq_sorted else 0
        params["n_low"] = n - 1
        params["remainder"] = sum(1 for d in degree_seq[1:] if d == params["k_base"] + 1)
        params["formula"] = "稀疏分支: k_max 受限于 2m-(n-1), k_base>=1"
    else:
        numerator = 2 * (avg_k**2) * ((avg_k - 1)**2)
        denominator = 2 * avg_k - 1
        A = (numerator / denominator) ** (1/3)
        k_max = int(round(A * (n ** (2/3))))
        k_max = min(k_max, n - 1)
        k_max = max(k_max, 2)
        n_high = 1
        n_low = n - n_high
        total_degree_needed = 2 * m
        degree_rem = total_degree_needed - k_max
        k_base = max(1, degree_rem // n_low)
        remainder = degree_rem % n_low
        degree_seq = [k_max] * n_high + [k_base + 1] * remainder + [k_base] * (n_low - remainder)
        params["A"] = A
        params["num_hubs"] = 1
        params["k_max"] = k_max
        params["k_base"] = k_base
        params["n_low"] = n_low
        params["remainder"] = remainder
        params["degree_rem"] = degree_rem
        params["formula"] = "k_max = A*N^(2/3), A=(2*(avg_k^2)*((avg_k-1)^2)/(2*avg_k-1))^(1/3)"
    return use_sparse, params, degree_seq


def degree_distribution_features(degree_seq):
    """度序列的统计特征"""
    if not degree_seq:
        return {}
    arr = np.array(degree_seq)
    cnt = Counter(degree_seq)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()) if len(arr) > 1 else 0.0,
        "sum": int(arr.sum()),
        "length": len(degree_seq),
        "unique_degrees": len(cnt),
        "degree_counts": dict(sorted(cnt.items(), reverse=True)),
        "is_graphical": nx.is_graphical(degree_seq),
    }


def main():
    parser = argparse.ArgumentParser(description="Colt 上双峰构造的度分布特征与参数")
    parser.add_argument("--dataset", type=str, default="Colt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hub-ratio", type=float, default=0.15)
    args = parser.parse_args()

    gml_path = os.path.join(PROJECT_ROOT, "dataset", "testdata", f"{args.dataset}.gml")
    if not os.path.exists(gml_path):
        gml_path = os.path.join(PROJECT_ROOT, "dataset", "all", "real", f"{args.dataset}.gml")
    if not os.path.exists(gml_path):
        print(f"未找到图文件: {args.dataset}.gml")
        return

    G, _ = load_graph(gml_path, verbose=False)
    if G is None:
        print("加载图失败")
        return
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    n = G.number_of_nodes()
    m = G.number_of_edges()
    avg_k_orig = 2 * m / n
    density_orig = (2 * m) / (n * (n - 1)) if n > 1 else 0

    print("=" * 70)
    print(f"数据集: {args.dataset}")
    print("=" * 70)
    print(f"  节点数 N     = {n}")
    print(f"  边数 M       = {m}")
    print(f"  平均度 avg_k = {avg_k_orig:.4f}")
    print(f"  密度        = {density_orig:.6f}")
    print()

    # ---------- 方法1: create_bimodal_theoretical 对应的度序列参数 ----------
    print("-" * 70)
    print("双峰构造方法: create_bimodal_theoretical(N, M)")
    print("-" * 70)
    use_sparse, params, degree_seq = compute_bimodal_params_theoretical(n, m, seed=args.seed)
    print(f"  是否走稀疏分支: {use_sparse} (avg_k < {SPARSE_AVG_DEGREE_THRESHOLD} 或 density < {SPARSE_DENSITY_THRESHOLD})")
    print(f"  公式说明: {params['formula']}")
    print(f"  参数:")
    print(f"    num_hubs (hub 数) = {params['num_hubs']}")
    print(f"    k_max (hub 度数)  = {params['k_max']}")
    print(f"    k_base (普通节点基础度数) = {params['k_base']}")
    print(f"    n_low (普通节点数) = {params['n_low']}")
    print(f"    remainder (度数为 k_base+1 的节点数) = {params.get('remainder', 'N/A')}")
    if "A" in params:
        print(f"    A (理论系数) = {params['A']:.4f}")
    print()
    print("  度序列 (降序):")
    sorted_seq = sorted(degree_seq, reverse=True)
    print(f"    {sorted_seq[:20]}{'...' if len(sorted_seq) > 20 else ''}")
    print(f"    长度 = {len(degree_seq)}, 和 = {sum(degree_seq)} (应为 2*M = {2*m})")
    feats = degree_distribution_features(degree_seq)
    print(f"  度分布特征: min={feats['min']}, max={feats['max']}, mean={feats['mean']:.4f}, std={feats['std']:.4f}")
    print(f"  可图化: {feats['is_graphical']}")
    print(f"  度数取值及出现次数: {feats['degree_counts']}")
    print()

    # ---------- 方法2: create_bimodal_network_exact 构造后的实际度分布 ----------
    print("-" * 70)
    print("双峰构造方法: create_bimodal_network_exact(G, hub_ratio)")
    print("-" * 70)
    G_bimodal = create_bimodal_network_exact(G, hub_ratio=args.hub_ratio, seed=args.seed)
    exact_degrees = [d for _, d in G_bimodal.degree()]
    exact_sorted = sorted(exact_degrees, reverse=True)
    print(f"  hub_ratio = {args.hub_ratio}")
    print(f"  构造后 节点数 = {G_bimodal.number_of_nodes()}, 边数 = {G_bimodal.number_of_edges()}, 连通 = {nx.is_connected(G_bimodal)}")
    print(f"  度序列 (降序):")
    print(f"    {exact_sorted[:20]}{'...' if len(exact_sorted) > 20 else ''}")
    feats_exact = degree_distribution_features(exact_degrees)
    print(f"  度分布特征: min={feats_exact['min']}, max={feats_exact['max']}, mean={feats_exact['mean']:.4f}, std={feats_exact['std']:.4f}")
    print(f"  度数取值及出现次数: {feats_exact['degree_counts']}")
    print()

    # ---------- 理论构造得到的图的实际度分布（用于对比） ----------
    print("-" * 70)
    print("双峰理论构造图 create_bimodal_theoretical(N, M) 的实际度分布")
    print("-" * 70)
    G_theory = create_bimodal_theoretical(n, m, seed=args.seed)
    theory_degrees = [d for _, d in G_theory.degree()]
    theory_sorted = sorted(theory_degrees, reverse=True)
    print(f"  构造后 节点数 = {G_theory.number_of_nodes()}, 边数 = {G_theory.number_of_edges()}, 连通 = {nx.is_connected(G_theory)}")
    print(f"  度序列 (降序):")
    print(f"    {theory_sorted[:20]}{'...' if len(theory_sorted) > 20 else ''}")
    feats_theory = degree_distribution_features(theory_degrees)
    print(f"  度分布特征: min={feats_theory['min']}, max={feats_theory['max']}, mean={feats_theory['mean']:.4f}, std={feats_theory['std']:.4f}")
    print(f"  度数取值及出现次数: {feats_theory['degree_counts']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
