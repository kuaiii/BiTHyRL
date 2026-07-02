#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比各重构拓扑在 random / degree 攻击下的：
- AUC（GCC 比例随移除比例的曲线下面积）
- remove_num（GCC 首次 < 1% 时的移除节点数）
- 加权综合指标：0.5 * random + 0.5 * degree

同时验证双峰（Bimodal）拓扑是否呈现两个度峰值。
"""
import os
import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings('ignore')

from src.topology.generators import load_graph
from src.topology.reconstruction import create_bimodal_theoretical, _ensure_connectivity, _adjust_edge_count
from src.unified_ppo.topology_policy import TopologyPolicy
from src.unified_ppo.inference import unified_solve
import network_construction as nc


def to_int_graph(G):
    return nx.convert_node_labels_to_integers(G, label_attribute='orig_label')


def simulate_attack_gcc(G, attack_sequence):
    """返回 GCC 比例随移除节点数变化的数组，长度 N+1（包含 0 移除）。"""
    G2 = G.copy()
    n = G2.number_of_nodes()
    if n == 0:
        return np.array([0.0])
    decay = [1.0]
    for node in attack_sequence:
        if node in G2:
            G2.remove_node(node)
        if G2.number_of_nodes() == 0:
            decay.append(0.0)
            break
        comps = list(nx.connected_components(G2))
        gcc_size = max(len(c) for c in comps) if comps else 0
        decay.append(gcc_size / n)
    return np.array(decay)


def compute_auc_and_remove(gcc_ratios):
    """
    AUC：GCC 比例对移除比例的曲线下面积（0..1）。
    remove_num：GCC 首次 <= 1% 时的移除节点数；若未发生则为 len-1。
    remove_frac：remove_num / (len(gcc_ratios)-1)。
    """
    n_steps = len(gcc_ratios) - 1
    x = np.arange(len(gcc_ratios)) / n_steps if n_steps > 0 else np.array([0.0])
    auc = np.trapz(gcc_ratios, x)
    # 找到第一个 GCC <= 1% 的位置
    idx = np.argmax(gcc_ratios <= 0.01)
    if gcc_ratios[idx] > 0.01:
        remove_num = n_steps
    else:
        remove_num = idx
    remove_frac = remove_num / n_steps if n_steps > 0 else 1.0
    return auc, remove_num, remove_frac


def degree_attack_sequence(G):
    return [n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)]


def random_attack_sequence(G, seed=42):
    rng = np.random.RandomState(seed)
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    return nodes


def create_bimodal_two_peaks(n, m, hub_ratio=0.1, seed=42):
    """
    构造严格双峰度分布：hub 节点度为 k_hub，leaf 节点度为 k_leaf。
    hub 数量 = max(1, int(round(n * hub_ratio)))，不一定是 1 个。
    通过从高 hub 度向低 hub 度搜索，找到满足总度数约束且可图化的序列。
    """
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)
    if n < 2 or m < n - 1:
        return nx.path_graph(n) if n >= 2 else nx.Graph()

    num_hubs = max(1, int(round(n * hub_ratio)))
    num_leaves = n - num_hubs
    total_deg = 2 * m
    avg_k = total_deg / n

    # 理论 hub 度数（双峰公式）
    if avg_k > 1 and (2 * avg_k - 1) > 0:
        beta = (2 * (avg_k ** 2) * ((avg_k - 1) ** 2) / (2 * avg_k - 1)) ** (1 / 3)
        k_theory = int(round(beta * (n ** (2 / 3))))
    else:
        k_theory = max(2, int(avg_k * 3))
    k_theory = min(k_theory, n - 1)

    # hub 度不能过大，必须给每个 leaf 留下至少 1 度
    max_k_hub = (total_deg - num_leaves) // num_hubs if num_hubs > 0 else n - 1
    k_theory = min(k_theory, max_k_hub)
    k_theory = max(k_theory, 2)

    # 从理论 hub 度向下搜索，优先保留较高的 hub 度，但要求序列可图化
    best_seq = None
    min_k_hub = max(2, int(np.ceil(avg_k)) + 1)
    for k_hub in range(k_theory, min_k_hub - 1, -1):
        degree_rem = total_deg - k_hub * num_hubs
        if degree_rem < num_leaves:
            continue
        k_leaf = degree_rem // num_leaves
        if k_leaf < 1 or k_leaf >= k_hub:
            continue
        rem = degree_rem % num_leaves
        seq = [k_hub] * num_hubs + [k_leaf + 1] * rem + [k_leaf] * (num_leaves - rem)
        if nx.is_graphical(seq):
            best_seq = seq
            break

    if best_seq is None:
        return create_bimodal_theoretical(n, m, seed=seed)

    G = nx.havel_hakimi_graph(best_seq)
    if not nx.is_connected(G):
        _ensure_connectivity(G, seed=seed)
        _adjust_edge_count(G, m, seed=seed)
    return G


def load_unified_model(model_path):
    ck = torch.load(model_path, map_location='cpu', weights_only=False)
    cfg = ck.get('model_config', {})
    in_channels = cfg.get('in_channels', 259)
    model = TopologyPolicy(
        in_channels=in_channels,
        hidden_channels=cfg.get('hidden_channels', 128),
        heads=cfg.get('heads', 4),
        num_layers=cfg.get('num_layers', 3),
        M_candidates=cfg.get('M_candidates', 50),
    )
    model.load_state_dict(ck['model_state_dict'])
    if torch.cuda.is_available():
        model = model.cuda()
    return model, in_channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default='src/train/v1/checkpoints/unified_ppo_agent_full_v4.pth')
    parser.add_argument('--data-dir', type=str, default='dataset/testdata')
    parser.add_argument('--output-dir', type=str, default='results/attack_auc_remove_comparison')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, in_channels = load_unified_model(args.model_path)

    data_dir = Path(args.data_dir)
    datasets = sorted([p.stem for p in data_dir.glob('*.gml') if p.stat().st_size > 0])

    method_defs = [
        ('Baseline', lambda G: G.copy()),
        ('GA', lambda G: nc.construct(G, algorithm='GA', seed=args.seed)),
        ('QDLM', lambda G: nc.construct(G, algorithm='QDLM', seed=args.seed, pop_size=20, max_iterations=25, verbose=False)),
        ('UNITY', lambda G: nc.construct(G, algorithm='UNITY', seed=args.seed)),
        ('BiT-HyRL-theory-1hub', lambda G: create_bimodal_theoretical(G.number_of_nodes(), G.number_of_edges(), seed=args.seed)),
        ('BiT-HyRL-two-peaks-r0.1', lambda G: create_bimodal_two_peaks(G.number_of_nodes(), G.number_of_edges(), hub_ratio=0.1, seed=args.seed)),
        ('Unified-PPO-U-add', lambda G: unified_solve(G, model, B_budget=3, k_ratio=0.1,
                                                       embed_dim=in_channels - 3, seed=args.seed, deterministic=True)['G_final']),
    ]

    records = []
    degree_dist_records = []

    for dataset in datasets:
        gml_path = data_dir / f'{dataset}.gml'
        G_raw, _ = load_graph(str(gml_path), verbose=False)
        G_raw = nx.Graph(G_raw)
        if not nx.is_connected(G_raw):
            G_raw = G_raw.subgraph(max(nx.connected_components(G_raw), key=len)).copy()
        G_int = to_int_graph(G_raw)
        target_m = G_int.number_of_edges()
        print(f"\n[{dataset}] N={G_int.number_of_nodes()}, M={target_m}")

        for method_name, method_fn in method_defs:
            try:
                G_out = method_fn(G_int)
            except Exception as e:
                print(f"  {method_name:25s} failed: {e}")
                continue
            G_out = nx.Graph(G_out)
            # 边数一致性简单处理
            if G_out.number_of_edges() > target_m:
                edges = list(G_out.edges())
                for i in range(G_out.number_of_edges() - target_m):
                    u, v = edges[i]
                    if G_out.has_edge(u, v):
                        G_out.remove_edge(u, v)
            elif G_out.number_of_edges() < target_m:
                nodes = list(G_out.nodes())
                for _ in range(target_m - G_out.number_of_edges()):
                    u, v = np.random.choice(nodes, 2, replace=False)
                    if not G_out.has_edge(u, v):
                        G_out.add_edge(u, v)

            n = G_out.number_of_nodes()
            degrees = dict(G_out.degree())
            unique_degrees = sorted(set(degrees.values()))
            degree_dist_records.append({
                'dataset': dataset,
                'method': method_name,
                'unique_degrees': unique_degrees,
                'n_unique_degrees': len(unique_degrees),
                'max_degree': max(degrees.values()) if degrees else 0,
                'min_degree': min(degrees.values()) if degrees else 0,
            })

            seq_rand = random_attack_sequence(G_out, seed=args.seed)
            seq_deg = degree_attack_sequence(G_out)

            gcc_rand = simulate_attack_gcc(G_out, seq_rand)
            gcc_deg = simulate_attack_gcc(G_out, seq_deg)

            auc_rand, rem_rand, rem_frac_rand = compute_auc_and_remove(gcc_rand)
            auc_deg, rem_deg, rem_frac_deg = compute_auc_and_remove(gcc_deg)

            records.append({
                'dataset': dataset,
                'method': method_name,
                'N': n,
                'M': G_out.number_of_edges(),
                'AUC_random': auc_rand,
                'AUC_degree': auc_deg,
                'AUC_weighted': 0.5 * auc_rand + 0.5 * auc_deg,
                'remove_num_random': rem_rand,
                'remove_num_degree': rem_deg,
                'remove_num_weighted': 0.5 * rem_rand + 0.5 * rem_deg,
                'remove_frac_random': rem_frac_rand,
                'remove_frac_degree': rem_frac_deg,
                'remove_frac_weighted': 0.5 * rem_frac_rand + 0.5 * rem_frac_deg,
            })
            print(f"  {method_name:25s} AUC(r/d/w)={auc_rand:.3f}/{auc_deg:.3f}/{0.5*auc_rand+0.5*auc_deg:.3f}  "
                  f"remove_frac(r/d/w)={rem_frac_rand:.3f}/{rem_frac_deg:.3f}/{0.5*rem_frac_rand+0.5*rem_frac_deg:.3f}")

    df = pd.DataFrame(records)
    df.to_csv(output_dir / 'attack_metrics.csv', index=False)
    pd.DataFrame(degree_dist_records).to_csv(output_dir / 'degree_distributions.csv', index=False)

    summary = df.groupby('method').agg({
        'AUC_random': 'mean',
        'AUC_degree': 'mean',
        'AUC_weighted': 'mean',
        'remove_frac_random': 'mean',
        'remove_frac_degree': 'mean',
        'remove_frac_weighted': 'mean',
        'remove_num_random': 'mean',
        'remove_num_degree': 'mean',
        'remove_num_weighted': 'mean',
    }).round(4).sort_values('AUC_weighted', ascending=False)
    summary.to_csv(output_dir / 'summary.csv')
    print("\n=== 汇总（按 AUC_weighted 排序）===")
    print(summary.to_string())


if __name__ == '__main__':
    main()
