#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描双峰网络中“最大度节点所占总度数比例 r_max”对 random / degree 攻击 AUC 的影响。
构造方式：一个 hub 节点的度为 k_max = min(N-1, round(r_max * 2M))，
其余 N-1 个节点均分剩余度数作为 k_min。
"""
import os
import sys
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings('ignore')

from src.topology.generators import load_graph
from src.topology.reconstruction import _ensure_connectivity, _adjust_edge_count, _make_graphical


def to_int_graph(G):
    return nx.convert_node_labels_to_integers(G, label_attribute='orig_label')


def simulate_attack_gcc(G, attack_sequence):
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
    n_steps = len(gcc_ratios) - 1
    x = np.arange(len(gcc_ratios)) / n_steps if n_steps > 0 else np.array([0.0])
    auc = np.trapz(gcc_ratios, x)
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


def create_bimodal_degree_share(n, m, r_max, seed=42):
    """
    构造双峰网络：hub 节点度为 r_max * 2M（上限 N-1），
    其余 N-1 个节点均分剩余度数。
    """
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)

    if n < 2 or m < n - 1:
        return nx.path_graph(n) if n >= 2 else nx.Graph()

    D = 2 * m
    k_max_target = int(round(r_max * D))
    k_max = max(1, min(k_max_target, n - 1))

    n_leaves = n - 1
    D_leaves = D - k_max
    # 每个叶子至少 1 度
    if D_leaves < n_leaves:
        k_max = max(1, D - n_leaves)
        D_leaves = D - k_max

    k_min = D_leaves // n_leaves
    rem = D_leaves % n_leaves
    seq = [k_max] + [k_min + 1] * rem + [k_min] * (n_leaves - rem)

    # 若不可图化则做保守修正
    if not nx.is_graphical(seq):
        seq = _make_graphical(seq.copy())

    try:
        G = nx.havel_hakimi_graph(seq)
    except Exception:
        G_multi = nx.configuration_model(seq, seed=seed)
        G = nx.Graph(G_multi)
        G.remove_edges_from(nx.selfloop_edges(G))

    if not nx.is_connected(G):
        _ensure_connectivity(G, seed=seed)
        _adjust_edge_count(G, m, seed=seed)

    return G


def main():
    parser = __import__('argparse').ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='dataset/testdata')
    parser.add_argument('--output-dir', type=str, default='results/bimodal_degree_share_scan')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--r-max-list', type=str, default='1.0,0.30,0.20,0.15,0.12,0.08,0.04,0.01')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    r_max_list = [float(x) for x in args.r_max_list.split(',')]
    data_dir = Path(args.data_dir)
    datasets = sorted([p.stem for p in data_dir.glob('*.gml') if p.stat().st_size > 0])

    all_records = []

    for dataset in datasets:
        gml_path = data_dir / f'{dataset}.gml'
        G_raw, _ = load_graph(str(gml_path), verbose=False)
        G_raw = nx.Graph(G_raw)
        if not nx.is_connected(G_raw):
            G_raw = G_raw.subgraph(max(nx.connected_components(G_raw), key=len)).copy()
        G_int = to_int_graph(G_raw)
        n = G_int.number_of_nodes()
        target_m = G_int.number_of_edges()
        print(f"\n[{dataset}] N={n}, M={target_m}")

        records = []
        for r_max in r_max_list:
            t0 = time.time()
            G_out = create_bimodal_degree_share(n, target_m, r_max, seed=args.seed)
            build_time = time.time() - t0
            G_out = nx.Graph(G_out)

            # 最终校验并调整边数
            if G_out.number_of_edges() != target_m:
                _adjust_edge_count(G_out, target_m, seed=args.seed)

            seq_rand = random_attack_sequence(G_out, seed=args.seed)
            seq_deg = degree_attack_sequence(G_out)
            auc_rand, rem_rand, rem_frac_rand = compute_auc_and_remove(simulate_attack_gcc(G_out, seq_rand))
            auc_deg, rem_deg, rem_frac_deg = compute_auc_and_remove(simulate_attack_gcc(G_out, seq_deg))

            degrees = sorted([d for _, d in G_out.degree()], reverse=True)
            records.append({
                'dataset': dataset,
                'r_max': r_max,
                'N': G_out.number_of_nodes(),
                'M': G_out.number_of_edges(),
                'time_sec': build_time,
                'k_max': degrees[0] if degrees else 0,
                'k_min': degrees[-1] if degrees else 0,
                'AUC_random': auc_rand,
                'AUC_degree': auc_deg,
                'AUC_weighted': 0.5 * auc_rand + 0.5 * auc_deg,
                'remove_frac_random': rem_frac_rand,
                'remove_frac_degree': rem_frac_deg,
                'remove_frac_weighted': 0.5 * rem_frac_rand + 0.5 * rem_frac_deg,
            })
            print(f"  r_max={r_max:5.2f}  k_max={degrees[0] if degrees else 0:3d}  k_min={degrees[-1] if degrees else 0:3d}  "
                  f"AUC(r/d/w)={auc_rand:.3f}/{auc_deg:.3f}/{0.5*auc_rand+0.5*auc_deg:.3f}  "
                  f"remove_frac(r/d/w)={rem_frac_rand:.3f}/{rem_frac_deg:.3f}/{0.5*rem_frac_rand+0.5*rem_frac_deg:.3f}")

        df = pd.DataFrame(records)
        df.to_csv(output_dir / f'{dataset}_degree_share.csv', index=False)
        all_records.extend(records)

    full_df = pd.DataFrame(all_records)
    full_df.to_csv(output_dir / 'all_degree_share.csv', index=False)

    # 跨数据集平均
    avg = full_df.groupby('r_max').agg({
        'AUC_random': 'mean',
        'AUC_degree': 'mean',
        'AUC_weighted': 'mean',
        'remove_frac_random': 'mean',
        'remove_frac_degree': 'mean',
        'remove_frac_weighted': 'mean',
        'time_sec': 'mean',
    }).round(4)
    avg.to_csv(output_dir / 'average_by_rmax.csv')
    print("\n=== 跨数据集平均 ===")
    print(avg.to_string())


if __name__ == '__main__':
    main()
