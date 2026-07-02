#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按照 docs/双峰网络实验代码.docx 中的伪代码重新进行双峰网络实验。

构造三种固定成本的网络：
- G1: 双峰网络（Bimodal），中心枢纽 + 均匀低度叶子
- G2: BA 无标度网络
- G3: ER 随机网络

在随机攻击与动态定向攻击（每次移除后重新按度排序）下比较：
- LCC 曲线 AUC
- 临界移除比例 remove_frac（GCC < 1% 原始规模时的移除比例）

输出：
- results/bimodal_docx/{dataset}.csv
- results/bimodal_docx/summary.csv
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
from src.topology.reconstruction import create_bimodal_network_docx


def to_int_graph(G):
    return nx.convert_node_labels_to_integers(G, label_attribute='orig_label')


def simulate_attack_gcc(G, attack_sequence):
    """按 attack_sequence 顺序移除节点，返回归一化 GCC 列表（包含初始值 1.0）。"""
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
    """计算 AUC 与临界移除比例（GCC 首次 <= 0.01 时的移除比例）。"""
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


def random_attack_sequence(G, seed=42):
    """随机攻击顺序。"""
    rng = np.random.RandomState(seed)
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    return nodes


def dynamic_degree_attack_sequence(G):
    """动态定向攻击：每次移除当前度最大的节点。"""
    G_work = G.copy()
    sequence = []
    while G_work.number_of_nodes() > 0:
        node = max(G_work.degree(), key=lambda x: x[1])[0]
        sequence.append(node)
        G_work.remove_node(node)
    return sequence


def build_ba_network(n, m, seed=42):
    """构造 BA 网络。取 m = round(M / N)，然后返回简单图。"""
    if n < 2:
        return nx.Graph()
    ba_m = max(1, int(round(m / n)))
    # BA(N, m) 要求 m < N
    ba_m = min(ba_m, n - 1)
    G = nx.barabasi_albert_graph(n, ba_m, seed=seed)
    return nx.Graph(G)


def build_er_network(n, m, seed=42):
    """构造 ER 随机网络 G(N, M)。"""
    return nx.gnm_random_graph(n, m, seed=seed)


def run_experiment(data_dir='dataset/testdata', output_dir='results/bimodal_docx', seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_dir)
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
        m = G_int.number_of_edges()
        print(f"\n[{dataset}] N={n}, M={m}, avg_k={2*m/n:.3f}")

        # 构造三种网络
        constructors = {}

        t0 = time.time()
        bimodal_res = create_bimodal_network_docx(G_int, seed=seed)
        bimodal_time = time.time() - t0
        if bimodal_res['G'] is None:
            print(f"  [跳过] 双峰网络不可构造")
            continue
        constructors['Bimodal-docx'] = (bimodal_res['G'], bimodal_time, bimodal_res)

        t0 = time.time()
        G_ba = build_ba_network(n, m, seed=seed)
        ba_time = time.time() - t0
        constructors['BA'] = (G_ba, ba_time, None)

        t0 = time.time()
        G_er = build_er_network(n, m, seed=seed)
        er_time = time.time() - t0
        constructors['ER'] = (G_er, er_time, None)

        records = []
        for name, (G_net, build_time, meta) in constructors.items():
            if G_net is None or G_net.number_of_nodes() == 0:
                continue

            seq_rand = random_attack_sequence(G_net, seed=seed)
            seq_deg = dynamic_degree_attack_sequence(G_net)

            gcc_rand = simulate_attack_gcc(G_net, seq_rand)
            gcc_deg = simulate_attack_gcc(G_net, seq_deg)

            auc_rand, rem_rand, rem_frac_rand = compute_auc_and_remove(gcc_rand)
            auc_deg, rem_deg, rem_frac_deg = compute_auc_and_remove(gcc_deg)

            degrees = sorted([d for _, d in G_net.degree()], reverse=True)
            k_max_actual = degrees[0] if degrees else 0
            k_min_actual = degrees[-1] if degrees else 0

            row = {
                'dataset': dataset,
                'method': name,
                'N': G_net.number_of_nodes(),
                'M_target': m,
                'M_actual': G_net.number_of_edges(),
                'time_sec': build_time,
                'k_min_target': meta['k_min_target'] if meta else None,
                'k_max_target': meta['k_max_target'] if meta else None,
                'strategy': meta['strategy'] if meta else None,
                'k_min_actual': k_min_actual,
                'k_max_actual': k_max_actual,
                'AUC_random': auc_rand,
                'AUC_degree': auc_deg,
                'AUC_weighted': 0.5 * auc_rand + 0.5 * auc_deg,
                'remove_num_random': rem_rand,
                'remove_num_degree': rem_deg,
                'remove_frac_random': rem_frac_rand,
                'remove_frac_degree': rem_frac_deg,
                'remove_frac_weighted': 0.5 * rem_frac_rand + 0.5 * rem_frac_deg,
            }
            records.append(row)
            print(f"  {name:15s} N={row['N']}, M={row['M_actual']}/{m}, "
                  f"k=(target {row['k_min_target'] or '-'}..{row['k_max_target'] or '-'}, "
                  f"actual {k_min_actual}..{k_max_actual}), strategy={row['strategy'] or '-'}, "
                  f"time={build_time:.4f}s, "
                  f"AUC(r/d/w)={auc_rand:.3f}/{auc_deg:.3f}/{row['AUC_weighted']:.3f}, "
                  f"remove_frac(r/d/w)={rem_frac_rand:.3f}/{rem_frac_deg:.3f}/{row['remove_frac_weighted']:.3f}")

        df = pd.DataFrame(records)
        df.to_csv(output_dir / f'{dataset}.csv', index=False)
        all_records.extend(records)

    full_df = pd.DataFrame(all_records)
    if not full_df.empty:
        full_df.to_csv(output_dir / 'all_records.csv', index=False)
        summary = full_df.groupby('method').agg({
            'N': 'mean',
            'M_target': 'mean',
            'M_actual': 'mean',
            'time_sec': 'mean',
            'k_min_actual': 'mean',
            'k_max_actual': 'mean',
            'AUC_random': 'mean',
            'AUC_degree': 'mean',
            'AUC_weighted': 'mean',
            'remove_frac_random': 'mean',
            'remove_frac_degree': 'mean',
            'remove_frac_weighted': 'mean',
        }).round(4)
        summary.to_csv(output_dir / 'summary.csv')
        print("\n=== 跨数据集平均 ===")
        print(summary.to_string())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='dataset/testdata')
    parser.add_argument('--output-dir', type=str, default='results/bimodal_docx')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_experiment(args.data_dir, args.output_dir, args.seed)
