#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描 BiT-HyRL 双峰网络的 hub 比例（hub_ratio / peaks_r），
使用 create_bimodal_network_exact（原代码实现）生成拓扑，
在 random / degree 攻击下评估 AUC 与 remove_frac。
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings('ignore')

from src.topology.generators import load_graph
from src.topology.reconstruction import create_bimodal_network_exact


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


def evaluate_one(G_int, hub_ratio, seed=42):
    n = G_int.number_of_nodes()
    hub_num = max(1, int(round(n * hub_ratio)))
    G_out = create_bimodal_network_exact(G_int, hub_num=hub_num, seed=seed)
    G_out = nx.Graph(G_out)
    n_out = G_out.number_of_nodes()
    m_out = G_out.number_of_edges()

    seq_rand = random_attack_sequence(G_out, seed=seed)
    seq_deg = degree_attack_sequence(G_out)

    gcc_rand = simulate_attack_gcc(G_out, seq_rand)
    gcc_deg = simulate_attack_gcc(G_out, seq_deg)

    auc_rand, rem_rand, rem_frac_rand = compute_auc_and_remove(gcc_rand)
    auc_deg, rem_deg, rem_frac_deg = compute_auc_and_remove(gcc_deg)

    degrees = dict(G_out.degree())
    unique_degrees = sorted(set(degrees.values()))

    return {
        'hub_ratio': hub_ratio,
        'hub_num': hub_num,
        'N': n_out,
        'M': m_out,
        'AUC_random': auc_rand,
        'AUC_degree': auc_deg,
        'AUC_weighted': 0.5 * auc_rand + 0.5 * auc_deg,
        'remove_frac_random': rem_frac_rand,
        'remove_frac_degree': rem_frac_deg,
        'remove_frac_weighted': 0.5 * rem_frac_rand + 0.5 * rem_frac_deg,
        'max_degree': max(degrees.values()) if degrees else 0,
        'min_degree': min(degrees.values()) if degrees else 0,
        'n_unique_degrees': len(unique_degrees),
        'unique_degrees': unique_degrees,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='dataset/testdata')
    parser.add_argument('--output-dir', type=str, default='results/bimodal_hub_ratio_scan')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ratios', type=str,
                        default='0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.25,0.30,0.35,0.40,0.45,0.50')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ratios = [float(x) for x in args.ratios.split(',')]
    data_dir = Path(args.data_dir)
    datasets = sorted([p.stem for p in data_dir.glob('*.gml') if p.stat().st_size > 0])

    records = []
    for dataset in datasets:
        gml_path = data_dir / f'{dataset}.gml'
        G_raw, _ = load_graph(str(gml_path), verbose=False)
        G_raw = nx.Graph(G_raw)
        if not nx.is_connected(G_raw):
            G_raw = G_raw.subgraph(max(nx.connected_components(G_raw), key=len)).copy()
        G_int = to_int_graph(G_raw)
        n = G_int.number_of_nodes()
        m = G_int.number_of_edges()
        print(f"\n[{dataset}] N={n}, M={m}")

        for r in ratios:
            hub_num = max(1, int(round(n * r)))
            if hub_num >= n:
                continue
            try:
                res = evaluate_one(G_int, r, seed=args.seed)
            except Exception as e:
                print(f"  r={r:.2f} failed: {e}")
                continue
            res['dataset'] = dataset
            records.append(res)
            print(f"  r={r:.2f} (h={hub_num:3d}) AUC(r/d/w)="
                  f"{res['AUC_random']:.3f}/{res['AUC_degree']:.3f}/{res['AUC_weighted']:.3f}  "
                  f"remove_frac(r/d/w)={res['remove_frac_random']:.3f}/{res['remove_frac_degree']:.3f}/{res['remove_frac_weighted']:.3f}")

    df = pd.DataFrame(records)
    df.to_csv(output_dir / 'scan_results.csv', index=False)

    # 按 AUC_weighted 找每个数据集最优的 r
    best_auc = df.loc[df.groupby('dataset')['AUC_weighted'].idxmax()][['dataset', 'hub_ratio', 'AUC_weighted', 'AUC_random', 'AUC_degree', 'remove_frac_weighted']]
    best_remove = df.loc[df.groupby('dataset')['remove_frac_weighted'].idxmax()][['dataset', 'hub_ratio', 'remove_frac_weighted', 'AUC_weighted']]

    best_auc.to_csv(output_dir / 'best_by_auc_weighted.csv', index=False)
    best_remove.to_csv(output_dir / 'best_by_remove_frac_weighted.csv', index=False)

    print("\n=== 每个数据集按 AUC_weighted 最优的 hub_ratio ===")
    print(best_auc.to_string(index=False))
    print("\n=== 每个数据集按 remove_frac_weighted 最优的 hub_ratio ===")
    print(best_remove.to_string(index=False))

    # 平均意义上的最优 r
    summary = df.groupby('hub_ratio').agg({
        'AUC_random': 'mean',
        'AUC_degree': 'mean',
        'AUC_weighted': 'mean',
        'remove_frac_random': 'mean',
        'remove_frac_degree': 'mean',
        'remove_frac_weighted': 'mean',
    }).round(4)
    summary.to_csv(output_dir / 'average_by_ratio.csv')
    print("\n=== 按 hub_ratio 平均 ===")
    print(summary.to_string())
    print(f"\n平均 AUC_weighted 最优 r = {summary['AUC_weighted'].idxmax():.2f}, 值 = {summary['AUC_weighted'].max():.4f}")
    print(f"平均 remove_frac_weighted 最优 r = {summary['remove_frac_weighted'].idxmax():.2f}, 值 = {summary['remove_frac_weighted'].max():.4f}")


if __name__ == '__main__':
    import argparse
    main()
