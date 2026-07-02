#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 docs/Bimodal.txt 里的“随机 + 目标攻击”综合指标（无控制器，只看 GCC）
验证双峰拓扑的韧性。对应 src.topology.reconstruction._evaluate_robustness 的
R_weight = 0.5 * R_degree + 0.5 * R_random。
"""
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings('ignore')

from src.topology.generators import load_graph
from src.topology.reconstruction import create_bimodal_theoretical, _evaluate_robustness
import network_construction as nc


def to_int_graph(G):
    return nx.convert_node_labels_to_integers(G, label_attribute='orig_label')


def main():
    data_dir = Path('dataset/testdata')
    datasets = sorted([p.stem for p in data_dir.glob('*.gml') if p.stat().st_size > 0])
    methods = ['Baseline', 'GA', 'QDLM', 'UNITY', 'BiT-HyRL-theory']
    records = []
    timing = []

    for dataset in datasets:
        gml_path = data_dir / f'{dataset}.gml'
        G_raw, _ = load_graph(str(gml_path), verbose=False)
        G_raw = nx.Graph(G_raw)
        if not nx.is_connected(G_raw):
            G_raw = G_raw.subgraph(max(nx.connected_components(G_raw), key=len)).copy()
        G_int = to_int_graph(G_raw)
        print(f"\n[{dataset}] N={G_int.number_of_nodes()}, M={G_int.number_of_edges()}")

        for method in methods:
            t0 = time.time()
            if method == 'Baseline':
                G_out = G_int.copy()
            elif method == 'GA':
                G_out = nc.construct(G_int, algorithm='GA', seed=42)
            elif method == 'QDLM':
                G_out = nc.construct(G_int, algorithm='QDLM', seed=42, pop_size=20, max_iterations=25, verbose=False)
            elif method == 'UNITY':
                G_out = nc.construct(G_int, algorithm='UNITY', seed=42)
            elif method == 'BiT-HyRL-theory':
                G_out = create_bimodal_theoretical(G_int.number_of_nodes(), G_int.number_of_edges(), seed=42)
            else:
                continue
            elapsed = time.time() - t0

            if G_out.number_of_edges() != G_int.number_of_edges():
                # quick edge count fix
                if G_out.number_of_edges() > G_int.number_of_edges():
                    for _ in range(G_out.number_of_edges() - G_int.number_of_edges()):
                        if G_out.number_of_edges() == 0:
                            break
                        G_out.remove_edge(*list(G_out.edges())[0])
                else:
                    nodes = list(G_out.nodes())
                    for _ in range(G_int.number_of_edges() - G_out.number_of_edges()):
                        u, v = nodes[:2]
                        if not G_out.has_edge(u, v):
                            G_out.add_edge(u, v)

            R_weight, R_degree, R_random = _evaluate_robustness(G_out, seed=42)
            records.append({
                'dataset': dataset,
                'method': method,
                'R_weight': R_weight,
                'R_degree': R_degree,
                'R_random': R_random,
            })
            timing.append({'dataset': dataset, 'method': method, 'time_sec': elapsed})
            print(f"  {method:20s} time={elapsed:7.3f}s  R_weight={R_weight:.4f}  R_degree={R_degree:.4f}  R_random={R_random:.4f}")

    df = pd.DataFrame(records)
    df.to_csv('results/bimodal_verification/percolation.csv', index=False)
    summary = df.groupby('method')[['R_weight', 'R_degree', 'R_random']].mean().sort_values('R_weight', ascending=False)
    summary['time_sec'] = pd.DataFrame(timing).groupby('method')['time_sec'].mean()
    summary.to_csv('results/bimodal_verification/percolation_summary.csv')
    print("\n=== 无控制器 percolation 指标汇总 ===")
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
