#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅比较单层网络构造方法在 random / degree / PageRank 攻击下的表现。
方法：Baseline、GA、QDLM、UNITY，以及 BiT-HyRL 双峰在不同 hub 比例 r。
每个拓扑输出一张表（CSV），行是方法，列是攻击指标。
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
from src.topology.reconstruction import create_bimodal_network_exact, create_bimodal_network_docx, _ensure_connectivity
from scripts.normalize_edge_count import normalize_edge_count
import network_construction as nc
import time


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
    """动态定向攻击：每次移除当前度最大的节点。"""
    G_work = G.copy()
    sequence = []
    while G_work.number_of_nodes() > 0:
        node = max(G_work.degree(), key=lambda x: x[1])[0]
        sequence.append(node)
        G_work.remove_node(node)
    return sequence


def random_attack_sequence(G, seed=42):
    rng = np.random.RandomState(seed)
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    return nodes


def pr_attack_sequence(G, alpha=0.85):
    pr = nx.pagerank(G, alpha=alpha)
    return [n for n, _ in sorted(pr.items(), key=lambda x: x[1], reverse=True)]


def ensure_connected_and_normalize(G_out, target_m, seed=42):
    """保证连通并把边数调整到 target_m。"""
    G_out = nx.Graph(G_out)
    if not nx.is_connected(G_out):
        _ensure_connectivity(G_out, seed=seed)
    if G_out.number_of_edges() != target_m:
        G_out, _ = normalize_edge_count(G_out, target_m, seed=seed)
    return G_out


def evaluate_attacks(G_out, seed=42):
    seq_rand = random_attack_sequence(G_out, seed=seed)
    seq_deg = degree_attack_sequence(G_out)
    seq_pr = pr_attack_sequence(G_out)

    gcc_rand = simulate_attack_gcc(G_out, seq_rand)
    gcc_deg = simulate_attack_gcc(G_out, seq_deg)
    gcc_pr = simulate_attack_gcc(G_out, seq_pr)

    auc_rand, rem_rand, rem_frac_rand = compute_auc_and_remove(gcc_rand)
    auc_deg, rem_deg, rem_frac_deg = compute_auc_and_remove(gcc_deg)
    auc_pr, rem_pr, rem_frac_pr = compute_auc_and_remove(gcc_pr)

    return {
        'AUC_random': auc_rand,
        'AUC_degree': auc_deg,
        'AUC_PR': auc_pr,
        'AUC_avg': (auc_rand + auc_deg + auc_pr) / 3.0,
        'remove_num_random': rem_rand,
        'remove_num_degree': rem_deg,
        'remove_num_PR': rem_pr,
        'remove_frac_random': rem_frac_rand,
        'remove_frac_degree': rem_frac_deg,
        'remove_frac_PR': rem_frac_pr,
        'remove_frac_avg': (rem_frac_rand + rem_frac_deg + rem_frac_pr) / 3.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='dataset/testdata')
    parser.add_argument('--output-dir', type=str, default='results/single_layer_attack_comparison')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ratios', type=str, default='0.05,0.10,0.12,0.14,0.15,0.17,0.20')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ratios = [float(x) for x in args.ratios.split(',')]
    data_dir = Path(args.data_dir)
    datasets = sorted([p.stem for p in data_dir.glob('*.gml') if p.stat().st_size > 0])

    all_summaries = []

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

        # 定义方法
        def build_ba(G):
            n = G.number_of_nodes()
            target_m = G.number_of_edges()
            ba_m = max(1, int(round(target_m / (n - 1))))
            return nx.barabasi_albert_graph(n, ba_m, seed=args.seed)

        def build_random(G):
            n = G.number_of_nodes()
            target_m = G.number_of_edges()
            return nx.gnm_random_graph(n, target_m, seed=args.seed)

        method_defs = [('Baseline', lambda G: G.copy())]
        method_defs.append(('Random', build_random))
        method_defs.append(('BA', build_ba))
        method_defs.append(('GA', lambda G: nc.construct(G, algorithm='GA', seed=args.seed)))
        method_defs.append(('QDLM', lambda G: nc.construct(G, algorithm='QDLM', seed=args.seed, pop_size=20, max_iterations=25, verbose=False)))
        method_defs.append(('UNITY', lambda G: nc.construct(G, algorithm='UNITY', seed=args.seed)))
        method_defs.append(('BiT-HyRL-docx', lambda G: create_bimodal_network_docx(G, seed=args.seed)['G']))
        for r in ratios:
            method_defs.append((f'BiT-HyRL-r{r:.2f}', lambda G, r=r: create_bimodal_network_exact(G, hub_num=max(1, int(round(G.number_of_nodes() * r))), seed=args.seed)))

        records = []
        for method_name, method_fn in method_defs:
            try:
                t0 = time.time()
                G_out = method_fn(G_int)
                build_time = time.time() - t0
            except Exception as e:
                print(f"  {method_name:20s} failed: {e}")
                continue
            G_out = nx.Graph(G_out)
            # Option A：BiT-HyRL-docx 保持配置模型后的实际边数，不进行强制 M 归一化
            if method_name != 'BiT-HyRL-docx':
                G_out = ensure_connected_and_normalize(G_out, target_m, seed=args.seed)

            res = evaluate_attacks(G_out, seed=args.seed)
            res['method'] = method_name
            res['N'] = G_out.number_of_nodes()
            res['M'] = G_out.number_of_edges()
            res['time_sec'] = build_time
            records.append(res)
            print(f"  {method_name:20s} time={build_time:.3f}s  AUC(r/d/p/avg)="
                  f"{res['AUC_random']:.3f}/{res['AUC_degree']:.3f}/{res['AUC_PR']:.3f}/{res['AUC_avg']:.3f}  "
                  f"remove_frac(r/d/p/avg)={res['remove_frac_random']:.3f}/{res['remove_frac_degree']:.3f}/"
                  f"{res['remove_frac_PR']:.3f}/{res['remove_frac_avg']:.3f}")

        df = pd.DataFrame(records)
        cols = ['method', 'N', 'M', 'time_sec',
                'AUC_random', 'AUC_degree', 'AUC_PR', 'AUC_avg',
                'remove_frac_random', 'remove_frac_degree', 'remove_frac_PR', 'remove_frac_avg']
        df = df[cols]
        df.to_csv(output_dir / f'{dataset}_table.csv', index=False)

        # 为总表记录平均
        summary_row = df.set_index('method').mean(numeric_only=True).to_dict()
        summary_row['dataset'] = dataset
        all_summaries.append(summary_row)

    # 跨数据集平均汇总
    summary_df = pd.DataFrame(all_summaries)
    if not summary_df.empty:
        summary_df = summary_df[['dataset'] + [c for c in summary_df.columns if c != 'dataset']]
        summary_df.to_csv(output_dir / 'summary_avg.csv', index=False)
        print("\n=== 跨数据集平均汇总 ===")
        print(summary_df.to_string(index=False))


if __name__ == '__main__':
    import argparse
    main()
