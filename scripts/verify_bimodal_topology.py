#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证双峰（Bimodal）拓扑在保持 N/M/k 一致时的鲁棒性与重构速度。

与 Baseline、GA、QDLM、UNITY、Unified-PPO-U-add 在相同控制器（PPO Phase2）下对比。
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
from src.topology.reconstruction import create_bimodal_theoretical
from src.unified_ppo.topology_policy import TopologyPolicy
from src.unified_ppo.inference import unified_solve
from src.unified_ppo.observation import GraphState, build_observation
from src.bit_hyrl import config as bit_config
from src.bit_hyrl.reward import _simulate_attack_sequence
import network_construction as nc
from network_dismantling.unified_interface import dismantle
from scripts.normalize_edge_count import normalize_edge_count
from scripts.fair_topology_comparison import (
    to_int_graph, compute_topology_metrics, ppo_select_controllers, robustness_r,
    load_unified_model, ADVERSARIAL_ATTACK_METHODS
)


def generate_bimodal_correct(G_int, seed=42):
    """使用 docs/Bimodal.txt 中的理论双峰构造（num_hubs=1, k_max ~ beta N^{2/3}）。"""
    n = G_int.number_of_nodes()
    m = G_int.number_of_edges()
    return create_bimodal_theoretical(n, m, seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default='src/train/v1/checkpoints/unified_ppo_agent_full_v4.pth')
    parser.add_argument('--data-dir', type=str, default='dataset/testdata')
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='results/bimodal_verification')
    parser.add_argument('--k-ratio', type=float, default=0.1)
    parser.add_argument('--attack-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, in_channels = load_unified_model(args.model_path)

    data_dir = Path(args.data_dir)
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',')]
    else:
        datasets = sorted([p.stem for p in data_dir.glob('*.gml') if p.stat().st_size > 0])

    methods = ['Baseline', 'GA', 'QDLM', 'UNITY', 'BiT-HyRL-theory', 'Unified-PPO-U-add']
    records = []
    timing = []

    for dataset in datasets:
        gml_path = data_dir / f'{dataset}.gml'
        if not gml_path.exists():
            continue
        G_raw, _ = load_graph(str(gml_path), verbose=False)
        if G_raw is None:
            continue
        G_raw = nx.Graph(G_raw)
        if not nx.is_connected(G_raw):
            G_raw = G_raw.subgraph(max(nx.connected_components(G_raw), key=len)).copy()
        G_int = to_int_graph(G_raw)
        target_edges = G_int.number_of_edges()
        print(f"\n[{dataset}] N={G_int.number_of_nodes()}, M={target_edges}")

        for method in methods:
            t0 = time.time()
            try:
                if method == 'Baseline':
                    G_out = G_int.copy()
                elif method == 'GA':
                    G_out = nc.construct(G_int, algorithm='GA', seed=args.seed)
                elif method == 'QDLM':
                    G_out = nc.construct(G_int, algorithm='QDLM', seed=args.seed, pop_size=20, max_iterations=25, verbose=False)
                elif method == 'UNITY':
                    G_out = nc.construct(G_int, algorithm='UNITY', seed=args.seed)
                elif method == 'BiT-HyRL-theory':
                    G_out = generate_bimodal_correct(G_int, seed=args.seed)
                elif method == 'Unified-PPO-U-add':
                    res = unified_solve(G_int, model, B_budget=3, k_ratio=args.k_ratio,
                                        embed_dim=in_channels - 3, seed=args.seed, deterministic=True)
                    G_out = res['G_final']
                else:
                    raise ValueError(method)
            except Exception as e:
                print(f"  {method} failed: {e}")
                continue
            elapsed = time.time() - t0
            G_out = nx.Graph(G_out)
            if G_out.number_of_edges() != target_edges:
                G_out, _ = normalize_edge_count(G_out, target_edges, seed=args.seed)
            timing.append({'dataset': dataset, 'method': method, 'time_sec': elapsed})

            controllers = ppo_select_controllers(G_out, model, in_channels, k_ratio=args.k_ratio, seed=args.seed)
            metrics = compute_topology_metrics(G_out)
            rec = {'dataset': dataset, 'method': method, 'time_sec': elapsed}
            rec.update(metrics)
            for attack in ADVERSARIAL_ATTACK_METHODS:
                rec[f'R_{attack}'] = robustness_r(G_out, controllers, attack_method=attack, attack_ratio=args.attack_ratio)
            records.append(rec)
            print(f"  {method:20s} time={elapsed:7.3f}s  R_avg={np.mean([rec[f'R_{a}'] for a in ADVERSARIAL_ATTACK_METHODS]):.3f}")

    df = pd.DataFrame(records)
    df.to_csv(output_dir / 'verification.csv', index=False)
    pd.DataFrame(timing).to_csv(output_dir / 'timing.csv', index=False)

    # 汇总表
    attack_cols = [f'R_{a}' for a in ADVERSARIAL_ATTACK_METHODS]
    summary = df.groupby('method')[attack_cols + ['time_sec']].mean()
    summary['R_avg'] = summary[attack_cols].mean(axis=1)
    summary = summary.sort_values('R_avg', ascending=False)
    summary.to_csv(output_dir / 'summary.csv')
    print("\n=== 汇总（PPO 统一控制器）===")
    print(summary.round(4).to_string())

    # 按数据集
    dataset_summary = df.groupby(['dataset', 'method']).apply(
        lambda g: pd.Series({'R_avg': g[attack_cols].values.mean()})
    ).unstack().round(4)
    dataset_summary.to_csv(output_dir / 'dataset_summary.csv')


if __name__ == '__main__':
    main()
