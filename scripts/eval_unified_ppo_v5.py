#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified-PPO v5 快速评估脚本：支持动态 B 预算。
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topology.generators import load_graph
from src.unified_ppo.topology_policy import TopologyPolicy
from src.unified_ppo.inference import unified_solve
from src.unified_ppo.action_space import CandidatePoolGenerator
from src.bit_hyrl.reward import _simulate_attack_sequence
from network_dismantling.unified_interface import dismantle


ADVERSARIAL_ATTACK_METHODS = ['degree', 'betweenness', 'pagerank', 'eigenvector', 'random']


def robustness_r(G, controllers, attack_method='degree', attack_ratio=0.15):
    try:
        seq = dismantle(G, attack_method)
        if seq is None or len(seq) == 0:
            return np.nan
        retention = _simulate_attack_sequence(G, controllers, seq, attack_ratio)
        return retention
    except Exception:
        return np.nan


def dynamic_b_budget(G, base_b=3, scale=0.02, max_b=15):
    return min(max_b, base_b + int(G.number_of_nodes() * scale))


def evaluate_model(model_path, datasets, data_dir='dataset/testdata',
                   k_ratio=0.1, base_b=3, dynamic_b=False, b_scale=0.02, b_max=15,
                   attack_ratio=0.15, seed=42):
    ck = torch.load(model_path, map_location='cpu', weights_only=False)
    cfg = ck.get('model_config', {})
    in_channels = cfg.get('in_channels', 259)
    hidden_channels = cfg.get('hidden_channels', 128)
    heads = cfg.get('heads', 4)
    num_layers = cfg.get('num_layers', 3)
    M_candidates = cfg.get('M_candidates', 50)

    model = TopologyPolicy(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        heads=heads,
        num_layers=num_layers,
        M_candidates=M_candidates,
    )
    model.load_state_dict(ck['model_state_dict'])
    if torch.cuda.is_available():
        model = model.cuda()

    rows = []
    for dataset in datasets:
        gml_path = Path(data_dir) / f'{dataset}.gml'
        if not gml_path.exists():
            print(f"[warn] {gml_path} not found")
            continue
        G, _ = load_graph(str(gml_path), verbose=False)
        if G is None:
            continue

        B = dynamic_b_budget(G, base_b, b_scale, b_max) if dynamic_b else base_b
        print(f"[{dataset}] N={G.number_of_nodes()}, B={B}")

        result = unified_solve(
            G, model, B_budget=B, k_ratio=k_ratio,
            embed_dim=in_channels - 3, seed=seed, deterministic=True
        )
        G_final = result['G_final']
        controllers = result['controllers']

        for attack in ADVERSARIAL_ATTACK_METHODS:
            r = robustness_r(G_final, controllers, attack_method=attack, attack_ratio=attack_ratio)
            rows.append({
                'Dataset': dataset,
                'Attack': attack,
                'B': B,
                'Robustness': r,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='dataset/testdata')
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--base-b', type=int, default=3)
    parser.add_argument('--dynamic-b', action='store_true')
    parser.add_argument('--b-scale', type=float, default=0.02)
    parser.add_argument('--b-max', type=int, default=15)
    parser.add_argument('--k-ratio', type=float, default=0.1)
    parser.add_argument('--attack-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results/unified_ppo_v5_eval.csv')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',')]
    else:
        datasets = sorted([p.stem for p in data_dir.glob('*.gml') if p.stat().st_size > 0])

    df = evaluate_model(
        args.model_path, datasets, data_dir=args.data_dir,
        k_ratio=args.k_ratio, base_b=args.base_b, dynamic_b=args.dynamic_b,
        b_scale=args.b_scale, b_max=args.b_max,
        attack_ratio=args.attack_ratio, seed=args.seed
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\n=== Average Robustness by Attack ===")
    print(df.groupby('Attack')['Robustness'].mean().round(4))
    print(f"\nOverall average R: {df['Robustness'].mean():.4f}")
    print(f"Results saved: {out_path}")


if __name__ == '__main__':
    main()
