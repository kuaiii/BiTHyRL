#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在真实网络数据集上自动运行 main.py 对比实验，并汇总 Unified-PPO 与其他方法的 GCC R 值。

用法：
  python scripts/run_unified_ppo_comparison.py \
      --unified-ppo-model src/train/v1/checkpoints/unified_ppo_agent_full.pth \
      --datasets Chinanet,Colt,GtsCe,Cogentco,UsCarrier \
      --attacks random,degree --B-budget 5
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import csv
import subprocess
import json
import numpy as np
from pathlib import Path


def find_latest_experiment_dir(dataset, attack):
    base = os.path.join(ROOT, 'results', dataset, attack)
    if not os.path.exists(base):
        return None
    subdirs = [d for d in os.listdir(base) if d.isdigit()]
    if not subdirs:
        return None
    latest = max(int(d) for d in subdirs)
    return os.path.join(base, str(latest))


def read_metrics(exp_dir):
    path = os.path.join(exp_dir, 'comprehensive_metrics.csv')
    if not os.path.exists(path):
        return {}
    result = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['Method']
            r_key = [k for k in row.keys() if 'Robustness' in k]
            if r_key:
                try:
                    result[method] = float(row[r_key[0]])
                except ValueError:
                    pass
    return result


def main():
    parser = argparse.ArgumentParser(description='Unified-PPO multi-dataset comparison')
    parser.add_argument('--unified-ppo-model', type=str, required=True)
    parser.add_argument('--datasets', type=str, default='Chinanet,Colt,GtsCe,Cogentco,UsCarrier')
    parser.add_argument('--attacks', type=str, default='random,degree')
    parser.add_argument('--B-budget', type=int, default=5)
    parser.add_argument('--M-candidates', type=int, default=50)
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',')]
    attacks = [a.strip() for a in args.attacks.split(',')]

    all_methods = set()
    table = []

    for dataset in datasets:
        for attack in attacks:
            print(f"\n{'='*60}")
            print(f"Running {dataset} / {attack}")
            print('='*60)
            cmd = [
                'python', 'main.py',
                '--dataset', dataset,
                '--attack', attack,
                '--batch', str(args.batch),
                '--unified-ppo-model', args.unified_ppo_model,
                '--unified-ppo-B', str(args.B_budget),
                '--unified-ppo-M', str(args.M_candidates),
            ]
            try:
                subprocess.run(cmd, cwd=ROOT, check=True, timeout=600)
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            exp_dir = find_latest_experiment_dir(dataset, attack)
            if exp_dir is None:
                print(f"No experiment dir found for {dataset}/{attack}")
                continue
            metrics = read_metrics(exp_dir)
            print(f"Metrics: {metrics}")
            all_methods.update(metrics.keys())
            for method, r_val in metrics.items():
                table.append({
                    'dataset': dataset,
                    'attack': attack,
                    'method': method,
                    'R': r_val,
                })

    if not table:
        print("No results collected.")
        return

    # 汇总：每个方法在每个数据集/攻击下的平均 R
    methods = sorted(all_methods)
    summary = {}
    for row in table:
        key = (row['dataset'], row['attack'], row['method'])
        summary.setdefault(key, []).append(row['R'])

    output_dir = os.path.join(ROOT, 'results', 'unified_ppo_comparison')
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'comparison_summary.csv')
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Attack'] + methods)
        for dataset in datasets:
            for attack in attacks:
                row = [dataset, attack]
                for method in methods:
                    vals = summary.get((dataset, attack, method), [])
                    row.append(f"{np.mean(vals):.4f}" if vals else '')
                writer.writerow(row)

    # 计算每个方法在全部实验上的平均 R
    method_avg = {}
    for method in methods:
        vals = [v for k, v in summary.items() if k[2] == method]
        method_avg[method] = np.mean(vals) if vals else 0.0

    print("\n" + "="*60)
    print("Average Robustness (R) across all experiments")
    print("="*60)
    for method, avg_r in sorted(method_avg.items(), key=lambda x: -x[1]):
        print(f"{method:<15} {avg_r:.4f}")
    print(f"\nDetailed summary saved: {output_csv}")


if __name__ == '__main__':
    main()
