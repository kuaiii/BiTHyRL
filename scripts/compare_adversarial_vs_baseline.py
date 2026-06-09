#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比对抗式训练模型与 Baseline 模型在多攻击验证下的性能。

用法:
  python scripts/compare_adversarial_vs_baseline.py \
      --adversarial results/adversarial_eval.csv \
      --baseline results/baseline_deep_gat_eval.csv
"""
import argparse
import pandas as pd
import numpy as np


def aggregate_by_method(df):
    return (
        df.groupby('attack_method')
        .agg({'gcc_auc': 'mean', 'collapse_ratio': 'mean', 'csa': 'mean', 'cce': 'mean', 'wcp': 'mean'})
        .reset_index()
    )


def main():
    parser = argparse.ArgumentParser(description='Compare adversarial vs baseline evaluation results')
    parser.add_argument('--adversarial', type=str, default='results/adversarial_eval.csv')
    parser.add_argument('--baseline', type=str, default='results/baseline_deep_gat_eval.csv')
    parser.add_argument('--output', type=str, default='results/adversarial_vs_baseline_summary.csv')
    args = parser.parse_args()

    adv = pd.read_csv(args.adversarial)
    base = pd.read_csv(args.baseline)

    adv_agg = aggregate_by_method(adv)
    base_agg = aggregate_by_method(base)

    merged = adv_agg.merge(base_agg, on='attack_method', suffixes=('_adv', '_base'))

    print('\n' + '=' * 110)
    print('Adversarial Model vs Baseline Model (跨数据集平均)')
    print('=' * 110)
    header = (
        'Attack Method'.ljust(15)
        + 'Adv GCC'.rjust(10)
        + 'Base GCC'.rjust(11)
        + 'ΔGCC'.rjust(10)
        + 'Adv CSA'.rjust(10)
        + 'Base CSA'.rjust(11)
        + 'Adv CCE'.rjust(10)
        + 'Base CCE'.rjust(11)
        + 'Adv WCP'.rjust(10)
        + 'Base WCP'.rjust(11)
    )
    print(header)
    print('-' * 110)

    for _, row in merged.iterrows():
        method = row['attack_method']
        print(
            f"{method:<15}"
            f"{row['gcc_auc_adv']:>10.4f}"
            f"{row['gcc_auc_base']:>11.4f}"
            f"{row['gcc_auc_adv'] - row['gcc_auc_base']:>+10.4f}"
            f"{row['csa_adv']:>10.4f}"
            f"{row['csa_base']:>11.4f}"
            f"{row['cce_adv']:>10.4f}"
            f"{row['cce_base']:>11.4f}"
            f"{row['wcp_adv']:>10.4f}"
            f"{row['wcp_base']:>11.4f}"
        )

    avg_delta_gcc = (merged['gcc_auc_adv'] - merged['gcc_auc_base']).mean()
    avg_delta_csa = (merged['csa_adv'] - merged['csa_base']).mean()
    avg_delta_cce = (merged['cce_adv'] - merged['cce_base']).mean()
    avg_delta_wcp = (merged['wcp_adv'] - merged['wcp_base']).mean()

    print('-' * 110)
    print(f"{'Average Δ':<15}{'':>10}{'':>11}{avg_delta_gcc:>+10.4f}{'':>10}{'':>11}{avg_delta_csa:>+10.4f}{'':>10}{'':>11}{avg_delta_cce:>+10.4f}{'':>10}{'':>11}{avg_delta_wcp:>+10.4f}")
    print('=' * 110)

    # 保存汇总 CSV
    merged.to_csv(args.output, index=False)
    print(f"\n汇总结果已保存: {args.output}")


if __name__ == '__main__':
    main()
