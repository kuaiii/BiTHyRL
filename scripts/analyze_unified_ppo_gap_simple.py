# -*- coding: utf-8 -*-
"""
轻量分析 Unified-PPO 与基线的性能差距。
读取 run_real_network_comparison.py 生成的 summary CSV，输出排名、差距与改进提示。
"""
import os
import sys
import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_summary(csv_path):
    df = pd.read_csv(csv_path)
    # 统一 robustness 列：main.py 对不同攻击使用不同列名
    def get_robustness(row):
        col = f"Robustness (R - {row['Attack']})"
        if col in row.index:
            return row[col]
        # 兼容旧版只有 random/degree 的情况
        if row['Attack'] == 'random':
            return row.get('Robustness (R - random)')
        return row.get('Robustness (R - degree)')
    df['Robustness'] = df.apply(get_robustness, axis=1)
    df['Robustness'] = pd.to_numeric(df['Robustness'], errors='coerce')
    df = df.dropna(subset=['Robustness'])
    return df


def rank_analysis(df):
    print("\n" + "=" * 80)
    print("各方法平均 Robustness 与平均排名")
    print("=" * 80)

    mean_r = df.groupby('Method')['Robustness'].mean().sort_values(ascending=False)
    print("\nMean Robustness:")
    print(mean_r.round(4).to_string())

    ranks = []
    for (ds, atk), g in df.groupby(['Dataset', 'Attack']):
        sub = g.sort_values('Robustness', ascending=False).reset_index(drop=True)
        sub['Rank'] = sub.index + 1
        ranks.append(sub[['Method', 'Rank']])
    rank_df = pd.concat(ranks)
    avg_rank = rank_df.groupby('Method')['Rank'].mean().sort_values()
    print("\nAverage Rank (lower is better):")
    print(avg_rank.round(2).to_string())

    # wins / second
    methods = sorted(df['Method'].unique())
    wins = {m: 0 for m in methods}
    seconds = {m: 0 for m in methods}
    for (ds, atk), g in df.groupby(['Dataset', 'Attack']):
        sub = g.sort_values('Robustness', ascending=False).reset_index(drop=True)
        wins[sub.iloc[0]['Method']] += 1
        if len(sub) > 1:
            seconds[sub.iloc[1]['Method']] += 1

    print("\nWins / Second counts:")
    for m in methods:
        print(f"  {m:15s}: {wins[m]:2d} wins, {seconds[m]:2d} second")

    return mean_r, avg_rank


def gap_analysis(df, target='Unified-PPO'):
    print("\n" + "=" * 80)
    print(f"Unified-PPO ({target}) 与各基线的 Robustness 差距")
    print("=" * 80)

    if target not in df['Method'].unique():
        print(f"[warn] 未找到 {target} 的结果")
        return None

    pivot = df.pivot_table(index=['Dataset', 'Attack'], columns='Method', values='Robustness')
    pivot = pivot.dropna()
    baseline_methods = [m for m in pivot.columns if m != target]

    gap_records = []
    for method in baseline_methods:
        diff = pivot[method] - pivot[target]
        mean_gap = diff.mean()
        worse_cases = (diff > 0).sum()
        total = len(diff)
        print(f"\nvs {method}:")
        print(f"  平均差距: {mean_gap:+.4f} (正数表示 {method} 更强)")
        print(f"  {method} 更优的场次: {worse_cases}/{total}")
        print(f"  最大落后: {diff.max():+.4f}")
        print(f"  最大领先: {diff.min():+.4f}")
        gap_records.append({
            'Baseline': method,
            'MeanGap': mean_gap,
            'WorseCount': worse_cases,
            'Total': total,
            'MaxBehind': diff.max(),
            'MaxAhead': diff.min(),
        })

    # per attack type
    print("\n" + "=" * 80)
    print("不同攻击方式下 Unified-PPO 的平均 Robustness")
    print("=" * 80)
    attack_mean = df[df['Method'] == target].groupby('Attack')['Robustness'].mean().sort_values(ascending=False)
    print(attack_mean.round(4).to_string())

    # per dataset
    print("\n" + "=" * 80)
    print("不同数据集下 Unified-PPO 的平均 Robustness")
    print("=" * 80)
    ds_mean = df[df['Method'] == target].groupby('Dataset')['Robustness'].mean().sort_values(ascending=False)
    print(ds_mean.round(4).to_string())

    return pd.DataFrame(gap_records)


def worst_cases(df, target='Unified-PPO', top_k=10):
    print("\n" + "=" * 80)
    print(f"Unified-PPO 最差的 {top_k} 个 (Dataset, Attack)")
    print("=" * 80)
    sub = df[df['Method'] == target][['Dataset', 'Attack', 'Robustness']].sort_values('Robustness')
    print(sub.head(top_k).to_string(index=False))


def recommendations(mean_r, avg_rank, gap_df):
    print("\n" + "=" * 80)
    print("基于轻量分析的改进建议")
    print("=" * 80)

    unified_rank = avg_rank.get('Unified-PPO', None)
    unified_mean = mean_r.get('Unified-PPO', None)
    if unified_rank is not None:
        print(f"- Unified-PPO 平均排名: {unified_rank:.2f}/{len(avg_rank)}")
    if unified_mean is not None:
        print(f"- Unified-PPO 平均 Robustness: {unified_mean:.4f}")

    if gap_df is not None and not gap_df.empty:
        # 找出平均差距最大的基线（即 Unified-PPO 最落后的对手）
        worst_baseline = gap_df.loc[gap_df['MeanGap'].idxmax()]
        print(f"- 相对于 {worst_baseline['Baseline']} 平均落后最多: {worst_baseline['MeanGap']:+.4f}")
        if worst_baseline['MeanGap'] > 0.03:
            print(f"  → 建议重点研究 {worst_baseline['Baseline']} 的拓扑/控制器策略，作为下一步改进方向。")
        if gap_df['WorseCount'].sum() > gap_df['Total'].sum() * 0.5:
            print("- 在多数场次中 Unified-PPO 均落后，说明整体策略仍有较大提升空间。")
        else:
            print("- Unified-PPO 在部分场次已具备竞争力，可针对落后场景做定向优化。")

    print("- 若训练奖励长期平台期，可尝试：")
    print("  1. 降低学习率或增大 topology 头学习率；")
    print("  2. 增加训练数据中真实网络/大图比例；")
    print("  3. 对落后攻击类型（如 betweenness/eigenvector）提高奖励权重；")
    print("  4. 增大 B-budget 或引入两阶段后处理（先 PPO 选边，再 RCP 微调控制器）。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/unified_ppo_comparison_summary.csv",
                        help="run_real_network_comparison.py 生成的汇总 CSV")
    parser.add_argument("--target", default="Unified-PPO")
    parser.add_argument("--output-dir", default="results/unified_ppo_gap_analysis")
    args = parser.parse_args()

    csv_path = ROOT / args.summary
    if not csv_path.exists():
        print(f"[error] 未找到汇总文件: {csv_path}")
        print("请先运行: python scripts/run_real_network_comparison.py --model-path <ckpt> --B-budget 5")
        return 1

    df = load_summary(csv_path)
    mean_r, avg_rank = rank_analysis(df)
    gap_df = gap_analysis(df, target=args.target)
    worst_cases(df, target=args.target)
    recommendations(mean_r, avg_rank, gap_df)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if gap_df is not None:
        gap_df.to_csv(out_dir / "gap_summary.csv", index=False)
    df.to_csv(out_dir / "processed_summary.csv", index=False)
    print(f"\n分析结果已保存: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
