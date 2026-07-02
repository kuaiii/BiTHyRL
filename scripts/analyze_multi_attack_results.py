#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总多攻击方式下的实验结果，重点对比 BiT-HyRL 与基线。
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

METHODS = ["Baseline", "GA+RL", "GA+RCP", "Onion+RL", "ROMEN+RL", "UNITY+RL", "FRED-ABL", "QDLM", "BiT-HyRL"]
METRICS = ["gcc", "csa", "cce", "wcp"]


def find_latest_complete_dir(base_dir, attack):
    if not base_dir.exists():
        return None
    subdirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not subdirs:
        return None
    for d in sorted(subdirs, key=lambda x: int(x.name), reverse=True):
        complete = True
        for metric in METRICS:
            if not (d / metric / "area" / f"area_values_{attack}_{metric}.csv").exists():
                complete = False
                break
        if complete:
            return d
    return None


def read_area_csv(path):
    data = {}
    if not path.exists():
        return data
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("Method", "").strip()
            if method not in METHODS:
                continue
            try:
                data[method] = float(row.get("Mean_Area", row.get("Mean_R", "0")))
            except Exception:
                pass
    return data


def rank_methods(data, higher_is_better=True):
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=higher_is_better)
    return {m: r for r, (m, _) in enumerate(sorted_items, 1)}


def summarize_attack(dataset, attack):
    base = RESULTS_DIR / dataset / attack
    exp_dir = find_latest_complete_dir(base, attack)
    if exp_dir is None:
        return None
    result = {"dataset": dataset, "attack": attack, "exp_id": exp_dir.name}
    for metric in METRICS:
        csv_path = exp_dir / metric / "area" / f"area_values_{attack}_{metric}.csv"
        data = read_area_csv(csv_path)
        ranks = rank_methods(data)
        result[metric] = {"values": data, "ranks": ranks}
    return result


def format_cell(value, rank):
    if value != value:
        return "-"
    return f"{value:.4f}(#{rank})"


def main():
    attacks = sys.argv[1:] if len(sys.argv) > 1 else ["degree", "wgcc", "betweenness", "random"]
    datasets = ["GtsCe", "Chinanet", "Colt", "UsCarrier", "Cogentco"]

    all_data = defaultdict(lambda: defaultdict(dict))
    for attack in attacks:
        for ds in datasets:
            s = summarize_attack(ds, attack)
            if s:
                all_data[attack][ds] = s

    # 输出每个攻击下 BiT-HyRL 的 GCC Area 与排名
    print("\n" + "=" * 100)
    print("BiT-HyRL 在不同攻击方式下的 GCC Area 表现")
    print("=" * 100)
    header = f"{'Attack':<12}" + "".join(f"{ds:>14}" for ds in datasets) + f"{'Avg Rank':>10}"
    print(header)
    print("-" * len(header))

    for attack in attacks:
        ranks = []
        cells = []
        for ds in datasets:
            s = all_data[attack].get(ds)
            if s and "gcc" in s and "BiT-HyRL" in s["gcc"]["values"]:
                val = s["gcc"]["values"]["BiT-HyRL"]
                rank = s["gcc"]["ranks"]["BiT-HyRL"]
                cells.append(format_cell(val, rank))
                ranks.append(rank)
            else:
                cells.append("-")
        avg_rank = sum(ranks) / len(ranks) if ranks else float("nan")
        avg_str = f"{avg_rank:.2f}" if ranks else "-"
        print(f"{attack:<12}" + "".join(f"{c:>14}" for c in cells) + f"{avg_str:>10}")

    # 输出每个攻击下 BiT-HyRL 各指标的平均排名
    print("\n" + "=" * 100)
    print("BiT-HyRL 各指标平均排名（按攻击方式）")
    print("=" * 100)
    header2 = f"{'Attack':<12}" + "".join(f"{m.upper():>10}" for m in METRICS) + f"{'Overall':>10}"
    print(header2)
    print("-" * len(header2))

    for attack in attacks:
        metric_ranks = {m: [] for m in METRICS}
        for ds in datasets:
            s = all_data[attack].get(ds)
            if not s:
                continue
            for metric in METRICS:
                if "BiT-HyRL" in s[metric]["ranks"]:
                    metric_ranks[metric].append(s[metric]["ranks"]["BiT-HyRL"])
        cells = []
        all_ranks = []
        for metric in METRICS:
            if metric_ranks[metric]:
                avg = sum(metric_ranks[metric]) / len(metric_ranks[metric])
                cells.append(f"{avg:.2f}")
                all_ranks.extend(metric_ranks[metric])
            else:
                cells.append("-")
        overall = f"{sum(all_ranks)/len(all_ranks):.2f}" if all_ranks else "-"
        print(f"{attack:<12}" + "".join(f"{c:>10}" for c in cells) + f"{overall:>10}")

    # 输出每个攻击下的完整 GCC 排名表（仅 BiT-HyRL vs 前两名）
    print("\n" + "=" * 100)
    print("各攻击方式下 GCC Area 第一名方法统计")
    print("=" * 100)
    for attack in attacks:
        first_counts = defaultdict(int)
        for ds in datasets:
            s = all_data[attack].get(ds)
            if not s or "gcc" not in s:
                continue
            ranks = s["gcc"]["ranks"]
            if ranks:
                first_method = min(ranks, key=ranks.get)
                first_counts[first_method] += 1
        print(f"\n{attack}:")
        for method, count in sorted(first_counts.items(), key=lambda x: -x[1]):
            print(f"  {method:<12}: {count}/{len(datasets)} 次第一")


if __name__ == "__main__":
    main()
