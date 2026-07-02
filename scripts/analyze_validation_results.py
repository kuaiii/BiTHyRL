#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总 main.py 生成的 degree/wgcc 实验结果，输出 BiT-HyRL 与基线的对比表格与排名。
"""
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

METRICS = ["gcc", "csa", "cce", "wcp"]
METHODS = ["Baseline", "GA+RL", "GA+RCP", "Onion+RL", "ROMEN+RL", "UNITY+RL", "FRED-ABL", "QDLM", "BiT-HyRL"]


def read_area_csv(csv_path):
    """读取 area_values_*_{metric}.csv，返回 {method: mean_area}。"""
    data = {}
    if not csv_path.exists():
        return data
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("Method", "").strip()
            if method not in METHODS:
                continue
            try:
                mean = float(row.get("Mean_Area", row.get("Mean_R", "0")))
            except Exception:
                continue
            data[method] = mean
    return data


def read_comprehensive_csv(csv_path):
    """读取 comprehensive_metrics.csv，返回 {method: {metric: value}}。"""
    data = defaultdict(dict)
    if not csv_path.exists():
        return data
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for row in reader:
            method = row.get("Method", "").strip()
            if method not in METHODS:
                continue
            for col in cols:
                if col in ("Method", "Nodes", "Edges", "Top-10 Degrees"):
                    continue
                # Robustness R 列名随攻击方式变化，统一提取
                if col.startswith("Robustness (R -"):
                    try:
                        data[method]["Robustness R"] = float(row[col])
                    except Exception:
                        pass
                else:
                    try:
                        data[method][col] = float(row[col])
                    except Exception:
                        pass
    return data


def rank_methods(data, higher_is_better=True):
    """根据数值排序，返回 {method: rank}，从 1 开始。"""
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=higher_is_better)
    ranks = {}
    for rank, (method, _) in enumerate(sorted_items, 1):
        ranks[method] = rank
    return ranks


def summarize_dataset(dataset, attack="degree", exp_id=None):
    """汇总单个数据集某次实验。exp_id 为 None 时取最新。"""
    base = RESULTS_DIR / dataset / attack
    if not base.exists():
        return None
    if exp_id is None:
        subdirs = [d for d in base.iterdir() if d.is_dir() and d.name.isdigit()]
        if not subdirs:
            return None
        # 选择包含完整 area CSV 的最新实验目录（避免选中正在运行的未完成目录）
        valid = []
        for d in sorted(subdirs, key=lambda x: int(x.name), reverse=True):
            complete = True
            for metric in METRICS:
                if not (d / metric / "area" / f"area_values_{attack}_{metric}.csv").exists():
                    complete = False
                    break
            if complete:
                valid.append(d)
        if not valid:
            return None
        exp_dir = valid[0]
    else:
        exp_dir = base / str(exp_id)
        if not exp_dir.exists():
            return None

    summary = {"dataset": dataset, "attack": attack, "exp_id": exp_id}
    area_data = {}
    for metric in METRICS:
        csv_path = exp_dir / metric / "area" / f"area_values_{attack}_{metric}.csv"
        area_data[metric] = read_area_csv(csv_path)
    summary["area"] = area_data

    comp = read_comprehensive_csv(exp_dir / "comprehensive_metrics.csv")
    summary["comprehensive"] = comp

    # 排名
    ranks = {}
    for metric in METRICS:
        ranks[metric] = rank_methods(area_data.get(metric, {}))
    summary["ranks"] = ranks
    return summary


def print_summary_table(summaries, metric="gcc"):
    """打印某指标下各方法在各数据集上的 area 与排名。"""
    datasets = [s["dataset"] for s in summaries]
    print(f"\n=== {metric.upper()} Area (higher is better) ===")
    header = f"{'Method':<14}" + "".join(f"{d:>12}" for d in datasets) + f"{'Avg Rank':>10}"
    print(header)
    print("-" * len(header))

    method_rows = {m: [] for m in METHODS}
    for s in summaries:
        data = s["area"].get(metric, {})
        ranks = s["ranks"].get(metric, {})
        for m in METHODS:
            val = data.get(m, float("nan"))
            rank = ranks.get(m, "-")
            if val != val:
                method_rows[m].append(("-", "-"))
            else:
                method_rows[m].append((f"{val:.4f}", f"#{rank}"))

    avg_ranks = {}
    for m in METHODS:
        ranks = []
        for s in summaries:
            r = s["ranks"].get(metric, {}).get(m)
            if r is not None:
                ranks.append(r)
        avg_ranks[m] = sum(ranks) / len(ranks) if ranks else float("nan")

    # 按平均排名排序
    for m in sorted(METHODS, key=lambda x: avg_ranks.get(x, 999)):
        cells = method_rows[m]
        row_str = f"{m:<14}" + "".join(f"{v:>6}({r:>3})" for v, r in cells)
        avg = avg_ranks[m]
        if avg != avg:
            row_str += f"{'-':>10}"
        else:
            row_str += f"{avg:>10.2f}"
        print(row_str)


def print_robustness_table(summaries):
    """打印 Robustness (R) 指标。"""
    datasets = [s["dataset"] for s in summaries]
    print("\n=== Robustness R (higher is better) ===")
    header = f"{'Method':<14}" + "".join(f"{d:>12}" for d in datasets) + f"{'Avg Rank':>10}"
    print(header)
    print("-" * len(header))

    method_vals = {m: [] for m in METHODS}
    for s in summaries:
        comp = s["comprehensive"]
        data = {m: comp.get(m, {}).get("Robustness R", float("nan")) for m in METHODS}
        ranks = rank_methods(data)
        for m in METHODS:
            val = data[m]
            rank = ranks.get(m, "-")
            method_vals[m].append((val, rank))

    avg_ranks = {}
    for m in METHODS:
        ranks = [s["comprehensive"].get(m, {}).get("Robustness (R - degree)", float("nan")) for s in summaries]
        valid_ranks = []
        for s in summaries:
            comp = s["comprehensive"]
            val = comp.get(m, {}).get("Robustness R", float("nan"))
            if val == val:
                valid_ranks.append(val)
        # 计算平均排名
        ranks_list = []
        for s in summaries:
            comp = s["comprehensive"]
            data = {mm: comp.get(mm, {}).get("Robustness (R - degree)", float("nan")) for mm in METHODS}
            ranks = rank_methods(data)
            r = ranks.get(m)
            if r is not None:
                ranks_list.append(r)
        avg_ranks[m] = sum(ranks_list) / len(ranks_list) if ranks_list else float("nan")

    for m in sorted(METHODS, key=lambda x: avg_ranks.get(x, 999)):
        cells = method_vals[m]
        row_str = f"{m:<14}" + "".join(f"{v:>6.4f}({r:>3})" if v == v else f"{'-':>6}({'-':>3})" for v, r in cells)
        avg = avg_ranks[m]
        if avg != avg:
            row_str += f"{'-':>10}"
        else:
            row_str += f"{avg:>10.2f}"
        print(row_str)


def main():
    attack = sys.argv[1] if len(sys.argv) > 1 else "degree"
    datasets = sys.argv[2:] if len(sys.argv) > 2 else ["BA-50", "BA-100", "BA-200", "BA-300", "BA-400", "BA-500", "Chinanet", "Colt", "Cogentco", "GtsCe", "UsCarrier"]

    summaries = []
    for ds in datasets:
        s = summarize_dataset(ds, attack=attack)
        if s:
            summaries.append(s)
        else:
            print(f"[WARN] No results found for {ds}/{attack}", file=sys.stderr)

    for metric in METRICS:
        print_summary_table(summaries, metric=metric)
    print_robustness_table(summaries)


if __name__ == "__main__":
    main()
