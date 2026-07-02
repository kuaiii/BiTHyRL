# -*- coding: utf-8 -*-
"""
批量运行 main.py 对比 Unified-PPO 与基线方法（真实网络）。
"""
import os
import re
import sys
import glob
import subprocess
import argparse
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset" / "testdata"
RESULTS_DIR = PROJECT_ROOT / "results"


def parse_metrics(csv_path):
    """读取 comprehensive_metrics.csv，返回 {method: {metric: value}}。"""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    out = {}
    for _, row in df.iterrows():
        method = row["Method"]
        out[method] = {col: row[col] for col in df.columns if col != "Method"}
    return out


def run_single(dataset, attack, model_path, B_budget, batch):
    cmd = [
        sys.executable, "main.py",
        "--dataset", dataset,
        "--attack", attack,
        "--batch", str(batch),
        "--unified-ppo-model", model_path,
        "--unified-ppo-B", str(B_budget),
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)


def collect_latest(dataset, attack):
    attack_dir = RESULTS_DIR / dataset / attack
    if not attack_dir.exists():
        return None
    # 找最新的时间戳子目录
    subdirs = [p for p in attack_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
    csv = latest / "comprehensive_metrics.csv"
    return parse_metrics(csv) if csv.exists() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Unified-PPO checkpoint path")
    parser.add_argument("--B-budget", type=int, default=3)
    parser.add_argument("--attacks", default="random,degree,betweenness,pagerank,eigenvector",
                        help="逗号分隔的攻击类型")
    parser.add_argument("--datasets", default=None, help="逗号分隔的数据集名，默认 dataset/testdata 下所有 .gml")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", default="results/unified_ppo_comparison_summary.csv")
    args = parser.parse_args()

    attacks = [a.strip() for a in args.attacks.split(",")]
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = sorted([
            Path(p).stem for p in DATASET_DIR.glob("*.gml")
            if Path(p).stat().st_size > 0
        ])

    rows = []
    for dataset in datasets:
        for attack in attacks:
            print(f"\n{'='*60}\nDataset: {dataset} | Attack: {attack}\n{'='*60}")
            run_single(dataset, attack, args.model_path, args.B_budget, args.batch)
            metrics = collect_latest(dataset, attack)
            if metrics is None:
                print(f"[warn] 未找到 {dataset}/{attack} 的结果")
                continue
            for method, vals in metrics.items():
                row = {"Dataset": dataset, "Attack": attack, "Method": method}
                row.update(vals)
                rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        out_path = PROJECT_ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n汇总结果已保存: {out_path}")
    else:
        print("没有收集到任何结果。")


if __name__ == "__main__":
    main()
