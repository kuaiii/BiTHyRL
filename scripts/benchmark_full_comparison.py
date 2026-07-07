#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大规模对比实验脚本。

在多个数据集、多种攻击模式下重复运行 main.py，收集：
- comprehensive_metrics.csv 中的 R、CSA、H_C、WCP
- execution_times.csv 中的各阶段耗时
- area_values_*.csv 中每次重复的 R 值（用于计算 mean/std）

输出：
- results/benchmark_summary/<timestamp>/summary_R.csv
- results/benchmark_summary/<timestamp>/summary_CSA.csv
- results/benchmark_summary/<timestamp>/summary_WCP.csv
- results/benchmark_summary/<timestamp>/summary_times.csv
- results/benchmark_summary/<timestamp>/raw_results.json

用法：
    python scripts/benchmark_full_comparison.py
    python scripts/benchmark_full_comparison.py --datasets BA-100,Colt --attacks random,degree --repeats 10

可通过命令行参数或修改 DEFAULT_CONFIG 自定义数据集、攻击模式和重复次数。
"""

import os
import sys
import csv
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# --------------------------- 用户可配置区域 ---------------------------
DEFAULT_CONFIG = {
    "datasets": [
        "BA-100",
        "BA-300",
        # "BA-500",
        "Colt",
        "GtsCe",
        # "Chinanet",
        # "Cogentco",
    ],
    "attacks": [
        "random",
        "degree",
        "betweenness",
        # "adaptive_bridge",
        # "wgcc",
    ],
    "repeats": 10,
    "rate": 0.1,
    "use_gnn_controller": False,   # True 时 BiT-HyRL 使用 GNN 控制器
    "include_base": True,          # 是否包含 BiT-HyRL-base
    "skip_existing": True,         # 若结果目录已存在且非空则跳过
}
# --------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="BiT-HyRL 大规模对比 benchmark")
    parser.add_argument("--datasets", type=str, default=None,
                        help="逗号分隔的数据集列表，默认使用 DEFAULT_CONFIG['datasets']")
    parser.add_argument("--attacks", type=str, default=None,
                        help="逗号分隔的攻击模式列表，默认使用 DEFAULT_CONFIG['attacks']")
    parser.add_argument("--repeats", type=int, default=None, help="重复次数，默认 10")
    parser.add_argument("--rate", type=float, default=None, help="控制器比例，默认 0.1")
    parser.add_argument("--gnn-controller", action="store_true", help="BiT-HyRL 使用 GNN 控制器")
    parser.add_argument("--no-base", action="store_true", help="不包含 BiT-HyRL-base")
    parser.add_argument("--no-skip", action="store_true", help="不跳过已有结果，强制重新运行")
    parser.add_argument("--dry-run", action="store_true", help="只解析已有结果，不运行新实验")
    return parser.parse_args()


def build_config(args):
    cfg = DEFAULT_CONFIG.copy()
    if args.datasets:
        cfg["datasets"] = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.attacks:
        cfg["attacks"] = [a.strip() for a in args.attacks.split(",") if a.strip()]
    if args.repeats is not None:
        cfg["repeats"] = args.repeats
    if args.rate is not None:
        cfg["rate"] = args.rate
    if args.gnn_controller:
        cfg["use_gnn_controller"] = True
    if args.no_base:
        cfg["include_base"] = False
    if args.no_skip:
        cfg["skip_existing"] = False
    return cfg


METHOD_ORDER = [
    "Baseline", "GA+RCP", "GA+RL", "Onion+RL", "ROMEN+RL",
    "UNITY+RL", "FRED-ABL", "QDLM", "BiT-HyRL", "BiT-HyRL-base", "Unified-PPO"
]

METRIC_FILES = {
    "R": "Robustness (R - {attack})",
    "CSA": "CSA",
    "H_C": "Control Entropy (H_C)",
    "WCP": "Weighted Control Potential (WCP)",
}


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_experiment(dataset, attack, cfg):
    """调用 main.py 运行一次 batch=REPEATS 的实验。"""
    cmd = [
        sys.executable, "main.py",
        "--dataset", dataset,
        "--attack", attack,
        "--batch", str(cfg["repeats"]),
        "--rate", str(cfg["rate"]),
    ]
    if cfg["include_base"]:
        cmd.append("--unified-ppo-include-base")
    if cfg["use_gnn_controller"]:
        cmd.append("--unified-ppo-use-gnn-controller")

    env = os.environ.copy()
    env["ALLTEST_QUIET"] = "1"

    log(f"RUN: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        log(f"ERROR: 运行失败，耗时 {elapsed:.1f}s")
        log(result.stderr[-2000:])
        return None
    log(f"DONE: 耗时 {elapsed:.1f}s")
    return result


def find_run_ids(base_dir, expected_count=None):
    """获取某配置下所有数字实验目录。"""
    base = Path(base_dir)
    if not base.exists():
        return []
    ids = []
    for d in base.iterdir():
        if d.is_dir() and d.name.isdigit():
            ids.append(int(d.name))
    ids.sort()
    if expected_count:
        # 取最新的 expected_count 个
        ids = ids[-expected_count:]
    return ids


def run_has_repeats(exp_dir, attack, expected_repeats):
    """检查某实验目录是否已有 expected_repeats 次重复的完整结果。"""
    area_file = exp_dir / "gcc" / "area" / f"area_values_{attack}_gcc.csv"
    if not area_file.exists():
        return False
    try:
        with open(area_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                batch_cols = [k for k in row if k.startswith("Batch_")]
                if len(batch_cols) >= expected_repeats:
                    return True
                return False
    except Exception:
        return False
    return False


def read_comprehensive(csv_path, attack):
    """读取 comprehensive_metrics.csv，返回 {method: {metric: value}}。"""
    if not csv_path.exists():
        return {}
    results = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header_R = METRIC_FILES["R"].format(attack=attack)
        for row in reader:
            method = row["Method"]
            try:
                results[method] = {
                    "R": float(row[header_R]) if header_R in row else float("nan"),
                    "CSA": float(row["CSA"]) if row.get("CSA") else float("nan"),
                    "H_C": float(row["Control Entropy (H_C)"]) if row.get("Control Entropy (H_C)") else float("nan"),
                    "WCP": float(row["Weighted Control Potential (WCP)"]) if row.get("Weighted Control Potential (WCP)") else float("nan"),
                    "Nodes": int(float(row["Nodes"])) if row.get("Nodes") else None,
                    "Edges": int(float(row["Edges"])) if row.get("Edges") else None,
                }
            except Exception as e:
                log(f"WARN: 解析 {csv_path} 行失败: {e}")
    return results


def read_execution_times(csv_path):
    """读取 execution_times.csv，返回 {stage: {'mean': ..., 'std': ...}}。"""
    if not csv_path.exists():
        return {}
    times = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = row.get("Stage", "")
            try:
                times[stage] = {
                    "mean": float(row["Mean Time (s)"]),
                    "std": float(row.get("Std Time (s)", 0.0)) if row.get("Std Time (s)") else 0.0,
                }
            except Exception:
                pass
    return times


def read_area_values(csv_path):
    """读取 area_values_*.csv，返回 {method: [r_val_batch0, ...]}。"""
    if not csv_path.exists():
        return {}
    results = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["Method"]
            vals = []
            for k in row:
                if k.startswith("Batch_"):
                    try:
                        vals.append(float(row[k]))
                    except Exception:
                        pass
            results[method] = vals
    return results


def aggregate_values(values_list):
    """计算列表的均值、标准差。"""
    if not values_list:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}
    n = len(values_list)
    mean = sum(values_list) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in values_list) / (n - 1)
        std = var ** 0.5
    else:
        std = 0.0
    return {"mean": mean, "std": std, "count": n}


def main():
    args = parse_args()
    cfg = build_config(args)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / "benchmark_summary" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"输出目录: {out_dir}")
    log(f"配置: {cfg}")

    # 原始数据收集
    raw = defaultdict(lambda: defaultdict(dict))
    summary = defaultdict(lambda: defaultdict(list))

    total = len(cfg["datasets"]) * len(cfg["attacks"])
    done = 0

    for dataset in cfg["datasets"]:
        for attack in cfg["attacks"]:
            done += 1
            key = f"{dataset}/{attack}"
            base_dir = Path("results") / dataset / attack

            # 检查是否已有足够重复次数的结果（用于断点续跑）
            run_id = None
            if cfg["skip_existing"]:
                for rid in reversed(find_run_ids(base_dir)):
                    cand = base_dir / str(rid)
                    if (cand / "comprehensive_metrics.csv").exists() and run_has_repeats(cand, attack, cfg["repeats"]):
                        log(f"SKIP ({done}/{total}): {key} 已存在 run_id={rid} 且满足 {cfg['repeats']} 次重复")
                        run_id = rid
                        break

            if run_id is None:
                if args.dry_run:
                    log(f"SKIP ({done}/{total}): {key} 无结果且 dry-run 模式")
                    continue
                res = run_experiment(dataset, attack, cfg)
                if res is None:
                    continue
                # 运行后获取最新 run_id
                existing = find_run_ids(base_dir)
                if not existing:
                    log(f"ERROR: 未找到 {key} 的结果目录")
                    continue
                run_id = existing[-1]

            exp_dir = base_dir / str(run_id)
            comp = read_comprehensive(exp_dir / "comprehensive_metrics.csv", attack)
            times = read_execution_times(exp_dir / "execution_times.csv")
            area_gcc = read_area_values(exp_dir / "gcc" / "area" / f"area_values_{attack}_gcc.csv")

            raw[dataset][attack] = {
                "run_id": run_id,
                "comprehensive": comp,
                "times": times,
                "area_gcc": area_gcc,
            }

            # 汇总每个 method 的指标
            for method in comp:
                # R 使用 area_values 中的每次重复值，可计算标准差
                if method in area_gcc and area_gcc[method]:
                    for val in area_gcc[method]:
                        summary[(dataset, attack, "R")][method].append(val)
                else:
                    summary[(dataset, attack, "R")][method].append(comp[method].get("R", float("nan")))
                # CSA/H_C/WCP 仅有 comprehensive 平均值
                for metric in ["CSA", "H_C", "WCP"]:
                    summary[(dataset, attack, metric)][method].append(comp[method].get(metric, float("nan")))
                # 拓扑属性只取一次
                summary[(dataset, attack, "Nodes")][method].append(comp[method].get("Nodes"))
                summary[(dataset, attack, "Edges")][method].append(comp[method].get("Edges"))

            # 耗时汇总（execution_times.csv 内部已按 repeats 计算 mean/std）
            for stage, t in times.items():
                summary[(dataset, attack, f"time_{stage}")]["_global_mean"].append(t["mean"])
                summary[(dataset, attack, f"time_{stage}")]["_global_std"].append(t["std"])

    # 保存原始 JSON
    with open(out_dir / "raw_results.json", "w", encoding="utf-8") as f:
        # defaultdict 不能 json dump，先转换
        raw_plain = {k: dict(v) for k, v in raw.items()}
        json.dump(raw_plain, f, ensure_ascii=False, indent=2)

    # 生成汇总表格
    write_metric_table(summary, cfg, out_dir, "R")
    write_metric_table(summary, cfg, out_dir, "CSA")
    write_metric_table(summary, cfg, out_dir, "H_C")
    write_metric_table(summary, cfg, out_dir, "WCP")
    write_time_table(summary, cfg, out_dir)
    write_latex_table(summary, cfg, out_dir, "R")

    log(f"汇总完成，结果保存至 {out_dir}")


def write_metric_table(summary, cfg, out_dir, metric):
    """为每个指标写一个 CSV，行是 (dataset, attack)，列是方法。"""
    rows = []
    header = ["Dataset", "Attack"] + METHOD_ORDER
    rows.append(header)

    for dataset in cfg["datasets"]:
        for attack in cfg["attacks"]:
            row = [dataset, attack]
            for method in METHOD_ORDER:
                vals = summary[(dataset, attack, metric)].get(method, [])
                agg = aggregate_values(vals)
                cell = f"{agg['mean']:.4f}±{agg['std']:.4f}" if vals else "N/A"
                row.append(cell)
            rows.append(row)

    path = out_dir / f"summary_{metric}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    log(f"已保存 {path}")


def write_time_table(summary, cfg, out_dir):
    """输出各阶段耗时表格。"""
    # 收集所有 stage
    stages = set()
    for dataset in cfg["datasets"]:
        for attack in cfg["attacks"]:
            for key in summary:
                if key[0] == dataset and key[1] == attack and key[2].startswith("time_"):
                    stages.add(key[2])
    stages = sorted(stages)

    rows = []
    header = ["Dataset", "Attack"] + stages
    rows.append(header)

    for dataset in cfg["datasets"]:
        for attack in cfg["attacks"]:
            row = [dataset, attack]
            for stage in stages:
                means = summary[(dataset, attack, stage)].get("_global_mean", [])
                stds = summary[(dataset, attack, stage)].get("_global_std", [])
                if means:
                    mean_val = sum(means) / len(means)
                    std_val = sum(stds) / len(stds) if stds else 0.0
                    cell = f"{mean_val:.2f}±{std_val:.2f}"
                else:
                    cell = "N/A"
                row.append(cell)
            rows.append(row)

    path = out_dir / "summary_times.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    log(f"已保存 {path}")


def write_latex_table(summary, cfg, out_dir, metric):
    """生成一个简化的 LaTeX 表格（仅 dataset×attack，突出 BiT-HyRL vs ROMEN+RL）。"""
    lines = []
    lines.append("\\begin{tabular}{ll" + "c" * len(METHOD_ORDER) + "}")
    lines.append("\\toprule")
    lines.append("Dataset & Attack & " + " & ".join(METHOD_ORDER) + " \\\\")
    lines.append("\\midrule")

    for dataset in cfg["datasets"]:
        for attack in cfg["attacks"]:
            cells = [dataset, attack]
            for method in METHOD_ORDER:
                vals = summary[(dataset, attack, metric)].get(method, [])
                agg = aggregate_values(vals)
                cell = f"{agg['mean']:.3f}" if vals else "--"
                cells.append(cell)
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    path = out_dir / f"summary_{metric}_latex.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"已保存 {path}")


if __name__ == "__main__":
    main()
