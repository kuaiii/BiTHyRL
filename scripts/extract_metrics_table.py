#!/usr/bin/env python3
"""
从 results 下各数据集的 wgcc 最新结果中提取 CCE、CSA、GCC、WCP 指标，
按指定格式（Method × Network）保存为 CSV 表格。
"""

import csv
import os
from pathlib import Path
from typing import Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TESTDATA_DIR = PROJECT_ROOT / "dataset" / "testdata"
OUTPUT_DIR = PROJECT_ROOT / "results" / "metrics_tables"

# 数据集名称（与 testdata 中 .gml 文件名一致，不含扩展名）
# 列顺序：BA-50, BA-100, BA-200, BA-300, BA-400, BA-500, Chinanet, Colt, Cogentco, GtsCe, UsCarrier
DATASETS = [
    "BA-50", "BA-100", "BA-200", "BA-300", "BA-400", "BA-500",
    "Chinanet", "Colt", "Cogentco", "GtsCe", "UsCarrier",
]

# 四种指标
METRICS = ["cce", "csa", "gcc", "wcp"]

# CSV 中的方法名 -> 表格行名映射
METHOD_MAP = {
    "Baseline": "Baseline",
    "Onion+RL": "Onion",
    "GA+RL": "GA-RL",
    "GA+RCP": "GA+RCP",
    "ROMEN+RL": "ROMEN",
    "UNITY+RL": "UNITY",
    "FRED-ABL": "FRED-ABL",
    "QDLM": "QDLM",
    "BiT-HyRL": "BiT-HyRL",
}

# 表格行顺序
ROW_ORDER = [
    "Baseline", "Onion", "GA-RL", "GA+RCP", "ROMEN", "UNITY",
    "FRED-ABL", "QDLM", "BiT-HyRL",
]


def get_testdata_datasets():
    """从 testdata 目录获取所有数据集名称（不含 .gml）"""
    names = set()
    for p in TESTDATA_DIR.iterdir():
        if p.suffix.lower() == ".gml":
            names.add(p.stem)
    return sorted(names, key=lambda x: (
        0 if x.startswith("BA-") else 1,
        x,
    ))


def find_latest_wgcc_dir(dataset: str) -> Optional[Path]:
    """返回该数据集 results/{dataset}/wgcc 下数字最大的子文件夹路径"""
    wgcc_base = RESULTS_DIR / dataset / "wgcc"
    if not wgcc_base.exists():
        return None
    subdirs = [d for d in wgcc_base.iterdir() if d.is_dir() and d.name.isdigit()]
    if not subdirs:
        return None
    latest = max(subdirs, key=lambda d: int(d.name))
    return latest


def read_r_values_csv(csv_path: Path) -> dict[str, str]:
    """
    读取 r_values_wgcc_*.csv，取前 3 列 Method, Mean_R, Std_R，
    返回 {表格行名: "mean±std"} 的字典
    """
    if not csv_path.exists():
        return {}
    data = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method_raw = row.get("Method", "").strip()
            if not method_raw:
                continue
            mean_r = row.get("Mean_R", "").strip()
            std_r = row.get("Std_R", "").strip()
            display_name = METHOD_MAP.get(method_raw, method_raw)
            try:
                mean = float(mean_r)
                std = float(std_r)
                data[display_name] = f"{mean:.4f}±{std:.4f}"
            except (ValueError, TypeError):
                data[display_name] = ""
    return data


def build_metric_table(metric: str) -> list[list[str]]:
    """
    构建某一指标（CCE/CSA/GCC/WCP）的表格：
    第一列：Method
    其余列：各数据集对应列
    """
    header = ["Method"] + DATASETS
    rows = []
    # 按 ROW_ORDER 填充
    for method in ROW_ORDER:
        row_data = [method]
        for ds in DATASETS:
            latest_wgcc = find_latest_wgcc_dir(ds)
            if latest_wgcc is None:
                row_data.append("")
                continue
            csv_path = latest_wgcc / metric / "collapse_R" / f"r_values_wgcc_{metric}.csv"
            data = read_r_values_csv(csv_path)
            val = data.get(method, "")
            row_data.append(val)
        rows.append(row_data)
    return [header] + rows


def save_table(rows: list[list[str]], output_path: Path):
    """保存表格为 CSV"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved: {output_path}")


def main():
    # 若 testdata 存在，优先用其中的数据集顺序（与用户指定一致）
    testdata_names = get_testdata_datasets()
    if testdata_names:
        global DATASETS
        # 保持用户指定的顺序，仅过滤出 testdata 中存在的
        ordered = [d for d in DATASETS if d in testdata_names]
        missing = [d for d in DATASETS if d not in testdata_names]
        if missing:
            ordered.extend(missing)
        DATASETS = ordered

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metric_names = {
        "cce": "CCE",
        "csa": "CSA",
        "gcc": "GCC",
        "wcp": "WCP",
    }

    for metric in METRICS:
        table = build_metric_table(metric)
        name = metric_names.get(metric, metric.upper())
        out_path = OUTPUT_DIR / f"metrics_table_{metric.upper()}.csv"
        save_table(table, out_path)

    print(f"\nAll tables saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
