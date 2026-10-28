# -*- coding: utf-8 -*-
"""统计口径复核：解释 Table 5 与 Table 11 的差异，并给出配对置信区间。

为什么需要这个脚本
------------------
审稿意见 Q4 指出：Table 5 的九个拓扑均值再平均，得到的优势约 0.01456 / 0.01611，
而 Table 11 报的是 0.0091 / 0.0108。两个数字都"对"，只是**聚合单位不同**：

* 口径 A（等权拓扑均值，Table 5 的做法）：先在每个 (dataset, attack) 单元内对 run
  求均值，再对单元取算术平均。每个拓扑/攻击权重相同，与 run 数无关。
* 口径 B（池化，Table 11 若为此种）：把所有 run 直接扔进一个池子求均值。run 数多的
  单元权重大，因此与口径 A 不同。
* 口径 C（配对，统计上最该用）：以 (dataset, attack) 为配对单位，算逐单元差值再平均。
  它消掉了拓扑难度带来的公共方差，标准误最小。

三者数值一般互不相等。本脚本同时输出 A/B/C 三种，并给出以单元为聚类单位的
bootstrap 置信区间与 Wilcoxon 符号秩检验，便于在正文里一次讲清"差异来自口径"。

输入
----
默认扫描 ``results/*/*/*/comprehensive_metrics.csv``（主流程输出结构：
``results/{dataset}/{attack}/{exp}/comprehensive_metrics.csv``，一行一个方法）。

也支持直接给一个 tidy CSV：需要列 ``dataset,attack,method,value``。

用法
----
    python scripts/statistical_analysis.py                          # 扫描 results/
    python scripts/statistical_analysis.py --metric "Robustness (R - degree)"
    python scripts/statistical_analysis.py --pair BiT-HyRL UNITY+RL
    python scripts/statistical_analysis.py --demo                   # 口径差异演示
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "stats"


# ---------------------------------------------------------------------------
# 读数据
# ---------------------------------------------------------------------------

def load_from_results(results_dir: Path, metric: str) -> pd.DataFrame:
    """扫描 results/{dataset}/{attack}/{exp}/comprehensive_metrics.csv。"""
    rows = []
    pattern = str(results_dir / "*" / "*" / "*" / "comprehensive_metrics.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame(columns=["dataset", "attack", "exp", "method", "value"])
    for f in files:
        parts = Path(f).parts
        dataset, attack, exp = parts[-4], parts[-3], parts[-2]
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            print(f"[warn] 读取失败 {f}: {exc}")
            continue
        if metric not in df.columns:
            # 允许按前缀匹配（攻击名会出现在列名里）
            cand = [c for c in df.columns if c.startswith(metric.split(" - ")[0])]
            if not cand:
                continue
            col = cand[0]
        else:
            col = metric
        for _, r in df.iterrows():
            try:
                v = float(r[col])
            except Exception:
                continue
            rows.append({"dataset": dataset, "attack": attack, "exp": exp,
                         "method": str(r["Method"]), "value": v})
    return pd.DataFrame(rows)


def load_tidy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"dataset", "attack", "method", "value"}
    if not need.issubset(df.columns):
        raise ValueError(f"tidy CSV 需要列 {need}，实际 {set(df.columns)}")
    if "exp" not in df.columns:
        df["exp"] = "single"
    return df[["dataset", "attack", "exp", "method", "value"]]


# ---------------------------------------------------------------------------
# 三种聚合口径
# ---------------------------------------------------------------------------

def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    """每个 (dataset, attack, method) 一个值：单元内 run 的算术平均。"""
    return (df.groupby(["dataset", "attack", "method"], as_index=False)["value"]
              .agg(cell_value="mean", n_runs="size"))


def aggregate_three_ways(cm: pd.DataFrame, m1: str, m2: str) -> Dict[str, float]:
    """对方法对 (m1, m2) 计算 A/B/C 三种口径的差值。"""
    piv = cm.pivot_table(index=["dataset", "attack"], columns="method",
                         values="cell_value", aggfunc="mean")
    if m1 not in piv.columns or m2 not in piv.columns:
        raise KeyError(f"缺少方法 {m1} 或 {m2}，现有 {list(piv.columns)}")
    sub = piv[[m1, m2]].dropna()
    if sub.empty:
        raise ValueError("没有同时包含两个方法的单元")

    # 口径 A：等权拓扑均值
    a = float(sub[m1].mean() - sub[m2].mean())
    # 口径 C：配对差值（同一单元内相减，再平均）
    diff = sub[m1] - sub[m2]
    c = float(diff.mean())
    # 口径 B：池化（按 run 数加权）
    n = cm[cm.method.isin([m1, m2])].groupby(["dataset", "attack", "method"])["n_runs"].first()
    w = n.reset_index().pivot_table(index=["dataset", "attack"], columns="method", values="n_runs")
    w = w.reindex(sub.index).fillna(1.0)
    b = float(np.average(sub[m1], weights=w[m1]) - np.average(sub[m2], weights=w[m2]))
    return {
        "n_cells": int(len(sub)),
        "mean_m1": float(sub[m1].mean()),
        "mean_m2": float(sub[m2].mean()),
        "delta_A_equal_weight": a,
        "delta_B_pooled": b,
        "delta_C_paired": c,
        "paired_std": float(diff.std(ddof=1)) if len(diff) > 1 else np.nan,
        "paired_se": float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan,
    }


# ---------------------------------------------------------------------------
# 聚类 bootstrap + Wilcoxon
# ---------------------------------------------------------------------------

def cluster_bootstrap(diff: np.ndarray, n_boot: int = 10000, seed: int = 0) -> Tuple[float, float]:
    """以 (dataset, attack) 单元为重采样单位。拓扑是天然的聚类，不能按 run 重采样。"""
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon(diff: np.ndarray) -> Tuple[float, float]:
    try:
        from scipy.stats import wilcoxon as _w
        if np.allclose(diff, 0):
            return (np.nan, 1.0)
        stat, p = _w(diff)
        return float(stat), float(p)
    except Exception:
        return (np.nan, np.nan)


# ---------------------------------------------------------------------------
# 演示：同一批数据在三种口径下给出不同差值
# ---------------------------------------------------------------------------

def demo() -> None:
    """构造一个"run 数不均 + 拓扑难度不同"的合成例子，展示口径差异。

    这正是 Table 5（0.01456）与 Table 11（0.0091）之间可能出现落差的原因：
    若某几个拓扑的 run 数更多、且这些拓扑上两方法差距更小，池化口径会被拉低。
    """
    rng = np.random.default_rng(7)
    rows = []
    datasets = [f"T{i}" for i in range(9)]
    # 前 4 个拓扑：run 多、差距小；后 5 个：run 少、差距大
    for i, ds in enumerate(datasets):
        n_runs = 6 if i < 4 else 3
        base = 0.30 + 0.02 * i
        gap = 0.006 if i < 4 else 0.021
        for r in range(n_runs):
            rows.append({"dataset": ds, "attack": "degree", "exp": f"b{r}",
                         "method": "BiT-HyRL", "value": base + gap + rng.normal(0, 0.002)})
            rows.append({"dataset": ds, "attack": "degree", "exp": f"b{r}",
                         "method": "UNITY+RL", "value": base + rng.normal(0, 0.002)})
    df = pd.DataFrame(rows)
    cm = cell_means(df)
    res = aggregate_three_ways(cm, "BiT-HyRL", "UNITY+RL")
    print("=== 口径演示（同一批数据）===")
    print(f"单元数 {res['n_cells']}，BiT-HyRL 均值 {res['mean_m1']:.5f}，"
          f"UNITY+RL 均值 {res['mean_m2']:.5f}")
    print(f"  A 等权拓扑均值差 : {res['delta_A_equal_weight']:+.5f}   ← Table 5 类口径")
    print(f"  B 池化（按run数）: {res['delta_B_pooled']:+.5f}   ← Table 11 若为此口径")
    print(f"  C 配对差值均值   : {res['delta_C_paired']:+.5f}   ← 统计上应报告的口径")
    print("\n结论：三个数都不是「错」的，差别完全来自聚合单位。论文必须写明用的是哪一个。")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--tidy", default="")
    ap.add_argument("--metric", default="Robustness (R - degree)")
    ap.add_argument("--pair", nargs=2, default=["BiT-HyRL", "UNITY+RL"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    if args.tidy:
        df = load_tidy(Path(args.tidy))
    else:
        df = load_from_results(Path(args.results_dir), args.metric)

    if df.empty:
        print(f"没有找到数据（metric={args.metric}）。")
        print("先跑主实验，或改用 --tidy 提供 tidy CSV，或 --demo 看口径演示。")
        return

    cm = cell_means(df)
    m1, m2 = args.pair
    res = aggregate_three_ways(cm, m1, m2)

    piv = cm.pivot_table(index=["dataset", "attack"], columns="method", values="cell_value")
    diff = (piv[m1] - piv[m2]).dropna().to_numpy()
    lo, hi = cluster_bootstrap(diff, args.n_boot)
    stat, p = wilcoxon(diff)

    res.update({
        "ci95_low": lo, "ci95_high": hi,
        "wilcoxon_stat": stat, "wilcoxon_p": p,
        f"better_in_cells": int((diff > 0).sum()),
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res]).to_csv(OUT_DIR / "pairwise_summary.csv", index=False, encoding="utf-8-sig")
    piv.to_csv(OUT_DIR / "cell_values.csv", encoding="utf-8-sig")

    print(f"指标: {args.metric}")
    print(f"配对: {m1} vs {m2}，单元数 {res['n_cells']}")
    print(f"  A 等权拓扑均值差 : {res['delta_A_equal_weight']:+.5f}")
    print(f"  B 池化（按run数）: {res['delta_B_pooled']:+.5f}")
    print(f"  C 配对差值均值   : {res['delta_C_paired']:+.5f} "
          f"(95% CI [{lo:+.5f}, {hi:+.5f}])")
    print(f"  优于对方的单元数 : {int((diff > 0).sum())}/{len(diff)}")
    print(f"  Wilcoxon p       : {p:.4g}")
    print(f"\n写入 {OUT_DIR}/pairwise_summary.csv, cell_values.csv")


if __name__ == "__main__":
    main()
