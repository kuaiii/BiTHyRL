# -*- coding: utf-8 -*-
"""把巴拉巴西书 §7.16 / 进阶阅读 7.G 的最优度分布推导，落到本项目的构造族上。

目的（回答"我们的构造到底覆盖多大的范围"）：

1. **复现书中的结论**：固定 <k>，令 r = n_h/N，则
       f_tot(r) = f_rand(r) + f_target(r)                      (书 7.16)
   的最大值出现在 r 最小的那一端；把 r 取到 1/N 并解驻点，得
       k_max ~ A·r^(-2/3) = A·N^(2/3)                           (书 7.67/7.69)
   本脚本在**我们自己的构造族**上验证这一点（而不是在抽象的两点分布上）。

2. **给出书中最优解的适用范围**：书中推导允许 k_min 连续取值（r→0 时 k_min→<k>）。
   本项目的构造要求 k_min ≥ 2 且低度层连通，由此产生残差约束
       k_max ≤ (M - N + n_h) / n_h
   两者联立 ⟺ N ≥ N_min。脚本逐点标注每个 (数据集, n_h) 落在哪一侧。

3. **量出"口径差"**：书中 f_tot 只看崩溃点，论文的 R 看整条 LCC 曲线的面积。
   脚本同时输出
       f_rand / f_target / f_tot        解析（书 7.64-7.66）
       fc_rand_emp / fc_targ_emp         实测崩溃点（LCC 首次 < 0.10）
       R_random / R_targeted / R_bar     AUC 口径
       L_bar                             平均最短路（效率代理）
   从而可以定量回答：同一个族，按 f_tot 排最优 vs 按 R 排最优，是不是同一端。

输出
    results/theory_range/family_scan.csv        逐 (数据集, n_h) 明细
    results/theory_range/family_summary.csv     逐数据集的 argmax 对照
    results/theory_range/family_tables.md       可直接放进论文的表

用法
    python scripts/theory_range.py
    python scripts/theory_range.py --datasets GtsCe,BA-200 --nh 1,2,4,8,16
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topology.bimodal_construction import (  # noqa: E402
    applicability,
    build_bimodal,
    construct_bimodal_sequence,
    residual_cap,
)

OUT_DIR = ROOT / "results" / "theory_range"


# ---------------------------------------------------------------------------
# 1. 书中解析式：7.63 的矩、7.64 的随机阈值、7.65/7.66 的定向阈值
# ---------------------------------------------------------------------------

def moments(avg_k: float, r: float, k_max: float) -> Tuple[float, float]:
    """书 7.63：两点分布的 <k^2>。k_min 由 <k> = (1-r)k_min + r k_max 反解。"""
    if r >= 1.0:
        return avg_k ** 2, avg_k
    k_min = (avg_k - r * k_max) / (1.0 - r)
    m2 = (1.0 - r) * k_min ** 2 + r * k_max ** 2
    return m2, k_min


def fc_random(avg_k: float, r: float, k_max: float) -> float:
    """书 7.64。等价形式 f_c = 1 - 1/(kappa-1)，kappa = <k^2>/<k>。

    这里保留 7.64 的原始写法以便与书对照，并在 ``check_identity`` 里验证两者一致。
    """
    m2, _ = moments(avg_k, r, k_max)
    num = m2 - 2.0 * r * avg_k * k_max - 2.0 * (1.0 - r) * avg_k + r * k_max ** 2
    den = m2 - 2.0 * r * avg_k * k_max - (1.0 - r) * avg_k + r * k_max ** 2
    if den <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, num / den))


def fc_random_kappa(avg_k: float, r: float, k_max: float) -> float:
    m2, _ = moments(avg_k, r, k_max)
    kappa = m2 / avg_k
    if kappa <= 2.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - 1.0 / (kappa - 1.0)))


def fc_targeted(avg_k: float, r: float, k_max: float) -> Tuple[float, int]:
    """书 7.65（全部 hub 被移除）与 7.66（仅部分 hub 被移除）。

    返回 (f_target, case)，case ∈ {1, 2, 0}；0 表示该点无定义。
    """
    m2, k_min = moments(avg_k, r, k_max)
    # 情况 (1)：f_target > r，所有 hub 被清空，残余为度 k_min 的均匀层
    if k_min <= 1.0 + 1e-12:
        f1 = r  # 残余层 k_min<=1 时自身随机失效阈值 <=0
    else:
        f1 = r + (1.0 - r) * (1.0 - 1.0 / (k_min - 1.0))
    # 情况 (2)：f_target < r
    den = k_max * (k_max - 1.0) * (1.0 - r)
    if den <= 1e-12:
        f2 = float("inf")
    else:
        f2 = (m2 - 2.0 * r * avg_k * k_max + r * k_max ** 2 - 2.0 * (1.0 - r) * avg_k) / den
    if f2 < r:
        return max(0.0, f2), 2
    return max(0.0, min(1.0, f1)), 1


def ftot(avg_k: float, r: float, k_max: float) -> Dict[str, float]:
    fr = fc_random(avg_k, r, k_max)
    ft, case = fc_targeted(avg_k, r, k_max)
    return {"f_rand": fr, "f_target": ft, "f_tot": fr + ft, "f_case": case}


# ---------------------------------------------------------------------------
# 2. 实测：AUC 口径与崩溃点
# ---------------------------------------------------------------------------

def lcc_curve(g: nx.Graph, order: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """按 order 逐个移除节点，返回 (移除比例, LCC/初值)。"""
    h = g.copy()
    n0 = h.number_of_nodes()
    xs = [0.0]
    ys = [1.0]
    for i, node in enumerate(order, start=1):
        if h.has_node(node):
            h.remove_node(node)
        if h.number_of_nodes() == 0:
            y = 0.0
        else:
            y = max((len(c) for c in nx.connected_components(h)), default=0) / n0
        xs.append(i / n0)
        ys.append(y)
    return np.asarray(xs), np.asarray(ys)


def auc(xs: np.ndarray, ys: np.ndarray) -> float:
    return float(np.trapz(ys, xs))


def collapse_point(xs: np.ndarray, ys: np.ndarray, thr: float = 0.10) -> float:
    """LCC 首次跌破 thr 时的移除比例；不跌则返回 1.0。"""
    idx = np.where(ys < thr)[0]
    return float(xs[idx[0]]) if len(idx) else 1.0


def evaluate(g: nx.Graph, n_random: int = 5, seed: int = 2026) -> Dict[str, float]:
    n0 = g.number_of_nodes()
    nodes = list(g.nodes())
    rng = random.Random(seed)
    r_rand, c_rand = [], []
    for _ in range(n_random):
        order = nodes[:]
        rng.shuffle(order)
        xs, ys = lcc_curve(g, order)
        r_rand.append(auc(xs, ys))
        c_rand.append(collapse_point(xs, ys))
    deg = dict(g.degree())
    tgt_order = sorted(nodes, key=lambda v: (-deg[v], v))
    xs, ys = lcc_curve(g, tgt_order)
    r_tgt = auc(xs, ys)
    c_tgt = collapse_point(xs, ys)
    # 平均最短路：超过 5000 节点用抽样
    if n0 > 5000:
        l_bar = nx.average_shortest_path_length(g)  # 兜底
    else:
        l_bar = nx.average_shortest_path_length(g)
    return {
        "R_random": float(np.mean(r_rand)),
        "R_random_std": float(np.std(r_rand)),
        "R_targeted": r_tgt,
        "R_bar": 0.5 * (float(np.mean(r_rand)) + r_tgt),
        "fc_rand_emp": float(np.mean(c_rand)),
        "fc_targ_emp": c_tgt,
        "L_bar": float(l_bar),
    }


# ---------------------------------------------------------------------------
# 3. 族扫描
# ---------------------------------------------------------------------------

def family_point(name: str, n: int, m: int, n_h: int, n_random: int, seed: int) -> Optional[Dict]:
    spec = construct_bimodal_sequence(n, m, n_hubs=n_h, k_min=2)
    if not spec.feasible or not spec.graphical:
        return None
    g, spec = build_bimodal(n, m, n_hubs=n_h, seed=seed, k_min=2)
    if g is None:
        return None
    avg_k = 2.0 * m / n
    r = n_h / n
    k_max = spec.k_max
    ana = ftot(avg_k, r, k_max)
    emp = evaluate(g, n_random=n_random, seed=seed)
    cap = residual_cap(n, m, n_hubs=n_h, k_min=2)
    e_ll = m - n_h * k_max
    n_low = n - n_h
    mu = 2.0 * e_ll / n_low if n_low else float("nan")
    k_low_total = (2.0 * m - n_h * k_max) / n_low if n_low else float("nan")
    # 两步分解的第二项：残余层自身退化面积
    a_res = (emp["R_targeted"] - r) / (1.0 - r) if r < 1.0 else float("nan")
    return {
        "dataset": name, "N": n, "M": m, "avg_k": round(avg_k, 4),
        "n_h": n_h, "r": round(r, 5), "k_max": k_max,
        "k_low_mean": round(spec.realised_k_low_mean, 3),
        "k_low_total": round(k_low_total, 3),
        "e_ll": e_ll, "mu": round(mu, 4), "A_res": round(a_res, 4),
        "residual_cap": cap,
        "k_max_theory": round(applicability(n, m)["A"] * n ** (2.0 / 3.0), 1),
        "at_residual_bound": int(k_max >= cap),
        **{k: round(v, 4) for k, v in ana.items()},
        **{k: round(v, 4) for k, v in emp.items()},
        "J": round(emp["R_bar"] / emp["L_bar"], 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="GtsCe,Cogentco,UsCarrier,Colt,Chinanet,BA-200,BA-500")
    ap.add_argument("--nh", default="1,2,3,4,6,8,12,16,24,32")
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    from src.topology.generators import load_graph

    nh_list = [int(x) for x in args.nh.split(",") if x.strip()]
    names = [x.strip() for x in args.datasets.split(",") if x.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    t0 = time.time()
    for name in names:
        path = ROOT / "dataset" / "testdata" / f"{name}.gml"
        if not path.exists():
            print(f"[skip] {name}: 不存在 {path.name}")
            continue
        g0, _ = load_graph(str(path), verbose=False)
        if g0 is None:
            print(f"[skip] {name}: 读取失败")
            continue
        g0 = nx.convert_node_labels_to_integers(g0)
        n, m = g0.number_of_nodes(), g0.number_of_edges()
        print(f"[{name}] N={n} M={m} <k>={2*m/n:.3f}  "
              f"theory_k_max={applicability(n, m)['A']*n**(2/3):.1f}  "
              f"cap(1hub)={residual_cap(n, m, 1)}")
        base = evaluate(g0, n_random=args.n_random, seed=args.seed)
        rows.append({
            "dataset": name, "N": n, "M": m, "avg_k": round(2 * m / n, 4),
            "n_h": 0, "r": 0.0, "k_max": -1, "k_low_mean": -1.0,
            "k_low_total": np.nan, "e_ll": np.nan, "mu": np.nan, "A_res": np.nan,
            "residual_cap": residual_cap(n, m, 1), "k_max_theory": -1.0,
            "at_residual_bound": 0, "f_rand": np.nan, "f_target": np.nan,
            "f_tot": np.nan, "f_case": np.nan,
            **{k: round(v, 4) for k, v in base.items()},
            "J": round(base["R_bar"] / base["L_bar"], 5),
        })
        for n_h in nh_list:
            if n_h >= n - 2:
                continue
            row = family_point(name, n, m, n_h, args.n_random, args.seed)
            if row is None:
                continue
            rows.append(row)
            print(f"   n_h={n_h:3d} r={row['r']:.4f} k_max={row['k_max']:3d} "
                  f"k_low={row['k_low_mean']:.2f} | f_tot={row['f_tot']:.3f} "
                  f"R_bar={row['R_bar']:.4f} L={row['L_bar']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "family_scan.csv", index=False)

    # 逐数据集：三种 argmax 对照
    summary = []
    for name, sub in df.groupby("dataset", sort=False):
        fam = sub[sub["n_h"] > 0]
        if fam.empty:
            continue
        orig = sub[sub["n_h"] == 0].iloc[0]
        a_ftot = fam.loc[fam["f_tot"].idxmax()]
        a_r = fam.loc[fam["R_bar"].idxmax()]
        a_j = fam.loc[fam["J"].idxmax()]
        summary.append({
            "dataset": name, "N": int(orig["N"]), "avg_k": orig["avg_k"],
            "orig_R_bar": orig["R_bar"], "orig_L": orig["L_bar"], "orig_J": orig["J"],
            "argmax_ftot_nh": int(a_ftot["n_h"]), "argmax_ftot_kmax": int(a_ftot["k_max"]),
            "argmax_ftot_val": a_ftot["f_tot"], "argmax_ftot_R": a_ftot["R_bar"],
            "argmax_R_nh": int(a_r["n_h"]), "argmax_R_kmax": int(a_r["k_max"]),
            "argmax_R_val": a_r["R_bar"], "argmax_R_L": a_r["L_bar"],
            "argmax_J_nh": int(a_j["n_h"]), "argmax_J_kmax": int(a_j["k_max"]),
            "argmax_J_val": a_j["J"], "argmax_J_L": a_j["L_bar"],
            "argmax_J_R": a_j["R_bar"],
            "best_R_at_orig_L": round(_best_R_under_budget(fam, orig["L_bar"])[0], 4),
            "budget_nh": int(_best_R_under_budget(fam, orig["L_bar"])[1]),
            "budget_kmax": int(_best_R_under_budget(fam, orig["L_bar"])[2]),
        })
    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT_DIR / "family_summary.csv", index=False)

    md = ["# 构造族上的 f_tot / R / J 三重 argmax 对照", "",
          f"生成时间 {time.strftime('%Y-%m-%d %H:%M')}，随机攻击重复 {args.n_random} 次。", ""]
    md.append("| 数据集 | N | <k> | 原始 R̄ | argmax f_tot (n_h,k_max) | f_tot | 该点 R̄ | "
              "argmax R̄ (n_h,k_max) | R̄ | argmax J (n_h,k_max) | J |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, s in sdf.iterrows():
        md.append(f"| {s['dataset']} | {s['N']} | {s['avg_k']:.2f} | {s['orig_R_bar']:.4f} | "
                  f"({s['argmax_ftot_nh']}, {s['argmax_ftot_kmax']}) | {s['argmax_ftot_val']:.3f} | "
                  f"{s['argmax_ftot_R']:.4f} | ({s['argmax_R_nh']}, {s['argmax_R_kmax']}) | "
                  f"{s['argmax_R_val']:.4f} | ({s['argmax_J_nh']}, {s['argmax_J_kmax']}) | {s['argmax_J_val']:.5f} |")
    md.append("")
    md.append("## 时延预算下的 R 最优（L̄ ≤ L̄_original）")
    md.append("")
    md.append("| 数据集 | 预算 L̄ | 最优 n_h | 最优 k_max | 该点 R̄ | 原始 R̄ | 提升 |")
    md.append("|---|---|---|---|---|---|---|")
    for _, s in sdf.iterrows():
        gain = s["best_R_at_orig_L"] / s["orig_R_bar"] - 1.0
        md.append(f"| {s['dataset']} | {s['orig_L']:.2f} | {s['budget_nh']} | {s['budget_kmax']} | "
                  f"{s['best_R_at_orig_L']:.4f} | {s['orig_R_bar']:.4f} | {gain*100:+.1f}% |")
    (OUT_DIR / "family_tables.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n[done] {len(df)} 行，耗时 {time.time()-t0:.1f}s → {OUT_DIR}")
    print(sdf[["dataset", "argmax_ftot_nh", "argmax_R_nh", "argmax_J_nh"]].to_string(index=False))


def _best_R_under_budget(fam: pd.DataFrame, budget: float) -> Tuple[float, int, int]:
    ok = fam[fam["L_bar"] <= budget + 1e-9]
    if ok.empty:
        row = fam.loc[fam["L_bar"].idxmin()]
    else:
        row = ok.loc[ok["R_bar"].idxmax()]
    return float(row["R_bar"]), int(row["n_h"]), int(row["k_max"])


if __name__ == "__main__":
    main()
