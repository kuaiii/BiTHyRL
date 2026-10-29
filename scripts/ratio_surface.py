# -*- coding: utf-8 -*-
"""两个峰的比例：``(ρ, κ)`` 上的二维性能面。

一维族扫描（``scripts/phase_diagram.py``）只沿构造器自己的轨迹走——``k_max`` 由
``min(A N^(2/3), M−N+n_h)`` 决定。要回答"两个峰各占多少"，必须把 ``k_max`` 也放开，
因为它只受 ``μ ≥ 2``（即 ``k_max ≤ 残差上限``）与 ``k_max > k_low`` 约束。

本脚本在 ``(n_h, k_max)`` 网格上直接测 ``R̄``，得到：

    R̄(ρ, κ)   在 μ ≥ 2 的可行域上

并检验两个理论预测：

* **P1** ``μ=2`` 边界上（``ρ = (⟨k⟩−2)/(2(k_max−1))``），``R̄`` 关于 ``κ`` 单调降
  ——即"hub 越集中越差"；
* **P2** 固定 ``ρ`` 时，``R̄`` 关于 ``k_max`` 在可行域上界附近取极大
  ——即"hub 尽量大但仍不破坏残余层"。

用法
    python scripts/ratio_surface.py                       # 默认 4 个代表点
    python scripts/ratio_surface.py --ns 240 --avg-ks 2.6 --n-random 3

输出
    results/ratio_surface/surface.csv      逐 (N, ⟨k⟩, n_h, k_max) 明细
    results/ratio_surface/surface_best.csv 逐点的 argmax 与 P1/P2 判定
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topology.bimodal_construction import (  # noqa: E402
    build_bimodal,
    construct_bimodal_sequence,
    residual_cap,
)
from src.topology.phase_theory import params  # noqa: E402
from scripts.theory_range import evaluate  # noqa: E402

OUT_DIR = ROOT / "results" / "ratio_surface"


def kmax_candidates(n: int, m: int, n_h: int) -> List[int]:
    """给定 ``n_h``，``k_max`` 的可行网格：对数分布 + 上界本身。

    下界必须满足 ``k_max > k_low``（严格双峰），上界是残差上限。
    """
    cap = residual_cap(n, m, n_h, k_min=2)
    if cap < 3:
        return []
    lo = max(3, int(math.ceil(2.0 * m / n)) + 1)  # k_max > <k> 是严格双峰的必要条件
    if lo > cap:
        return []
    grid = {cap}
    for v in (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
        if lo <= v <= cap:
            grid.add(v)
    # 均匀补两点，避免对数网格在大端留空
    for f in (0.25, 0.5, 0.75):
        v = int(round(lo + f * (cap - lo)))
        if lo <= v <= cap:
            grid.add(v)
    return sorted(grid)


def run_point(n: int, avg_k: float, n_h: int, seed: int, n_random: int) -> List[Dict]:
    m = int(round(avg_k * n / 2.0))
    rows = []
    for k in kmax_candidates(n, m, n_h):
        spec = construct_bimodal_sequence(n, m, n_h, k_max_override=k)
        if not spec.feasible:
            continue
        try:
            g, spec = build_bimodal(n, m, n_h, seed=seed, k_max_override=k)
        except (ValueError, RuntimeError):
            continue
        met = evaluate(g, n_random=n_random, seed=seed)
        p = params(2.0 * m / n, n, n_h / n, spec.k_max, mu_c=2.0, n_h=n_h)
        rows.append(
            {
                "N": n,
                "avg_k": 2.0 * m / n,
                "n_h": n_h,
                "rho": n_h / n,
                "k_max": spec.k_max,
                "k_low": p.k_low,
                "kappa": p.kappa,
                "mu": p.mu,
                "x": p.x,
                "f_rand": p.f_rand,
                "f_target": p.f_target,
                "f_tot": p.f_tot,
                "R_bar": met["R_bar"],
                "R_random": met["R_random"],
                "R_targeted": met["R_targeted"],
                "L_bar": met["L_bar"],
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="240,480")
    ap.add_argument("--avg-ks", default="2.6,3.5,4.5")
    ap.add_argument("--hubs", default="1,2,4,8,16,32")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-random", type=int, default=3)
    args = ap.parse_args()

    ns = [int(x) for x in args.ns.split(",")]
    avg_ks = [float(x) for x in args.avg_ks.split(",")]
    hubs = [int(x) for x in args.hubs.split(",")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict] = []
    print("=" * 84)
    print(f"比例曲面：N ∈ {ns}，⟨k⟩ ∈ {avg_ks}，n_h ∈ {hubs}")
    print("=" * 84)

    for avg_k in avg_ks:
        for n in ns:
            m = int(round(avg_k * n / 2.0))
            t0 = time.time()
            cnt = 0
            for n_h in hubs:
                if n_h >= n - 3:
                    continue
                rows = run_point(n, avg_k, n_h, args.seed, args.n_random)
                all_rows.extend(rows)
                cnt += len(rows)
            print(f"  N={n:5d} ⟨k⟩={2*m/n:4.2f}  可行格点={cnt:3d}  ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "surface.csv", index=False)

    # ---- 逐 (N, ⟨k⟩, n_h) 的最优 k_max ----
    best_rows = []
    for (n, ak, n_h), s in df.groupby(["N", "avg_k", "n_h"]):
        b = s.loc[s.R_bar.idxmax()]
        best_rows.append(
            {
                "N": n, "avg_k": ak, "n_h": n_h, "rho": b.rho,
                "k_max": b.k_max, "kappa": b.kappa, "mu": b.mu, "x": b.x,
                "R_bar": b.R_bar, "L_bar": b.L_bar, "f_tot": b.f_tot,
                "grid_max": s.k_max.max(),
            }
        )
    bdf = pd.DataFrame(best_rows)

    # ---- P1：μ=2 边界上 κ 与 R̄ 的关系 ----
    print("\n" + "=" * 84)
    print("P1  μ≈2（残差饱和）时，κ 越小是否 R̄ 越高")
    print("=" * 84)
    edges = df[(df.mu <= 2.05)].copy()
    if len(edges):
        for (n, ak), s in edges.groupby(["N", "avg_k"]):
            if s.kappa.nunique() < 3:
                continue
            rho_s = s.groupby("n_h").agg(kappa=("kappa", "mean"), R=("R_bar", "mean")).sort_values("kappa")
            corr = rho_s.kappa.corr(rho_s.R)
            print(f"  N={n} ⟨k⟩={ak:.2f}: corr(κ, R̄) = {corr:+.3f}  "
                  f"κ∈[{rho_s.kappa.min():.2f},{rho_s.kappa.max():.2f}]  "
                  f"R̄∈[{rho_s.R.min():.4f},{rho_s.R.max():.4f}]")

    # ---- P2：固定 ρ，最优 k_max 是否贴近上界 ----
    print("\n" + "=" * 84)
    print("P2  固定 n_h 时，最优 k_max 相对可行上界的位置")
    print("=" * 84)
    with pd.option_context("display.width", 200):
        tmp = bdf.copy()
        tmp["k_ratio"] = tmp.k_max / tmp.grid_max
        print(tmp[["N", "avg_k", "n_h", "rho", "k_max", "grid_max", "k_ratio",
                   "kappa", "mu", "R_bar", "L_bar"]].round(3).to_string(index=False))

    bdf.to_csv(OUT_DIR / "surface_best.csv", index=False)
    print(f"\n完成：{OUT_DIR}")


if __name__ == "__main__":
    main()
