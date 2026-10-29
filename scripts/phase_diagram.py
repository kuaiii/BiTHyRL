# -*- coding: utf-8 -*-
"""``(N, ⟨k⟩)`` 平面上的构造相图实验。

回答三个问题（对应审稿人最可能追问的三条）：

1. **给定 ``(N, ⟨k⟩)``，族的哪个点最好？** 逐点扫描 ``n_h``，取实测 ``R̄`` 的 argmax，
   与解析 ``f^tot`` 的 argmax 对照。
2. **最优比例 ``(ρ*, κ*)`` 如何随 ``(N, ⟨k⟩)`` 变化？** 把 ``ρ* = n_h*/N`` 与
   ``κ* = k_max*/k_low*`` 画在相图上。
3. **分区是否与理论一致？** 三个区（``⟨k⟩≤3`` 不可达 / 理论解受压 / 理论解可用）
   的边界，与实测最优点的形态是否吻合。

用法
    python scripts/phase_diagram.py                    # 默认网格，约 5 分钟
    python scripts/phase_diagram.py --quick            # 小网格冒烟
    python scripts/phase_diagram.py --avg-ks 2.6,4,7 --ns 120,480

输出
    results/phase_diagram/phase_experiment.csv   逐 (N, ⟨k⟩, n_h) 明细
    results/phase_diagram/phase_best.csv         逐 (N, ⟨k⟩) 的 argmax 与解析预测
    results/phase_diagram/phase_theory.csv        解析相图（密集网格）
    results/phase_diagram/phase_tables.md         可直接进论文的表
"""

from __future__ import annotations

import argparse
import math
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
    max_bimodal_hubs,
    residual_cap,
    theory_k_max,
)
from src.topology.phase_theory import (  # noqa: E402
    coef_A,
    min_size,
    optimal_ratios,
    params,
    phase_scan,
    regime,
    REGIMES,
)
from scripts.theory_range import evaluate  # noqa: E402

OUT_DIR = ROOT / "results" / "phase_diagram"


def hub_grid(n: int, m: int) -> List[int]:
    """自适应 ``n_h`` 网格：几何分布，覆盖 1 到族右边界，再加密小端。

    小端（``n_h`` 小）是 I 区的最优区间，必须加密；大端决定退化边界，取几个点即可。
    """
    hi = max_bimodal_hubs(n, m)
    if hi < 1:
        return []
    cands = {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64}
    grid = sorted(c for c in cands if c <= hi)
    if not grid:
        grid = [1]
    if hi not in grid and hi > 1:
        grid.append(hi)
    # 对小图，hi 可能很小，补上中间值
    for frac in (0.1, 0.25, 0.5, 0.75):
        v = max(1, int(round(frac * hi)))
        if v not in grid:
            grid.append(v)
    return sorted(set(grid))


def eval_point(n: int, m: int, n_h: int, seed: int, n_random: int) -> Optional[Dict[str, float]]:
    """构造 + 评估一个 ``(N, M, n_h)``。不可行时返回 None。"""
    spec = construct_bimodal_sequence(n, m, n_h)
    if not spec.feasible:
        return None
    try:
        g, spec = build_bimodal(n, m, n_hubs=n_h, seed=seed)
    except (ValueError, RuntimeError):
        return None
    met = evaluate(g, n_random=n_random, seed=seed)
    rho = n_h / n
    p = params(2.0 * m / n, n, rho, spec.k_max, mu_c=2.0, n_h=n_h)
    return {
        "R_bar": met["R_bar"],
        "R_random": met["R_random"],
        "R_targeted": met.get("R_targeted", float("nan")),
        "L_bar": met["L_bar"],
        "J": met["R_bar"] / met["L_bar"] if met["L_bar"] > 0 else float("nan"),
        "k_max": float(spec.k_max),
        "mu": p.mu,
        "x": p.x,
        "kappa": p.kappa,
        "k_low": p.k_low,
        "f_rand": p.f_rand,
        "f_target": p.f_target,
        "f_tot": p.f_tot,
        "strategy": spec.strategy,
        "rho": rho,
    }


def run_dataset(n: int, avg_k: float, seed: int, n_random: int, verbose: bool = True) -> Tuple[List[Dict], Dict]:
    m = int(round(avg_k * n / 2.0))
    if m < n - 1:
        m = n - 1
    avg_k_real = 2.0 * m / n
    grid = hub_grid(n, m)
    rows: List[Dict] = []
    t0 = time.time()
    for n_h in grid:
        rec = eval_point(n, m, n_h, seed, n_random)
        if rec is None:
            continue
        rec.update(N=n, M=m, avg_k=avg_k_real, n_h=n_h)
        rows.append(rec)
    if verbose:
        best = max(rows, key=lambda r: r["R_bar"]) if rows else None
        tag = f"n_h*={best['n_h']}" if best else "no feasible"
        print(
            f"  N={n:5d} ⟨k⟩={avg_k_real:4.2f} M={m:6d} 族={len(rows):2d} 点 "
            f"{tag} R̄*={best['R_bar']:.4f} μ*={best['mu']:.2f} ({time.time()-t0:.1f}s)"
            if best
            else f"  N={n:5d} ⟨k⟩={avg_k_real:4.2f} 无可行点",
            flush=True,
        )
    return rows, {"N": n, "M": m, "avg_k": avg_k_real, "n_grid": len(grid)}


def summarize(rows: List[Dict], meta: Dict, mu_c: float = 3.0) -> Dict[str, object]:
    n, m, avg_k = meta["N"], meta["M"], meta["avg_k"]
    if not rows:
        return {"N": n, "M": m, "avg_k": avg_k, "regime": regime(avg_k, n), "note": "无可行点"}
    best = max(rows, key=lambda r: r["R_bar"])
    # 解析最优（mu_c=2 可构造域），用于对照
    o = optimal_ratios(avg_k, n, mu_c=2.0, n_rho=32, n_k=40)
    a = applicability(n, m)
    return {
        "N": n,
        "M": m,
        "avg_k": avg_k,
        "A": coef_A(avg_k),
        "regime": regime(avg_k, n, mu_c),
        "n_min": min_size(avg_k, mu_c),
        "n_over_min": n / min_size(avg_k, mu_c) if math.isfinite(min_size(avg_k, mu_c)) else float("inf"),
        "n_h_max_family": max_bimodal_hubs(n, m),
        "n_h_star": best["n_h"],
        "rho_star": best["rho"],
        "k_max_star": best["k_max"],
        "mu_star": best["mu"],
        "x_star": best["x"],
        "kappa_star": best["kappa"],
        "f_tot_star": best["f_tot"],
        "R_bar_star": best["R_bar"],
        "L_bar_star": best["L_bar"],
        "theory_k_max": theory_k_max(n, m),
        "cap_1hub": residual_cap(n, m, 1, 2),
        "mu_residual_1hub": a["mu_residual"],
        "usable_1hub": a["usable"],
        "analytic_n_h": o.get("n_h_star", float("nan")),
        "analytic_k_max": o.get("k_max_star", float("nan")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=str, default="60,120,240,480,960")
    ap.add_argument("--avg-ks", type=str, default="2.2,2.6,3.0,3.5,4.5,7.0")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-theory-grid", action="store_true")
    args = ap.parse_args()

    ns = [int(x) for x in args.ns.split(",")]
    avg_ks = [float(x) for x in args.avg_ks.split(",")]
    if args.quick:
        ns, avg_ks = [120, 480], [2.6, 4.5]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 86)
    print(f"相图实验：N ∈ {ns}，⟨k⟩ ∈ {avg_ks}，n_random={args.n_random}")
    print("=" * 86)

    all_rows: List[Dict] = []
    metas: List[Dict] = []
    for avg_k in avg_ks:
        print(f"\n[⟨k⟩≈{avg_k}]")
        for n in ns:
            rows, meta = run_dataset(n, avg_k, args.seed, args.n_random)
            all_rows.extend(rows)
            metas.append(meta)

    exp = pd.DataFrame(all_rows)
    exp.to_csv(OUT_DIR / "phase_experiment.csv", index=False)
    # 按 (N, avg_k) 分组汇总，保证与 metas 对齐
    summ = []
    for meta in metas:
        sub = [r for r in all_rows if r["N"] == meta["N"] and abs(r["avg_k"] - meta["avg_k"]) < 1e-9]
        summ.append(summarize(sub, meta))
    bdf = pd.DataFrame(summ)
    bdf.to_csv(OUT_DIR / "phase_best.csv", index=False)

    print("\n" + "=" * 86)
    print("实测最优 vs 解析预测")
    print("=" * 86)
    cols = ["N", "avg_k", "regime", "n_h_star", "rho_star", "k_max_star", "kappa_star",
            "mu_star", "f_tot_star", "R_bar_star", "L_bar_star", "usable_1hub", "analytic_n_h"]
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(bdf[cols].round(3).to_string(index=False))

    if not args.no_theory_grid:
        print("\n[解析相图网格]")
        theo = pd.DataFrame(phase_scan(
            avg_ks=[round(2.0 + 0.1 * i, 2) for i in range(31)],
            ns=[50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000],
            mu_c=3.0, n_rho=32, n_k=40,
        ))
        theo.to_csv(OUT_DIR / "phase_theory.csv", index=False)
        print(f"  写入 {len(theo)} 行")

    write_tables(bdf, OUT_DIR / "phase_tables.md")
    print(f"\n完成：{OUT_DIR}")


def write_tables(bdf: pd.DataFrame, path: Path) -> None:
    lines = ["# 相图：实测最优比例", ""]
    for regime_name, g in bdf.groupby("regime", sort=True):
        lines.append(f"## {regime_name} — {REGIMES.get(regime_name, '')}")
        lines.append("")
        lines.append("| N | ⟨k⟩ | N/N_min | n_h* | ρ* | k_max* | κ* | μ* | x* | f_tot* | R̄* | L̄* |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in g.sort_values(["avg_k", "N"]).itertuples():
            lines.append(
                f"| {r.N} | {r.avg_k:.2f} | {getattr(r,'n_over_min',float('nan')):.2f} | "
                f"{r.n_h_star} | {r.rho_star:.3f} | {r.k_max_star:.0f} | {r.kappa_star:.2f} | "
                f"{r.mu_star:.2f} | {r.x_star:.3f} | {r.f_tot_star:.3f} | "
                f"{r.R_bar_star:.4f} | {r.L_bar_star:.2f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
