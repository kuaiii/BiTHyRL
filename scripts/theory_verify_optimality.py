# -*- coding: utf-8 -*-
"""理论验证实验：双峰最优性与类双峰族的可行域内最优。

回答两件事（对应论文需要证明的两条主张）：

A. **理论区内**（``applicability()['single_hub_feasible'] == True``）：把 k_max 从
   k_min 扫到残差上限，看平均鲁棒性 R̄ = (R_random + R_targeted)/2 峰在哪里。
   若峰落在 k_max = A·N^(2/3)，则"理论先验确实是最优构造"成立。
   若峰落在上限（即越大越好），说明在该密度下最优点被可行性边界截断——
   这本身就是"理论区外需要替代方案"的证据。

B. **理论区外**：在类双峰族 B(n_h)（n_h = 1,2,3,…）上扫描，看 R̄ 是否有内部最优，
   以及该最优相对 null model（保度随机重连，即"同为 N、M、同度序列的均匀化版本"）
   与原始图能提升多少。

输出：
    results/theory_opt/optimality_sweep.csv     每次评估的明细
    results/theory_opt/optimality_summary.csv   每个数据集的结论行
    results/theory_opt/optimality_tables.md     可直接粘进论文的 markdown 表

用法：
    python scripts/theory_verify_optimality.py                       # 全部 11 个数据集
    python scripts/theory_verify_optimality.py --datasets GtsCe,BA-200
    python scripts/theory_verify_optimality.py --seeds 5 --quick
"""

from __future__ import annotations

import argparse
import os
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

from src.topology.generators import load_graph  # noqa: E402
from src.topology.bimodal_construction import (  # noqa: E402
    applicability,
    build_bimodal,
    construct_bimodal_sequence,
    residual_cap,
    theory_k_max,
)
from src.topology.reconstruction import _attack_auc_lcc  # noqa: E402

OUT_DIR = ROOT / "results" / "theory_opt"


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

def _random_sequence(G: nx.Graph, seed: int) -> List[int]:
    rng = random.Random(seed)
    seq = list(G.nodes())
    rng.shuffle(seq)
    return seq


def _targeted_sequence(G: nx.Graph) -> List[int]:
    """静态最高度优先（与 reconstruction._attack_sequence_for_method 的 degree 口径一致）。"""
    return [v for v, _ in sorted(G.degree(), key=lambda kv: kv[1], reverse=True)]


def evaluate_topology(
    G: nx.Graph, seeds: List[int], hub_layer: int = 0
) -> Dict[str, float]:
    """返回 R_random / R_targeted / R_bar、效率项与"摘掉 hub 层后的 LCC"。

    效率用平均最短路径 L̄ 衡量（控制平面时延的直接代理）：L̄ 越小，控制器到交换机
    的平均跳数越少。``efficiency = 1/L̄`` 便于线性比较。
    """
    r_rand, r_targ = [], []
    for s in seeds:
        r_rand.append(_attack_auc_lcc(G, _random_sequence(G, s)))
        r_targ.append(_attack_auc_lcc(G, _targeted_sequence(G)))
    n0 = G.number_of_nodes()
    connected = nx.is_connected(G)
    # 效率口径：平均最短路在最大连通分量内计算（断连图也不至于出 NaN），
    # 同时记录连通性，避免把"靠断连换来的短路径"误读成效率高。
    target = G if connected else G.subgraph(max(nx.connected_components(G), key=len))
    try:
        l_avg = float(nx.average_shortest_path_length(target)) if target.number_of_nodes() > 1 else float("nan")
    except Exception:
        l_avg = float("nan")
    gcc_after = np.nan
    if hub_layer > 0:
        H = G.copy()
        H.remove_nodes_from(_targeted_sequence(G)[:hub_layer])
        if H.number_of_nodes() > 0:
            gcc_after = max(len(c) for c in nx.connected_components(H)) / n0
        else:
            gcc_after = 0.0
    r_bar = 0.5 * (float(np.mean(r_rand)) + float(np.mean(r_targ)))
    eff = 1.0 / l_avg if l_avg and not np.isnan(l_avg) else np.nan
    return {
        "R_random": float(np.mean(r_rand)),
        "R_random_std": float(np.std(r_rand, ddof=0)),
        "R_targeted": float(np.mean(r_targ)),
        "R_bar": r_bar,
        "L_avg": l_avg,
        "efficiency": eff,
        # 双目标联合指标：鲁棒性与效率的乘积。无需归一化参数，量纲为 [0,1]×[0,1]。
        "J": r_bar * eff if not np.isnan(eff) else np.nan,
        "connected": bool(connected),
        "gcc_after_hub_layer": gcc_after,
    }


def null_model(G: nx.Graph, seed: int = 7) -> nx.Graph:
    """保度随机重连（配置模型的 2-switch 版本）：同 N、同 M、同度序列，只打散结构。"""
    H = G.copy()
    m = H.number_of_edges()
    try:
        nx.double_edge_swap(H, nswap=10 * m, max_tries=100 * m, seed=seed)
    except nx.NetworkXAlgorithmError:
        pass
    return H


# ---------------------------------------------------------------------------
# 单数据集：k_max 扫描 + n_h 扫描
# ---------------------------------------------------------------------------

def sweep_one(
    name: str, G: nx.Graph, seeds: List[int], max_hubs: int = 20, quick: bool = False
) -> Tuple[List[dict], dict]:
    n, m = G.number_of_nodes(), G.number_of_edges()
    info = applicability(n, m)
    cap1 = info["k_max_cap_single_hub"]
    kth = info["k_max_theory"]
    rows: List[dict] = []

    def record(family: str, param: float, H: nx.Graph, kmax: float, klow: float,
               hubs: int, extra: Optional[dict] = None) -> dict:
        ev = evaluate_topology(H, seeds, hub_layer=hubs)
        row = {
            "dataset": name, "family": family, "param": param,
            "k_max": kmax, "k_low_mean": klow, "n_hubs": hubs, **ev,
        }
        if extra:
            row.update(extra)
        rows.append(row)
        return row

    # --- 参照组 ---
    record("original", 0, G.copy(), max(d for _, d in G.degree()),
           float(np.mean([d for _, d in G.degree()])), 0)
    record("null_model", 0, null_model(G), np.nan, np.nan, 0)

    # --- A. k_max 扫描（单 hub），只在可行域内 ---
    lo = max(4, int(round(2 * m / n)))          # 起点取平均度附近，避免退化为纯环
    if cap1 > lo:
        if quick:
            grid = sorted({lo, cap1 // 2, cap1})
        else:
            # 用 10 点均匀网格 + 端点，既能定位内部峰，也不会漏掉被边界截断的情形
            grid = sorted({int(round(x)) for x in np.linspace(lo, cap1, 10)} | {lo, cap1})
        for kmax in grid:
            spec = construct_bimodal_sequence(n, m, n_hubs=1, k_max_override=kmax)
            if not spec.feasible:
                continue
            try:
                H, _ = build_bimodal(n, m, n_hubs=1, seed=42, k_max_override=kmax)
            except Exception:
                continue
            record("kmax_sweep", kmax, H, spec.k_max,
                   round(spec.realised_k_low_mean, 3), 1,
                   {"k_max_theory": kth, "cap": cap1})

    # --- B. 类双峰族 n_h 扫描 ---
    nh_grid = [1, 2, 3, 5, 8, 12, 20] if not quick else [1, 3, 8]
    for nh in nh_grid:
        if nh >= n // 4 or nh > info["n_hubs_max"]:
            break
        spec = construct_bimodal_sequence(n, m, n_hubs=nh)
        if not spec.feasible:
            continue
        try:
            H, _ = build_bimodal(n, m, n_hubs=nh, seed=42)
        except Exception:
            continue
        record("hubs_sweep", nh, H, spec.k_max,
               round(spec.realised_k_low_mean, 3), nh,
               {"k_max_theory": kth, "cap": cap1})

    df = pd.DataFrame(rows)
    # --- 结论行：以 J = R̄·E 选点（鲁棒–效率联合最优），同时保留纯 R̄ 最优备查 ---
    fam = df[df.family.isin(["kmax_sweep", "hubs_sweep"])].dropna(subset=["J"])
    best = fam.loc[fam["J"].idxmax()] if len(fam) else None
    hub_rows = df[df.family == "hubs_sweep"].dropna(subset=["R_bar"])
    best_rbar = hub_rows.loc[hub_rows["R_bar"].idxmax()] if len(hub_rows) else None
    kmax_rows = df[df.family == "kmax_sweep"].dropna(subset=["R_bar"])
    orig = df[df.family == "original"].iloc[0]
    null = df[df.family == "null_model"].iloc[0]
    summary = {
        "dataset": name, "n": n, "m": m,
        "avg_k": round(info["avg_k"], 3),
        "A": round(info["A"], 3),
        "k_max_theory": kth,
        "k_max_cap": cap1,
        "clamp": info["clamp"],
        "n_min_theory": round(info["n_min_theory"], 1) if np.isfinite(info["n_min_theory"]) else np.inf,
        "regime": "理论区内" if info["single_hub_feasible"] else "理论区外",
        "R_bar_original": round(orig["R_bar"], 4),
        "R_bar_null": round(null["R_bar"], 4),
        "J_original": round(float(orig["J"]), 4),
        "J_null": round(float(null["J"]), 4),
        "J_best": round(float(best["J"]), 4) if best is not None else np.nan,
        "n_hubs_best": int(best["n_hubs"]) if best is not None else -1,
        "k_max_best": int(best["k_max"]) if best is not None else -1,
        "R_bar_best": round(float(best["R_bar"]), 4) if best is not None else np.nan,
        "L_avg_best": round(float(best["L_avg"]), 3) if best is not None else np.nan,
        "gain_J_vs_original": round(float(best["J"]) / float(orig["J"]), 3) if best is not None and orig["J"] > 0 else np.nan,
        "gain_J_vs_null": round(float(best["J"]) / float(null["J"]), 3) if best is not None and null["J"] > 0 else np.nan,
        "R_bar_max_family": round(float(best_rbar["R_bar"]), 4) if best_rbar is not None else np.nan,
        "n_hubs_at_Rbar_max": int(best_rbar["n_hubs"]) if best_rbar is not None else -1,
        "gcc_after_hub_layer_best": round(float(best["gcc_after_hub_layer"]), 4) if best is not None else np.nan,
        "kmax_sweep_argmax": int(kmax_rows.loc[kmax_rows["R_bar"].idxmax(), "k_max"]) if len(kmax_rows) else -1,
        "kmax_sweep_argmax_Rbar": round(float(kmax_rows["R_bar"].max()), 4) if len(kmax_rows) else np.nan,
    }
    return rows, summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-hubs", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.datasets:
        names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    else:
        names = sorted(p.stem for p in (ROOT / "dataset" / "testdata").glob("*.gml"))

    # 固定攻击种子，保证跨构造可比（对应审稿要求"同一攻击实现"）
    seeds = [101, 202, 303, 404, 505][: args.seeds]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows, summaries = [], []
    for name in names:
        path = ROOT / "dataset" / "testdata" / f"{name}.gml"
        if not path.exists():
            print(f"[skip] {name}: 不存在")
            continue
        t0 = time.time()
        G, _ = load_graph(str(path))
        G = nx.convert_node_labels_to_integers(G)
        try:
            rows, summ = sweep_one(name, G, seeds, args.max_hubs, args.quick)
        except Exception as exc:  # 单个数据集失败不应中断全量
            print(f"[fail] {name}: {type(exc).__name__}: {exc}")
            continue
        all_rows.extend(rows)
        summaries.append(summ)
        print(f"[ok] {name:10s} {time.time() - t0:6.1f}s  "
              f"n_h*={summ['n_hubs_best']:2d} k_max*={summ['k_max_best']:3d} "
              f"J={summ['J_best']:.4f} (orig {summ['J_original']:.4f}, null {summ['J_null']:.4f}) "
              f"R̄={summ['R_bar_best']:.4f} L̄={summ['L_avg_best']:.2f}  {summ['regime']}")

    if not all_rows:
        print("没有产出任何结果")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "optimality_sweep.csv", index=False, encoding="utf-8-sig")
    sdf = pd.DataFrame(summaries)
    sdf.to_csv(OUT_DIR / "optimality_summary.csv", index=False, encoding="utf-8-sig")

    # ---- markdown 表 ----
    md = ["# 理论验证：双峰最优性与类双峰族的可行域最优", ""]
    md.append(f"攻击种子：{seeds}；R̄ = (R_random + R_targeted)/2；"
              "R 为移除比例–LCC AUC。")
    md.append("")
    md.append("## 1. 适用范围判定")
    md.append("")
    md.append("| 数据集 | N | M | ⟨k⟩ | A | k_max(理论原始值) | 残差上限 | 截断原因 | N_min | N/N_min | 区间 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        nmin = s["n_min_theory"]
        ratio = s["n"] / nmin if np.isfinite(nmin) else float("inf")
        md.append(
            f"| {s['dataset']} | {s['n']} | {s['m']} | {s['avg_k']:.2f} | {s['A']:.2f} | "
            f"{s['k_max_theory']} | {s['k_max_cap']} | {s['clamp']} | "
            f"{nmin if np.isfinite(nmin) else '∞'} | {ratio:.2f} | {s['regime']} |"
        )
    md.append("")
    md.append("## 2. 类双峰族的最优点（按 J = R̄·(1/L̄) 选点）")
    md.append("")
    md.append("| 数据集 | J(原始) | J(保度null) | J(最优) | n_h* | k_max* | 相对原始 | 相对null | R̄* | L̄* | 摘 hub 层后 LCC |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        md.append(
            f"| {s['dataset']} | {s['J_original']:.4f} | {s['J_null']:.4f} | {s['J_best']:.4f} | "
            f"{s['n_hubs_best']} | {s['k_max_best']} | "
            f"{s['gain_J_vs_original']:.2f}× | {s['gain_J_vs_null']:.2f}× | "
            f"{s['R_bar_best']:.4f} | {s['L_avg_best']:.2f} | {s['gcc_after_hub_layer_best']:.3f} |"
        )
    md.append("")
    md.append("## 3. 纯鲁棒性最优（R̄ 最大，不看效率）——用于说明取舍")
    md.append("")
    md.append("| 数据集 | R̄(族内最大) | n_h(纯鲁棒最优) | R̄(联合最优) | L̄(联合最优) |")
    md.append("|---|---|---|---|---|")
    for s in summaries:
        md.append(
            f"| {s['dataset']} | {s['R_bar_max_family']:.4f} | {s['n_hubs_at_Rbar_max']} | "
            f"{s['R_bar_best']:.4f} | {s['L_avg_best']:.2f} |"
        )
    md.append("")
    md.append("## 4. k_max 扫描的峰位（单 hub）")
    md.append("")
    md.append("| 数据集 | k_max(理论) | 残差上限 | 扫描最优 k_max | R̄@最优 | R̄@族内最大 | 说明 |")
    md.append("|---|---|---|---|---|---|---|")
    for s in summaries:
        note = "峰位=上限，被可行性边界截断" if s["kmax_sweep_argmax"] == s["k_max_cap"] else "内部峰"
        md.append(
            f"| {s['dataset']} | {s['k_max_theory']} | {s['k_max_cap']} | "
            f"{s['kmax_sweep_argmax']} | {s['kmax_sweep_argmax_Rbar']:.4f} | "
            f"{s['R_bar_max_family']:.4f} | {note} |"
        )
    (OUT_DIR / "optimality_tables.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n写入 {OUT_DIR}")
    print("  optimality_sweep.csv / optimality_summary.csv / optimality_tables.md")


if __name__ == "__main__":
    main()
