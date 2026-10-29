# -*- coding: utf-8 -*-
"""相图理论的五项自检：恒等式、等价性、渐近、最优标度、判据一致性。

任何一项失败都说明理论部分不能写进论文，因此本脚本是 ``phase_theory`` 的
配套闸门，CI 与人工都以它为准。

    python scripts/test_phase_theory.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topology.bimodal_construction import (  # noqa: E402
    applicability,
    build_bimodal,
    residual_cap,
)
from src.topology.generators import load_graph  # noqa: E402
from src.topology.phase_theory import (  # noqa: E402
    coef_A,
    f_tot_of,
    hub_edge_share_cap,
    min_size,
    optimal_ratios,
    params,
    regime,
)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# 1. 恒等式 (III) 在真实构造图上成立
# ---------------------------------------------------------------------------
def t1_identity() -> None:
    print("\n--- T1  ⟨k⟩ = μ(1−ρ) + 2ρ·k_max 在构造图上成立 ---")
    cases = [(150, 193), (200, 591), (500, 1491)]
    for n, m in cases:
        avg_k = 2.0 * m / n
        for n_h in (1, 2, 3, 5):
            try:
                G, spec = build_bimodal(n, m, n_hubs=n_h, seed=7)
            except ValueError:
                continue
            deg = dict(G.degree())
            hubs = list(G.graph["hub_nodes"])
            low = [v for v in G if v not in set(hubs)]
            # 实测 x 与 mu
            hub_deg = sum(deg[v] for v in hubs)
            x_emp = hub_deg / m
            e_ll = sum(1 for u, v in G.edges() if u in low_set(low) and v in low_set(low))
            mu_emp = 2.0 * e_ll / len(low)
            rho = n_h / n
            lhs = avg_k
            rhs = mu_emp * (1 - rho) + 2 * rho * (hub_deg / n_h)
            ok = abs(lhs - rhs) < 1e-9
            # x_emp 应等于 2*rho*k_max/<k>
            ok2 = abs(x_emp - 2 * rho * (hub_deg / n_h) / avg_k) < 1e-12
            check(
                f"T1 N={n} n_h={n_h}",
                ok and ok2,
                f"⟨k⟩={lhs:.6f} rhs={rhs:.6f} μ_emp={mu_emp:.4f} x_emp={x_emp:.4f}",
            )


def low_set(nodes):
    return set(nodes)


# ---------------------------------------------------------------------------
# 2. μ ≥ 2  ⟺  k_max ≤ (M−N+n_h)/n_h  （残差约束的等价形式）
# ---------------------------------------------------------------------------
def t2_equivalence() -> None:
    print("\n--- T2  μ≥2 与残差约束 (★) 等价 ---")
    rng = [(n, m) for n in (100, 200, 500, 1000) for m in (int(1.3 * n), int(2.5 * n), int(4 * n))]
    worst = 0.0
    for n, m in rng:
        avg_k = 2.0 * m / n
        if avg_k <= 2.0:
            continue
        for n_h in (1, 2, 3, 4, 6, 8):
            cap = residual_cap(n, m, n_h, k_min=2)
            if cap <= 2:
                continue
            rho = n_h / n
            mu_at_cap = (avg_k - 2 * rho * cap) / (1 - rho)
            mu_over = (avg_k - 2 * rho * (cap + 1)) / (1 - rho)
            capped_by_n = cap >= n - 1
            if capped_by_n:
                # k_max 先撞到简单图上限 N-1，此时残差约束仍宽裕，mu > 2 属正常
                ok = mu_at_cap >= 2.0 - 1e-9
                tag = "（撞 N−1）"
            else:
                worst = max(worst, abs(mu_at_cap - 2.0))
                ok = mu_at_cap >= 2.0 - 1e-9 and mu_over < 2.0 + 1e-9
                tag = ""
            check(f"T2 N={n} M={m} n_h={n_h}{tag}", ok,
                  f"cap={cap} μ(cap)={mu_at_cap:.6f} μ(cap+1)={mu_over:.4f}")
    print(f"     max |μ(cap) − 2| = {worst:.2e}")


# ---------------------------------------------------------------------------
# 3. 渐近：ρ→0 时 f_tot → 2 − 1/(⟨k⟩−1)，速度是 O(A·r^(1/3))
# ---------------------------------------------------------------------------
def t3_asymptotic() -> None:
    print("\n--- T3  ρ→0 极限与收敛速度 ---")
    for avg_k in (3.0, 4.0, 6.0):
        A = coef_A(avg_k)
        limit = 2.0 - 1.0 / (avg_k - 1.0)
        scaled = []
        for n in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6):
            r = 1.0 / n
            k_max = A * n ** (2.0 / 3.0)
            if k_max >= n - 1:
                continue
            f = f_tot_of(avg_k, r, k_max, mu_c=2.0)
            gap = limit - f
            scaled.append(gap * n ** (1.0 / 3.0))
            print(f"    ⟨k⟩={avg_k} N=1e{int(math.log10(n))} f={f:.6f} gap={gap:.3e} gap·N^(1/3)={scaled[-1]:.4f}")
        # 修正项 ~ A·N^(-1/3)，故 gap·N^(1/3) 应趋于常数（漂移 < 30%）
        ok = len(scaled) >= 3 and abs(scaled[-1] / scaled[0] - 1.0) < 0.30
        check(f"T3 ⟨k⟩={avg_k} 收敛到 f_∞={limit:.4f}，速率 N^(-1/3)", ok)


# ---------------------------------------------------------------------------
# 4. 最优标度：k_max* · r^(2/3) → A
# ---------------------------------------------------------------------------
def t4_scaling() -> None:
    print("\n--- T4  最优点的 k_max*·ρ^(2/3) → A ---")
    for avg_k in (4.0, 6.0):
        A = coef_A(avg_k)
        vals = []
        for n in (2000, 20000, 200000):
            o = optimal_ratios(avg_k, n, mu_c=2.0, n_rho=40, n_k=64)
            if o.get("best") is None:
                continue
            rho = o["rho_star"]
            k = o["k_max_star"]
            vals.append((n, rho, k, k * rho ** (2.0 / 3.0)))
            print(f"    ⟨k⟩={avg_k} N={n:7d} ρ*={rho:.3e} k*={k:8.1f} k*·ρ^(2/3)={k*rho**(2/3):.3f} (A={A:.3f})")
        ok = bool(vals) and abs(vals[-1][3] - A) / A < 0.25
        check(f"T4 ⟨k⟩={avg_k} 标度收敛到 A", ok, f"A={A:.3f} 末值={vals[-1][3]:.3f}" if vals else "")


# ---------------------------------------------------------------------------
# 5. 判据一致性：regime 与 construction.applicability 不矛盾
# ---------------------------------------------------------------------------
def t5_regime() -> None:
    print("\n--- T5  regime 判据 vs applicability ---")
    datasets = ["GtsCe", "Cogentco", "UsCarrier", "Colt", "Chinanet", "BA-200", "BA-300", "BA-500"]
    for name in datasets:
        try:
            g, _ = load_graph(f"dataset/testdata/{name}.gml", verbose=False)
        except Exception as exc:  # noqa: BLE001
            print(f"    [skip] {name}: {exc}")
            continue
        g = nx.convert_node_labels_to_integers(g)
        n, m = g.number_of_nodes(), g.number_of_edges()
        avg_k = 2.0 * m / n
        a = applicability(n, m)
        r = regime(avg_k, n)
        nmin = min_size(avg_k)
        consistent = (r == "III_asymptotic") == bool(a["usable"])
        check(
            f"T5 {name:10s}",
            consistent,
            f"⟨k⟩={avg_k:.2f} N={n} regime={r} usable={a['usable']} μ={a['mu_residual']:.2f} N_min={nmin:.3g}",
        )


# ---------------------------------------------------------------------------
# 6. 相图单调性：N 增大 ⟹ 最优点向 ρ=1/N（单 hub）移动
# ---------------------------------------------------------------------------
def t6_monotone() -> None:
    print("\n--- T6  N ↑ ⟹ n_h* ↓（μ≥3 门槛下，小图内点、大图单 hub）---")
    for avg_k in (3.5, 4.0, 6.0):
        ns = [200, 500, 2000, 10000, 100000]
        vals = []
        for n in ns:
            o = optimal_ratios(avg_k, n, mu_c=3.0, n_rho=40, n_k=48)
            vals.append((n, o.get("n_h_star", float("nan")), o.get("rho_star", float("nan"))))
        seq = [v[1] for v in vals if v[1] == v[1]]
        ok = all(seq[i] >= seq[i + 1] - 1e-9 for i in range(len(seq) - 1))
        check(f"T6 ⟨k⟩={avg_k}", ok, " n_h*: " + ", ".join(f"{v[1]:.1f}" for v in vals))
        for n, nh, rho in vals:
            print(f"    N={n:7d} n_h*={nh:8.2f} ρ*={rho:.3e}")


if __name__ == "__main__":
    print("=" * 78)
    print("相图理论自检")
    print("=" * 78)
    t1_identity()
    t2_equivalence()
    t3_asymptotic()
    t4_scaling()
    t5_regime()
    t6_monotone()
    print("\n" + "=" * 78)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL PASSED")
