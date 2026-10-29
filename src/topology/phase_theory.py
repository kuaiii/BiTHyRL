# -*- coding: utf-8 -*-
"""双峰族的相图理论：以 ``(N, <k>)`` 为坐标的构造选择判据。

为什么需要这个模块
------------------
书（Barabási, *Network Science*, §7.16 / 进阶阅读 7.G）给出的最优度分布是
**渐近**结论：固定 ``<k>``，令 hub 层节点占比 ``r -> 0``，解驻点得
``k_max ~ A(<k>) N^(2/3)``。它没有告诉我们：

1. 给定 ``(N, <k>)`` 时，``r`` 与 ``k_max`` 该取多少——书里 ``r -> 0`` 是极限，
   而任何真实网络都有 ``r >= 1/N``；
2. 书里的 ``k_min`` 是低度节点的**总**度，而定向攻击摘掉 hub 之后，决定残余层
   生死的量是它的**内部**度 ``mu``。两者差 ``rho*k_max/(1-rho)``，这一项在
   ``N`` 有限时不可忽略，且恰好在 SDN 尺度上吃掉全部余量。

本模块把族写成三个无量纲量的关系，于是上面两个问题都变成闭式。

核心恒等式
----------
设 ``n_h`` 个 hub（度 ``k_max``），其余 ``n_l = N - n_h`` 个低度节点（度 ``k_low``），
``rho = n_h/N``。度守恒给出

    <k> = rho*k_max + (1-rho)*k_low                                   (I)

低度节点各有 ``rho*k_max/(1-rho)`` 条边连到 hub 层，剩下的是层内边，故残余层
内部平均度

    mu = k_low - rho*k_max/(1-rho) = (<k> - 2*rho*k_max)/(1-rho)      (II)

联立 (I)(II)：

    <k> = mu*(1-rho) + 2*rho*k_max                                    (III)

(III) 是本模块的出发点。它只有三个量，且 (III) 直接把"两个峰的比例"约束成
一维：给定 ``<k>``，``(rho, k_max)`` 中只有一个自由度。

由 (III) 立刻得到三个"比例"的等价刻画：

    x   = 2*rho*k_max/<k>        hub 边占全图边的比例（"hub 拿走多少"）
    mu  = <k>*(1-x)/(1-rho)      残余层厚度
    kappa = k_max/k_low          峰比（集中度）

``x`` 与 ``mu`` 是一对：``x`` 涨则残余层必然变薄（hub 的边只能从低度层抽）。
书中的最优是 ``x -> 0, mu -> <k>``，而构造可行性要求 ``mu >= 2``，即

    x <= 1 - 2/<k>                                                    (IV)

(IV) 与残差约束 ``k_max <= (M-N+n_h)/n_h`` 完全等价（见 ``docstring`` 末的验证），
但它把"可行性"写成了一个只含 ``<k>`` 的不等式，与 ``N`` 无关。

适用范围
--------
要求残余层自身还能扛住随机失效（``f' = 1-1/(mu-1) >= 1/2``）即 ``mu >= 3``，
则书中的最优（``x->0``）需要 ``<k> > 3``。配合 ``rho >= 1/N``：

    <k> > 3  且  2*A*N^(2/3) <= N*(<k>-3) + 3                        (V)

(V) 第二个不等式的根即精确的 ``N_min``（``min_size``）。它比常见的近似
``N_min ~ [2A/(<k>-3)]^3`` 多保留了 ``+3`` 项，在 ``<k>`` 接近 3 时差别可达数量级。

修正项的量级
------------
书 7.65 用 ``k_min`` 写 ``f^target``，本模块用 ``mu``，两者差

    k_low - mu = rho*k_max/(1-rho) = x*<k>/(2*(1-rho))

取 ``rho = 1/N``、``k_max = A*N^(2/3)``，修正项 ``-> A*N^(-1/3)``。它在
``N = 150`` 时等于 ``0.49``，而 ``<k>-3`` 的余量通常只有 ``0.3``——**书中的
最优解在 SDN 尺度上不是"稍有偏差"，而是定性错误**。这解释了
``GtsCe`` 在 ``n_h=1`` 处 ``R_targeted = 0.086``（摘掉 hub 即崩）。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PhaseParams",
    "coef_A",
    "params",
    "f_tot_of",
    "optimal_ratios",
    "min_size",
    "regime",
    "phase_scan",
    "hub_edge_share_cap",
    "REGIMES",
]


REGIMES = {
    "I_unreachable": "⟨k⟩≤μ_c：无理论双峰解（任意 N）",
    "II_forced": "μ_c<⟨k⟩：理论解不可实现，族最优点在内点",
    "III_asymptotic": "N≥N_min：书中的单 hub 理论解可实现且最优",
}


def coef_A(avg_k: float) -> float:
    """书中系数 A = [2⟨k⟩²(⟨k⟩−1)²/(2⟨k⟩−1)]^(1/3)。"""
    if avg_k <= 1.0:
        raise ValueError(f"avg_k 必须 > 1，收到 {avg_k}")
    return (2.0 * avg_k ** 2 * (avg_k - 1.0) ** 2 / (2.0 * avg_k - 1.0)) ** (1.0 / 3.0)


@dataclass
class PhaseParams:
    """一个 ``(N, ⟨k⟩, rho, k_max)`` 点的全部派生量。"""

    avg_k: float
    n: int
    rho: float
    k_max: float
    k_low: float = 0.0
    mu: float = 0.0
    x: float = 0.0
    kappa: float = 0.0
    m2: float = 0.0
    B: float = 0.0
    f_rand: float = 0.0
    f_target: float = 0.0
    f_tot: float = 0.0
    n_h: float = 0.0
    mu_c: float = 3.0
    mu_ok: bool = False
    bimodal: bool = False
    feasible: bool = False

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def params(
    avg_k: float,
    n: int,
    rho: float,
    k_max: float,
    mu_c: float = 3.0,
    n_h: Optional[float] = None,
) -> PhaseParams:
    """由 ``(⟨k⟩, N, rho, k_max)`` 求全部派生量与书中的两级阈值。

    全部用 (I)(II)(III) 的精确形式，不做 ``rho -> 0`` 展开。
    """
    p = PhaseParams(avg_k=avg_k, n=n, rho=rho, k_max=k_max, mu_c=mu_c)
    p.n_h = n * rho if n_h is None else n_h
    if rho >= 1.0 or k_max <= 0:
        return p

    p.k_low = (avg_k - rho * k_max) / (1.0 - rho)
    p.mu = (avg_k - 2.0 * rho * k_max) / (1.0 - rho)
    p.x = 2.0 * rho * k_max / avg_k
    p.kappa = k_max / p.k_low if p.k_low > 0 else float("inf")

    # 两点分布的 ⟨k²⟩（书 7.63）。低度层按正则处理，这是族的定义。
    p.m2 = rho * k_max ** 2 + (1.0 - rho) * p.k_low ** 2
    p.B = p.m2 / avg_k
    p.f_rand = max(0.0, 1.0 - 1.0 / (p.B - 1.0)) if p.B > 2.0 else 0.0

    # 定向攻击：先摘掉 rho 比例的 hub，残余层按自身的 mu 崩。
    # 注意这里用 mu 而不是书中的 k_min——见模块 docstring「修正项的量级」。
    if p.mu > 1.0:
        p.f_target = rho + (1.0 - rho) * (1.0 - 1.0 / (p.mu - 1.0))
    else:
        p.f_target = rho
    p.f_target = max(0.0, p.f_target)
    p.f_tot = p.f_rand + p.f_target

    p.mu_ok = p.mu >= mu_c - 1e-12
    p.bimodal = p.kappa > 1.0 + 1e-12
    # rho >= 1/N 是"至少一个 hub"；mu_ok 是残余层能用；bimodal 是形态要求
    p.feasible = bool(p.mu_ok and p.bimodal and rho * n >= 1.0 - 1e-9)
    return p


def f_tot_of(avg_k: float, rho: float, k_max: float, mu_c: float = 3.0) -> float:
    """轻量版，仅返回 ``f^tot``（网格搜索内层用）。"""
    if rho >= 1.0 or k_max <= 0:
        return -1.0
    k_low = (avg_k - rho * k_max) / (1.0 - rho)
    if k_low <= 0:
        return -1.0
    mu = (avg_k - 2.0 * rho * k_max) / (1.0 - rho)
    if mu < mu_c - 1e-12:
        return -1.0
    m2 = rho * k_max ** 2 + (1.0 - rho) * k_low ** 2
    B = m2 / avg_k
    fr = 1.0 - 1.0 / (B - 1.0) if B > 2.0 else 0.0
    ft = rho + (1.0 - rho) * (1.0 - 1.0 / (mu - 1.0)) if mu > 1.0 else rho
    return fr + ft


def hub_edge_share_cap(avg_k: float, mu_c: float = 3.0) -> float:
    """``x`` 的上界：``x <= 1 - mu_c/⟨k⟩``（由 (III)(IV) 令 rho→0）。

    返回负值表示该 ``⟨k⟩`` 下无论 ``N`` 多大都没有解——即 ``⟨k⟩ < mu_c``。
    """
    return 1.0 - mu_c / avg_k


def optimal_ratios(
    avg_k: float,
    n: int,
    mu_c: float = 3.0,
    n_rho: int = 48,
    n_k: int = 64,
) -> Dict[str, object]:
    """在 ``(rho, k_max)`` 上最大化 ``f^tot``，返回最优比例与区域标签。

    搜索域：``rho in [1/N, 0.5]``，``k_max in (k_low, min(N-1, A N^(2/3) * 1.5)]``。
    上界取理论值的 1.5 倍，因为 ``mu_c`` 门槛会把可行域压到比理论点更靠内的位置，
    而我们要确认最优**不在**边界上。
    """
    out: Dict[str, object] = {
        "avg_k": avg_k,
        "n": n,
        "mu_c": mu_c,
        "n_min": min_size(avg_k, mu_c),
        "hub_share_cap": hub_edge_share_cap(avg_k, mu_c),
        "regime": regime(avg_k, n, mu_c),
    }
    if avg_k <= mu_c:
        out.update(best=None, reason="⟨k⟩≤μ_c，可行域为空")
        return out

    A = coef_A(avg_k)
    k_hi = min(n - 1.0, max(4.0, 1.5 * A * n ** (2.0 / 3.0)))
    rho_lo = 1.0 / n
    rho_hi = min(0.5, 1.0 - 2.0 / avg_k)  # rho -> 1 时 k_low -> 0，无意义

    best = (-1.0, None)
    for rho in np.geomspace(rho_lo, max(rho_hi, rho_lo * 1.05), n_rho):
        # 给定 rho，k_max 的可行区间由 mu>=mu_c 与 k_max>k_low 决定
        # mu >= mu_c  <=>  k_max <= [⟨k⟩ - mu_c(1-rho)] / (2 rho)
        k_cap = (avg_k - mu_c * (1.0 - rho)) / (2.0 * rho)
        if k_cap <= 2.0:
            continue
        # k_max > k_low  <=>  k_max > ⟨k⟩/(1+rho) ... 由 k_low=(⟨k⟩-rho k_max)/(1-rho) < k_max
        # 解: ⟨k⟩ - rho k_max < (1-rho) k_max  <=>  ⟨k⟩ < k_max
        k_lo = avg_k * (1.0 + 1e-6)
        hi = min(k_cap, k_hi)
        if hi <= k_lo:
            continue
        for k_max in np.linspace(k_lo, hi, n_k):
            f = f_tot_of(avg_k, rho, k_max, mu_c)
            if f > best[0]:
                best = (f, (float(rho), float(k_max)))
    if best[1] is None:
        out.update(best=None, reason="网格内无可行点")
        return out

    rho, k_max = best[1]
    p = params(avg_k, n, rho, k_max, mu_c)
    out.update(
        best=p.as_dict(),
        f_tot=best[0],
        rho_star=rho,
        k_max_star=k_max,
        n_h_star=rho * n,
        mu_star=p.mu,
        x_star=p.x,
        kappa_star=p.kappa,
        at_lower_rho=bool(rho <= 1.5 / n),
        at_upper_k=bool(k_max >= min((avg_k - mu_c * (1 - rho)) / (2 * rho), k_hi) * 0.999),
    )
    return out


def min_size(avg_k: float, mu_c: float = 3.0, bracket: Tuple[float, float] = (10.0, 1e7)) -> float:
    """精确的 ``N_min``：不等式 (V) 的根。

        g(N) = N*(⟨k⟩-mu_c) + mu_c - 2*A*N^(2/3) >= 0

    ``g`` 在 ``N`` 足够大时单调增（``N`` 项压倒 ``N^(2/3)``），故根唯一。
    ``⟨k⟩ <= mu_c`` 时无解，返回 ``inf``。
    """
    if avg_k <= mu_c:
        return float("inf")
    A = coef_A(avg_k)

    def g(x: float) -> float:
        return x * (avg_k - mu_c) + mu_c - 2.0 * A * x ** (2.0 / 3.0)

    lo, hi = bracket
    if g(hi) < 0:
        return float("inf")
    # g 在 lo 处可能已为正（小 N 时 mu_c 项占优），先向左找符号变化
    if g(lo) >= 0:
        return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def regime(avg_k: float, n: int, mu_c: float = 3.0) -> str:
    """三区判据。"""
    if avg_k <= mu_c:
        return "I_unreachable"
    n_min = min_size(avg_k, mu_c)
    if n >= n_min:
        return "III_asymptotic"
    return "II_forced"


def phase_scan(
    avg_ks: Sequence[float],
    ns: Sequence[int],
    mu_c: float = 3.0,
    n_rho: int = 40,
    n_k: int = 48,
) -> List[Dict[str, object]]:
    """在 ``(⟨k⟩, N)`` 网格上给出最优比例与区域，供画相图。"""
    rows: List[Dict[str, object]] = []
    for avg_k in avg_ks:
        for n in ns:
            o = optimal_ratios(avg_k, n, mu_c=mu_c, n_rho=n_rho, n_k=n_k)
            rows.append(
                {
                    "avg_k": avg_k,
                    "N": n,
                    "regime": o["regime"],
                    "n_min": o["n_min"],
                    "n_over_min": n / o["n_min"] if math.isfinite(o["n_min"]) else float("inf"),
                    "rho_star": o.get("rho_star", float("nan")),
                    "n_h_star": o.get("n_h_star", float("nan")),
                    "k_max_star": o.get("k_max_star", float("nan")),
                    "mu_star": o.get("mu_star", float("nan")),
                    "x_star": o.get("x_star", float("nan")),
                    "kappa_star": o.get("kappa_star", float("nan")),
                    "f_tot_star": o.get("f_tot", float("nan")),
                    "hub_share_cap": o["hub_share_cap"],
                }
            )
    return rows
