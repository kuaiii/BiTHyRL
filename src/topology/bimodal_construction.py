# -*- coding: utf-8 -*-
"""双峰 / 类双峰拓扑的严格构造实现。

术语区分（对应审稿意见 Q2 末句 "The theoretical two-degree prior should be
distinguished from the graph ultimately evaluated"）：

* **理论先验 (theoretical prior)** —— 书中 k_max = A(<k>)·N^(2/3) 给出的目标度
  分布。它是一个理想目标，既不一定能在给定 (N, M) 下实现，也不保证残余低度层
  可连通。对应 :class:`BimodalSpec` 的 ``theory_*`` 字段。
* **可评估图 (evaluated graph)** —— 实际生成、并进入仿真的简单图。本模块保证
  四个不变量：

  I1  |V| = N
  I2  |E| = M                       （度序列和恒为 2M，见 ``construct_bimodal_sequence``）
  I3  图连通
  I4  去掉 hub 层后，残余低度层仍然连通，且其最小度 ≥ k_min ≥ 2

  I4 是"定向攻击下不一步崩"的必要条件，也是旧实现（先加边修连通、再删边回补
  边数）被系统性破坏的那一条。对应 ``realised_*`` 字段。

为什么 I4 是必要的
------------------
设 N 个节点、M 条边，n_h 个 hub、每个度 k_max，且 hub 只与低度节点相连。
记 e_ll 为低度层内部边数，则

    M = n_h·k_max + e_ll

残余低度层有 n_low = N − n_h 个节点，要连通至少需 n_low − 1 条边；要求其最小度
≥ 2（否则摘掉 hub 后会出现叶子，再被定向攻击逐个点掉）则需

    e_ll ≥ n_low    ⟹    k_max ≤ (M − N + n_h) / n_h          (★)

``residual_cap`` 就是 (★)。n_h = 1 时即 k_max ≤ M − N + 1 —— 真实 SDN 拓扑
（N < 200, <k> ≈ 2.3–2.6）代入后远小于 A·N^(2/3)，这正是"理论先验不可行"的
定量根源，见 ``applicability`` 与 :func:`theory_applicability_report`。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

__all__ = [
    "BimodalSpec",
    "coef_A",
    "theory_k_max",
    "residual_cap",
    "applicability",
    "construct_bimodal_sequence",
    "repair_topology",
    "build_bimodal",
    "create_quasi_bimodal",
    "scan_hub_counts",
    "theory_applicability_report",
]


# ---------------------------------------------------------------------------
# 1. 解析部分：A(<k>)、理论 k_max、残差上限、适用范围
# ---------------------------------------------------------------------------

def coef_A(avg_k: float) -> float:
    """公式中的系数 A = [ 2·<k>²·(<k>−1)² / (2<k>−1) ]^(1/3)。

    与 ``reconstruction.create_bimodal_theoretical`` 中的写法保持一致，
    以便历史结果可比。
    """
    if avg_k <= 1.0:
        raise ValueError(f"avg_k 必须 > 1，收到 {avg_k}")
    return (2.0 * avg_k ** 2 * (avg_k - 1.0) ** 2 / (2.0 * avg_k - 1.0)) ** (1.0 / 3.0)


def theory_k_max(n: int, m: int) -> int:
    """理论先验 k_max = round(A·N^(2/3))，并夹到 [2, N−1]。"""
    avg_k = 2.0 * m / n
    k = int(round(coef_A(avg_k) * n ** (2.0 / 3.0)))
    return max(2, min(k, n - 1))


def residual_cap(n: int, m: int, n_hubs: int = 1, k_min: int = 2) -> int:
    """由残差约束 (★) 反解出的 k_max 上限。

    k_max ≤ (M − N + n_h) / n_h，同时受"低度层平均度 ≥ k_min"与简单图 k ≤ N−1 约束。
    返回可能为 0，表示该 n_hubs 下不可行。
    """
    n_low = n - n_hubs
    if n_low < 3 or n_hubs < 1:
        return 0
    # (★) 残差层最小度 ≥ k_min ≥ 2  ⟺  e_ll ≥ n_low·k_min/2，k_min=2 时即 e_ll ≥ n_low
    cap_residual = (m - n_low) // n_hubs if k_min == 2 else int((2 * m - k_min * n_low) / (2 * n_hubs))
    cap_budget = (2 * m - k_min * n_low) // n_hubs
    return max(0, min(cap_residual, cap_budget, n - 1))


def applicability(n: int, m: int, k_min: int = 2) -> Dict[str, float]:
    """判断单 hub 双峰理论先验在给定 (N, M) 下是否可实现。

    单 hub 可行 ⟺ A·N^(2/3) ≤ M − N + 1（(★) 在 n_h=1 的情形）。
    把 M = <k>N/2 代入并忽略 O(1) 项，得闭式阈值

        N ≳ N_min = [ A(<k>) / (<k>/2 − 1) ]³                    (◆)

    即：**书中双峰最优构造只对"足够大且足够稠密"的网络可行**；<k> 越接近 2，
    N_min 增长得越陡（<k>=2.31 时 N_min≈1.4×10³，<k>=2.59 时≈3.2×10²）。
    """
    avg_k = 2.0 * m / n
    A = coef_A(avg_k) if avg_k > 1 else float("inf")
    raw = A * n ** (2.0 / 3.0) if math.isfinite(A) else float("inf")
    k_theory_raw = int(round(raw))
    k_theory = max(2, min(k_theory_raw, n - 1))
    cap1 = residual_cap(n, m, n_hubs=1, k_min=k_min)
    # 可行 <=> 理论值既没撞 N−1 上限，也没撞残差上限
    clamp = "none"
    if k_theory_raw > n - 1:
        clamp = "n_minus_1"
    elif k_theory_raw > cap1:
        clamp = "residual"
    feasible = clamp == "none"
    slope = avg_k / 2.0 - 1.0
    n_min = float("inf") if slope <= 0 else (A / slope) ** 3
    # 在 (★) 下允许的最大 hub 数：k_max ≥ 2 至少需 n_h ≤ (M − N)/1 + 1
    n_h_max = max(1, m - n + 2)
    return {
        "n": n,
        "m": m,
        "avg_k": avg_k,
        "A": A,
        "k_max_theory_raw": k_theory_raw,
        "k_max_theory": k_theory,
        "k_max_cap_single_hub": cap1,
        "clamp": clamp,
        "single_hub_feasible": feasible,
        "n_min_theory": n_min,
        "n_over_n_min": n / n_min if math.isfinite(n_min) else float("inf"),
        "k_min": k_min,
        "n_hubs_max": n_h_max,
    }


def theory_applicability_report(
    graphs: Dict[str, Tuple[int, int]], k_min: int = 2
) -> List[Dict[str, object]]:
    """批量产出适用范围表（供文档与脚本直接引用）。"""
    rows = []
    for name, (n, m) in graphs.items():
        info = applicability(n, m, k_min=k_min)
        info["name"] = name
        info["regime"] = "理论区内" if info["single_hub_feasible"] else "理论区外"
        rows.append(info)
    return rows


# ---------------------------------------------------------------------------
# 2. 度序列：显式残差分摊 + 奇偶自动成立 + 可图化修复
# ---------------------------------------------------------------------------

@dataclass
class BimodalSpec:
    """一个 (N, M, n_h) 下的构造规格。理论先验与可实现值分开记录。"""

    n: int
    m: int
    n_hubs: int
    k_min: int = 2
    theory_k_max: int = 0
    k_max: int = 0
    k_low_base: int = 0
    k_low_remainder: int = 0
    degree_sequence: List[int] = field(default_factory=list)
    feasible: bool = True
    strategy: str = ""          # theory / capped / infeasible
    graphical: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def n_low(self) -> int:
        return self.n - self.n_hubs

    @property
    def low_total_degree(self) -> int:
        return sum(self.degree_sequence[self.n_hubs:])

    @property
    def realised_k_low_mean(self) -> float:
        return self.low_total_degree / self.n_low if self.n_low else 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "n": self.n, "m": self.m, "n_hubs": self.n_hubs,
            "theory_k_max": self.theory_k_max, "k_max": self.k_max,
            "k_low_mean": round(self.realised_k_low_mean, 3),
            "sum_degree": sum(self.degree_sequence),
            "feasible": self.feasible, "graphical": self.graphical,
            "strategy": self.strategy,
        }


def _is_graphical(seq: Sequence[int]) -> bool:
    try:
        return bool(nx.is_graphical(list(seq), method="eg"))
    except Exception:
        return False


def construct_bimodal_sequence(
    n: int, m: int, n_hubs: int = 1, k_min: int = 2, k_max_override: Optional[int] = None
) -> BimodalSpec:
    """CONSTRUCTBIMODALSEQUENCE —— 显式构造双峰度序列。

    与旧实现的差别（正是 Q2 指出的 495 vs 588、奇数和问题）：

    1. **k_min 不是输入，而是下界**。旧代码先取 k_min = ⌊(2M − k_max)/(N−1)⌋
       再据此拼序列，于是总和只有 k_max + k_min·(N−1) = 495 < 2M = 588 且为奇数。
       本实现的流程相反：先定 k_max，再把**全部**剩余度数 2M − n_h·k_max 均摊到
       N − n_h 个低度节点上，取 base = ⌊·⌋、余数 rem 个节点取 base+1。由构造

           Σ = n_h·k_max + base·n_low + rem = 2M

       恒成立，且 2M 为偶数，**奇偶性无需额外修正**（旧代码的"奇数和"是截断造成的，
       不是度序列本身的性质）。

    2. k_max 先取理论先验，再与残差上限 (★) 取小；被夹住时记 ``strategy='capped'``，
       并把两个值分别留在 ``theory_k_max`` / ``k_max`` 里，供论文区分陈述。

    3. 若序列不可图化（Erdős–Gallai 失败），以 k_max ← k_max−1 逐步回退、把省下的
       度数重新均摊到低度层，**保持 Σ = 2M 与双峰形态不变**，而不是把序列抹平成近正则。

    Returns:
        :class:`BimodalSpec`。``feasible=False`` 表示该 n_hubs 下无解。
    """
    spec = BimodalSpec(n=n, m=m, n_hubs=n_hubs, k_min=k_min)
    n_low = n - n_hubs
    if n_low < 3:
        spec.feasible = False
        spec.strategy = "infeasible"
        spec.notes.append(f"低度层只剩 {n_low} 个节点，无法构造")
        return spec
    if 2 * m < k_min * n:
        spec.feasible = False
        spec.strategy = "infeasible"
        spec.notes.append(f"<k>=2M/N={2*m/n:.2f} < k_min={k_min}，任何图都不满足最小度约束")
        return spec

    spec.theory_k_max = theory_k_max(n, m)
    cap = residual_cap(n, m, n_hubs, k_min)
    if cap < k_min:
        spec.feasible = False
        spec.strategy = "infeasible"
        spec.notes.append(f"n_h={n_hubs} 时残差上限 {cap} < k_min={k_min}")
        return spec

    k_max = min(spec.theory_k_max, cap)
    if k_max < spec.theory_k_max:
        spec.strategy = "capped"
        spec.notes.append(
            f"理论 k_max={spec.theory_k_max} 超出残差上限 {cap}，取 k_max={k_max}"
        )
    else:
        spec.strategy = "theory"

    if k_max_override is not None:
        # 仅供理论验证扫描使用：显式指定 k_max，越界时直接判不可行，不做静默截断
        if not (k_min <= k_max_override <= cap):
            spec.feasible = False
            spec.strategy = "infeasible"
            spec.notes.append(f"k_max_override={k_max_override} 超出 [{k_min}, {cap}]")
            return spec
        k_max = k_max_override
        spec.strategy = "override"

    # 逐步回退直到可图化
    for _ in range(max(1, k_max - k_min + 1)):
        low_total = 2 * m - n_hubs * k_max
        if low_total < k_min * n_low:
            break
        base = low_total // n_low
        rem = low_total % n_low
        seq = [k_max] * n_hubs + [base + 1] * rem + [base] * (n_low - rem)
        assert sum(seq) == 2 * m, "度序列和必须等于 2M"
        if _is_graphical(seq):
            spec.k_max = k_max
            spec.k_low_base = base
            spec.k_low_remainder = rem
            spec.degree_sequence = seq
            spec.graphical = True
            spec.feasible = True
            return spec
        k_max -= 1
        spec.notes.append(f"k_max 回退至 {k_max}：上一个序列不可图化")

    spec.feasible = False
    spec.strategy = "infeasible"
    spec.notes.append("在 [k_min, cap] 内找不到可图化的双峰度序列")
    return spec


# ---------------------------------------------------------------------------
# 3. 拓扑修复：保度 2-switch（不再"先加边再删边"）
# ---------------------------------------------------------------------------

def repair_topology(
    G: nx.Graph,
    target_m: Optional[int] = None,
    seed: Optional[int] = None,
    max_attempts: int = 20000,
    randomize_swaps: Optional[int] = None,
) -> nx.Graph:
    """REPAIRTOPOLOGY —— 只做保度 2-switch 的连通修复。

    旧链路（``_ensure_connectivity`` → ``_adjust_edge_count``）的问题是：加边修连通时
    把 hub 度数顶高 1.7 倍，删边回补边数时又只守 ``degree > 1``，把低度层从 2 打到 1。
    两个目标互相抵消，最终 k_min 与 M 都变了。

    本函数保持**度数不变、边数不变**（2-switch 的定义即如此），因此 I1–I4 里除了
    连通性之外不需要再修任何东西。若 2-switch 无法在尝试次数内连通，返回原图并
    由调用方决定是否接受（不会偷偷改边数）。

    Args:
        G: 输入图（原地修改并返回）。
        target_m: 若给定且不等于 |E|，说明输入的度序列本身坏了，直接报错，
            而不是悄悄加删边。
        randomize_swaps: 修复连通后再做多少次随机 2-switch 打散 Havel–Hakimi 的
            强结构性（默认 5·|E|）。
    """
    if seed is not None:
        random.seed(seed)
    if target_m is not None and G.number_of_edges() != target_m:
        raise ValueError(
            f"repair_topology 要求 |E| 已等于目标 {target_m}，实际 {G.number_of_edges()}；"
            "边数不对说明度序列构造有误，不应在此处修补"
        )
    if G.number_of_nodes() < 2:
        return G

    # --- 先打散 Havel–Hakimi 的确定性结构（保度，但可能断连，故放在修复之前） ---
    nswap = randomize_swaps if randomize_swaps is not None else 5 * G.number_of_edges()
    if nswap > 0 and G.number_of_edges() > 2:
        try:
            nx.double_edge_swap(G, nswap=nswap, max_tries=nswap * 10, seed=seed)
        except nx.NetworkXAlgorithmError:
            # 图太小或可实现性不足，跳过打散即可，不影响不变量
            pass

    # --- 连通修复（保度） ---
    if not nx.is_connected(G):
        _connect_by_swaps(G, max_attempts=max_attempts)

    return G


def _connect_residual_by_swaps(
    G: nx.Graph,
    hubs: Sequence,
    max_attempts: int = 4000,
    seed: Optional[int] = None,
) -> bool:
    """只在**低度层内部**做 2-switch，把残余层拼成连通。

    这是与整图连通修复分开的一步，原因很关键：整图可以靠 hub 把若干个低度层分量
    "串"起来而显得连通，但摘掉 hub 后立刻碎裂——旧实现正是在这里失守，也是
    实验里 ``gcc_after_top1`` 掉到 0.04 的成因。

    2-switch 只动低度层内部的边，因此：hub 的度数不变、低度层每个节点的内部度不变、
    总边数不变，唯一改变的是低度层的连通结构。
    """
    if seed is not None:
        random.seed(seed)
    hub_set = set(hubs)
    low = [v for v in G.nodes() if v not in hub_set]
    if len(low) < 3:
        return True
    low_set = set(low)

    for _ in range(max_attempts):
        H = G.subgraph(low_set)
        comps = list(nx.connected_components(H))
        if len(comps) <= 1:
            return True
        comps.sort(key=len, reverse=True)
        main, small = comps[0], comps[1]

        cand_small = [(u, v) for u, v in H.edges(small) if u in small and v in small]
        cand_main = [(x, y) for x, y in H.edges(main) if x in main and y in main]
        if not cand_small or not cand_main:
            return False

        merged = False
        for _ in range(30):
            u, v = random.choice(cand_small)
            x, y = random.choice(cand_main)
            if len({u, v, x, y}) < 4:
                continue
            if random.random() < 0.5:
                x, y = y, x
            # 新边必须不是已有的、也不能造成自环（低度层内不含 hub，故只需查 G）
            if G.has_edge(u, x) or G.has_edge(v, y):
                continue
            G.remove_edge(u, v)
            G.remove_edge(x, y)
            G.add_edge(u, x)
            G.add_edge(v, y)
            merged = True
            break
        if not merged:
            return False

    return nx.is_connected(G.subgraph(low_set))


def _connect_by_swaps(G: nx.Graph, max_attempts: int = 20000) -> bool:
    """用 2-switch 把若干个连通分量拼成一个（度数、边数均不变）。

    每次取"小分量内的一条边 (u,v)"与"主分量内的一条边 (x,y)"，重连为
    (u,x),(v,y)：两个分量被合并，度数不变。为降低冲突率，候选边集每次重新抽样
    而不是全量扫描。
    """
    for _ in range(max_attempts):
        comps = list(nx.connected_components(G))
        if len(comps) <= 1:
            return True
        comps.sort(key=len, reverse=True)
        main = comps[0]
        small = comps[1]

        cand_small = [(u, v) for u, v in G.edges(small) if u in small and v in small]
        cand_main = [(x, y) for x, y in G.edges(main) if x in main and y in main]
        if not cand_small or not cand_main:
            return False

        for _ in range(20):
            u, v = random.choice(cand_small)
            x, y = random.choice(cand_main)
            if len({u, v, x, y}) < 4:
                continue
            if random.random() < 0.5:
                x, y = y, x
            if G.has_edge(u, x) or G.has_edge(v, y):
                continue
            G.remove_edge(u, v)
            G.remove_edge(x, y)
            G.add_edge(u, x)
            G.add_edge(v, y)
            break
    return nx.is_connected(G)


# ---------------------------------------------------------------------------
# 4. 生成图
# ---------------------------------------------------------------------------

def build_bimodal(
    n: int,
    m: int,
    n_hubs: int = 1,
    seed: int = 42,
    k_min: int = 2,
    max_resample: int = 8,
    k_max_override: Optional[int] = None,
) -> Tuple[nx.Graph, BimodalSpec]:
    """按 :func:`construct_bimodal_sequence` 的规格生成图，并校验 I1–I4。

    失败时自动换种子重采样，仍失败则抛异常——绝不返回一个偷偷改了 N/M/k_min 的图。
    """
    spec = construct_bimodal_sequence(
        n, m, n_hubs=n_hubs, k_min=k_min, k_max_override=k_max_override
    )
    if not spec.feasible:
        raise ValueError(f"该 (N={n}, M={m}, n_h={n_hubs}) 下无可行双峰构造：{spec.notes}")

    last_err = None
    for attempt in range(max_resample):
        s = seed + attempt
        H = nx.havel_hakimi_graph(spec.degree_sequence)
        # Havel–Hakimi 按序列顺序赋度：节点 0..n_h−1 即 hub 层
        hubs = list(range(n_hubs))
        try:
            repair_topology(H, target_m=m, seed=s)
        except ValueError as exc:
            last_err = exc
            continue
        if not nx.is_connected(H):
            last_err = RuntimeError("整图连通修复失败")
            continue
        # 残余层连通（摘掉 hub 后仍连通）——旧实现在这一步缺失
        if not _connect_residual_by_swaps(H, hubs, seed=s):
            last_err = RuntimeError("残余层连通修复失败")
            continue
        if not _check_invariants(H, n, m, spec, k_min, hubs=hubs):
            last_err = RuntimeError(
                f"不变量校验失败: N={H.number_of_nodes()} M={H.number_of_edges()} "
                f"connected={nx.is_connected(H)} k_min={min(d for _, d in H.degree())}"
            )
            continue
        H.graph["bimodal_spec"] = spec.as_dict()
        H.graph["bimodal_notes"] = spec.notes
        H.graph["hub_nodes"] = hubs
        H.graph["seed_used"] = s
        return H, spec

    raise RuntimeError(f"重采样 {max_resample} 次仍未得到满足 I1–I4 的图：{last_err}")


def _check_invariants(
    G: nx.Graph, n: int, m: int, spec: BimodalSpec, k_min: int, hubs: Optional[Sequence] = None
) -> bool:
    if G.number_of_nodes() != n or G.number_of_edges() != m:
        return False
    if not nx.is_connected(G):
        return False
    deg = dict(G.degree())
    if min(deg.values()) < k_min:
        return False
    # 残余层：去掉 hub 层后的诱导子图必须连通（I4）
    if hubs is None:
        hubs = sorted(deg, key=deg.get, reverse=True)[: spec.n_hubs]
    hub_set = set(hubs)
    rest = G.subgraph([v for v in G.nodes() if v not in hub_set])
    if rest.number_of_nodes() > 1 and not nx.is_connected(rest):
        return False
    # 双峰形态：hub 的度必须严格高于所有低度节点
    if hubs:
        max_low = max(deg[v] for v in G.nodes() if v not in hub_set)
        if min(deg[h] for h in hub_set) <= max_low:
            return False
    return True


def create_quasi_bimodal(
    n: int,
    m: int,
    n_hubs: int,
    seed: int = 42,
    k_min: int = 2,
) -> Tuple[nx.Graph, BimodalSpec]:
    """类双峰族 B(n_h)：把 n_h 作为参数，在 n_h=1（星状）与 n_h→N（近正则）之间插值。

    这是文献里"双峰不可行时的替代方案"的形式化：不是随便找一个 hub 数，而是在
    满足残差约束 (★) 的前提下，用**同一个双峰模板**张成的单参数族。n_h 增大时
    k_max 按 (M−N+n_h)/n_h 下降、低度层抬升，定向鲁棒性单调改善、随机鲁棒性缓慢
    下降，故平均指标存在内部最优——见 ``scan_hub_counts`` 与
    ``scripts/theory_verify_optimality.py``。
    """
    return build_bimodal(n, m, n_hubs=n_hubs, seed=seed, k_min=k_min)


def scan_hub_counts(n: int, m: int, k_min: int = 2) -> List[Dict[str, object]]:
    """枚举所有可行的 n_h，返回每个 n_h 的 (k_max, k_low, 是否可达理论先验)。"""
    info = applicability(n, m, k_min=k_min)
    out = []
    for n_h in range(1, int(info["n_hubs_max"]) + 1):
        spec = construct_bimodal_sequence(n, m, n_hubs=n_h, k_min=k_min)
        if not spec.feasible:
            continue
        out.append({
            "n_hubs": n_h,
            "k_max": spec.k_max,
            "k_low_mean": round(spec.realised_k_low_mean, 3),
            "hits_theory": spec.k_max == spec.theory_k_max,
            "max_deg_share": round(spec.k_max / (2 * m), 4),
        })
    return out
