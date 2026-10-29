# -*- coding: utf-8 -*-
"""在**同一 (N, M)** 预算下，把双峰构造与五类主流拓扑生成器做公平对比。

为什么必须固定 (N, M)
--------------------
拓扑鲁棒性的比较只有在"同样的节点数、同样的边数"下才有意义：
边数决定了总度预算 <k>N，任何多一条边的图都天然更抗打击。本脚本对每个
数据集都**恰好**生成 N 个节点、M 条边，并逐图断言（见 ``check_budget``）。

对比对象
--------
ours 分支：
  * ``bimodal_1hub``  书中理论点：n_h=1，k_max = A·N^(2/3)（clamp 到残差上限）
  * ``bimodal_ftot``  解析式选点：在族 B(n_h) 上取 argmax f_tot（不经实测 R，
                      因此没有"用测试指标调参"的泄漏）

基线分支（全部同 N、同 M、连通）：
  * ``random_connected``  随机连通图 = 随机生成树 + 随机弦（⟨k⟩<3 时 ER 必不连通，
                          这是"同预算下随机图"的正确对照）
  * ``ba``                优先连接（BA），边数用随机加/删边补齐
  * ``rand_regular``      近正则：k=⌊2M/N⌋ 正则 + 随机弦
  * ``config_swap``       与 bimodal_1hub **同度序列**、随机重连（保留度序列、
                          破坏"hub + 环"的空间排布）→ 分离"度分布"与"排布"
  * ``original_swap``     与原始拓扑同度序列、随机重连（原图自己的零模型）

参考（不作为待超越的方法）：``original`` 真实拓扑。

指标
----
  R_random           随机移除下的 LCC 面积（5 次洗牌取均值）
  R_targeted_deg     按初始度数降序移除的 LCC 面积
  R_targeted_adapt   每步重算最高度移除的 LCC 面积（自适应攻击）
  R_bar = mean(R_random, R_targeted_deg)   与既往结果口径一致
  L_bar              平均最短路
  J = R_bar / L_bar  效率-鲁棒性联合指标

用法
----
    python scripts/topology_vs_baselines.py --seeds 5
    python scripts/topology_vs_baselines.py --datasets GtsCe --seeds 2
"""

from __future__ import annotations

import argparse
import heapq
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
    build_bimodal,
    construct_bimodal_sequence,
    max_bimodal_hubs,
)
from scripts.theory_range import ftot  # noqa: E402

OUT_DIR = ROOT / "results" / "topology_vs_baselines"

OUR_METHODS = ("bimodal_1hub", "bimodal_ftot")
BASELINES = ("random_connected", "ba", "rand_regular", "config_swap", "original_swap")


# ---------------------------------------------------------------------------
# 1. 拓扑生成（严格同 (N, M)）
# ---------------------------------------------------------------------------

def check_budget(g: nx.Graph, n: int, m: int) -> None:
    assert g.number_of_nodes() == n, f"节点数不符: {g.number_of_nodes()} != {n}"
    assert g.number_of_edges() == m, f"边数不符: {g.number_of_edges()} != {m}"


def _add_random_edges(g: nx.Graph, target_m: int, seed: int) -> nx.Graph:
    """随机加边到恰好 target_m（不引入重边/自环）。"""
    rng = random.Random(seed)
    n = g.number_of_nodes()
    nodes = list(g.nodes())
    guard = 0
    while g.number_of_edges() < target_m:
        guard += 1
        if guard > 200 * target_m + 10_000:
            raise RuntimeError("加边失败：目标边数可能超过完全图上限")
        u, v = rng.sample(nodes, 2)
        if not g.has_edge(u, v):
            g.add_edge(u, v)
    return g


def _drop_random_edges(g: nx.Graph, target_m: int, seed: int) -> nx.Graph:
    """随机删边到恰好 target_m，且尽量保持连通（优先删非桥边）。"""
    rng = random.Random(seed)
    edges = list(g.edges())
    rng.shuffle(edges)
    for u, v in edges:
        if g.number_of_edges() <= target_m:
            break
        g.remove_edge(u, v)
        if not nx.is_connected(g):
            g.add_edge(u, v)
    if g.number_of_edges() != target_m:
        raise RuntimeError("删边失败：无法在保持连通的前提下达到目标边数")
    return g


def random_connected(n: int, m: int, seed: int) -> nx.Graph:
    """随机连通图：随机标签树（保证连通）+ 随机弦（补足边数）。"""
    g = nx.random_labeled_tree(n, seed=seed)
    g = nx.convert_node_labels_to_integers(g)
    return _add_random_edges(g, m, seed)


def ba_graph(n: int, m: int, seed: int) -> nx.Graph:
    """BA，选最接近目标边数的附着参数，再用随机加/删边精确对齐。"""
    best_k, best_err = 1, abs(1 * (n - 1) - m)
    for k in range(1, max(2, m // n + 3)):
        if k >= n:
            break
        err = abs(k * (n - k) - m)
        if err < best_err:
            best_k, best_err = k, err
    g = nx.barabasi_albert_graph(n, best_k, seed=seed)
    g = nx.convert_node_labels_to_integers(g)
    if g.number_of_edges() < m:
        g = _add_random_edges(g, m, seed)
    elif g.number_of_edges() > m:
        g = _drop_random_edges(g, m, seed)
    return g


def rand_regular(n: int, m: int, seed: int) -> nx.Graph:
    """近正则：k = ⌊2M/N⌋，再补随机弦到 M。

    k=2 的"正则图"必然是若干个互不相交的环（不连通），因此退化为
    "单个环 + 随机弦"——这正是同预算下最朴素的无 hub 正则拓扑。
    """
    k = max(2, int(2 * m // n))
    for cand in (k, k - 1, k + 1):
        if cand < 2 or cand >= n:
            continue
        if (n * cand) % 2 == 0:
            k = cand
            break
    if k <= 2:
        g = nx.cycle_graph(n)
    else:
        for attempt in range(50):
            g = nx.random_regular_graph(k, n, seed=seed + attempt)
            if nx.is_connected(g):
                break
        else:
            g = nx.cycle_graph(n)
    g = nx.convert_node_labels_to_integers(g)
    if g.number_of_edges() <= m:
        g = _add_random_edges(g, m, seed)
    else:
        g = _drop_random_edges(g, m, seed)
    return g


def swap_randomize(g: nx.Graph, seed: int, factor: int = 10) -> nx.Graph:
    """保度随机重连（double edge swap），保持 N、M、度序列与连通性。

    ``nx.double_edge_swap`` **不保证连通性**：若被交换的边含桥，可能把图切开。
    因此按块（≈M/10 次）推进，每块后校验连通性，断开则回滚该块并停止。
    返回图上的 ``graph['swap_done']`` 记录实际完成的重连次数。
    """
    h = g.copy()
    rng = random.Random(seed)
    target = factor * h.number_of_edges()
    chunk = max(1, h.number_of_edges() // 10)
    done = 0
    while done < target:
        snapshot = h.copy()
        try:
            nx.double_edge_swap(h, nswap=chunk, max_tries=chunk * 20,
                                seed=rng.randint(0, 10**9))
        except Exception:  # 度数集中 / 图过小时提前收敛
            h = snapshot
            break
        if not nx.is_connected(h):
            h = snapshot
            break
        done += chunk
    h.graph["swap_done"] = done
    h.graph["swap_target"] = target
    return h


# ---------------------------------------------------------------------------
# 2. 攻击曲线与指标
# ---------------------------------------------------------------------------

def lcc_curve(g: nx.Graph, order: List[int]) -> Tuple[np.ndarray, np.ndarray]:
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


def adaptive_degree_order(g: nx.Graph) -> List[int]:
    """自适应攻击：每步移除当前度数最大的节点（最大堆 + 惰性更新）。"""
    h = g.copy()
    deg = dict(h.degree())
    heap = [(-d, v) for v, d in deg.items()]
    heapq.heapify(heap)
    order: List[int] = []
    while heap:
        nd, v = heapq.heappop(heap)
        if not h.has_node(v):
            continue
        if -nd != h.degree(v):
            # 惰性删除：堆里的度数已过期，按当前度数重新入堆
            heapq.heappush(heap, (-h.degree(v), v))
            continue
        neighbours = list(h.neighbors(v))
        order.append(v)
        h.remove_node(v)
        for u in neighbours:
            if h.has_node(u):
                heapq.heappush(heap, (-h.degree(u), u))
    return order


def evaluate(g: nx.Graph, n_random: int = 5, seed: int = 2026) -> Dict[str, float]:
    n0 = g.number_of_nodes()
    nodes = list(g.nodes())
    rng = random.Random(seed)
    r_rand = []
    for _ in range(n_random):
        order = nodes[:]
        rng.shuffle(order)
        xs, ys = lcc_curve(g, order)
        r_rand.append(float(np.trapz(ys, xs)))

    deg = dict(g.degree())
    xs, ys = lcc_curve(g, sorted(nodes, key=lambda v: (-deg[v], v)))
    r_deg = float(np.trapz(ys, xs))

    xs, ys = lcc_curve(g, adaptive_degree_order(g))
    r_adapt = float(np.trapz(ys, xs))

    try:
        l_bar = float(nx.average_shortest_path_length(g))
    except Exception:
        l_bar = float("nan")

    r_random = float(np.mean(r_rand))
    return {
        "R_random": r_random,
        "R_random_std": float(np.std(r_rand)),
        "R_targeted_deg": r_deg,
        "R_targeted_adapt": r_adapt,
        "R_bar": 0.5 * (r_random + r_deg),
        "L_bar": l_bar,
        "assortativity": float(nx.degree_assortativity_coefficient(g)),
        "clustering": float(nx.average_clustering(g)),
    }


def graph_stats(g: nx.Graph) -> Dict[str, float]:
    n = g.number_of_nodes()
    deg = sorted((d for _, d in g.degree()), reverse=True)
    return {
        "k_max": deg[0],
        "k_min": deg[-1],
        "k_max_share": deg[0] / sum(deg),
        "degree_gini": _gini(deg),
    }


def _gini(values: List[float]) -> float:
    a = np.sort(np.asarray(values, dtype=float))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * idx - n - 1).dot(a) / (n * a.sum()))


# ---------------------------------------------------------------------------
# 3. 双峰族的解析选点（不碰实测 R）
# ---------------------------------------------------------------------------

def bimodal_ftot_choice(n: int, m: int, k_min: int = 2) -> int:
    """在族 B(n_h) 上取 argmax f_tot（书中 7.16 的解析式），返回 n_h。"""
    hi = max_bimodal_hubs(n, m, k_min=k_min)
    avg_k = 2.0 * m / n
    best, best_val = 1, -1.0
    for n_h in sorted({1, 2, 3, 4, 6, 8, 12, 16, 24, 32, hi}):
        if n_h < 1 or n_h > hi:
            continue
        spec = construct_bimodal_sequence(n, m, n_hubs=n_h, k_min=k_min)
        if not spec.feasible:
            continue
        val = ftot(avg_k, n_h / n, float(spec.k_max))["f_tot"]
        if val > best_val:
            best, best_val = n_h, val
    return best


def bimodal_oracle(n: int, m: int, seed: int, n_random: int,
                   k_min: int = 2) -> Optional[Tuple[float, nx.Graph, int]]:
    """族 B(n_h) 内按实测 R̄ 取最优 —— **oracle 上界**，不是可选方法。

    论文里用它回答"就算把 n_h 调到最好，双峰族能不能追上基线"。
    选点用了测试指标，因此只能作为上界报告，不能当作方法。
    """
    hi = max_bimodal_hubs(n, m, k_min=k_min)
    grid = sorted({1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, hi})
    best: Optional[Tuple[float, nx.Graph, int]] = None
    for n_h in grid:
        if n_h > hi or n_h < 1:
            continue
        try:
            g, _spec = build_bimodal(n, m, n_hubs=n_h, seed=seed, k_min=k_min)
        except Exception:
            continue
        if g is None:
            continue
        met = evaluate(g, n_random=n_random, seed=seed)
        if best is None or met["R_bar"] > best[0]:
            best = (met["R_bar"], g, n_h)
    return best


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------

def build_variants(g0: nx.Graph, seed: int, n_random: int = 5,
                   with_oracle: bool = True) -> Dict[str, nx.Graph]:
    n, m = g0.number_of_nodes(), g0.number_of_edges()
    out: Dict[str, nx.Graph] = {"original": g0.copy()}

    g1, spec = build_bimodal(n, m, n_hubs=1, seed=seed)
    out["bimodal_1hub"] = g1
    n_h_ft = bimodal_ftot_choice(n, m)
    if n_h_ft == 1:
        out["bimodal_ftot"] = g1.copy()
        out["bimodal_ftot"].graph["n_h"] = 1
    else:
        g2, spec2 = build_bimodal(n, m, n_hubs=n_h_ft, seed=seed)
        out["bimodal_ftot"] = g2
        out["bimodal_ftot"].graph["n_h"] = n_h_ft
    out["bimodal_1hub"].graph["n_h"] = 1

    if with_oracle:
        orc = bimodal_oracle(n, m, seed, n_random)
        if orc is not None:
            out["bimodal_oracle"] = orc[1]
            out["bimodal_oracle"].graph["n_h"] = orc[2]

    out["random_connected"] = random_connected(n, m, seed)
    out["ba"] = ba_graph(n, m, seed)
    out["rand_regular"] = rand_regular(n, m, seed)
    out["config_swap"] = swap_randomize(g1, seed)
    out["original_swap"] = swap_randomize(g0, seed)

    for name, g in out.items():
        check_budget(g, n, m)
        if not nx.is_connected(g):
            raise RuntimeError(f"{name} 不连通，无法在同等条件下比较")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="GtsCe,Cogentco,UsCarrier,Colt,Chinanet,BA-200,BA-500")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=2026)
    ap.add_argument("--no-oracle", action="store_true",
                    help="跳过族内 oracle 上界（省时间，但缺少'族的天花板'）")
    args = ap.parse_args()

    from src.topology.generators import load_graph

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    rows: List[Dict] = []
    t0 = time.time()

    for name in names:
        path = ROOT / "dataset" / "testdata" / f"{name}.gml"
        if not path.exists():
            print(f"[skip] {name}: 缺少 {path.name}")
            continue
        g0, _ = load_graph(str(path), verbose=False)
        g0 = nx.convert_node_labels_to_integers(g0)
        n, m = g0.number_of_nodes(), g0.number_of_edges()
        print(f"\n[{name}] N={n} M={m} <k>={2*m/n:.3f}  n_h(ftot)={bimodal_ftot_choice(n, m)}")

        for s in range(args.seeds):
            seed = args.seed0 + s
            variants = build_variants(g0, seed, n_random=args.n_random,
                                      with_oracle=not args.no_oracle)
            for method, g in variants.items():
                met = evaluate(g, n_random=args.n_random, seed=seed)
                st = graph_stats(g)
                row = {
                    "dataset": name, "method": method, "seed": seed,
                    "N": n, "M": m, "avg_k": round(2 * m / n, 4),
                    "n_h": int(g.graph.get("n_h", 0)),
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()},
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in met.items()},
                    "J": round(met["R_bar"] / met["L_bar"], 5) if met["L_bar"] else float("nan"),
                }
                rows.append(row)
                print(f"   seed={seed} {method:17s} R̄={met['R_bar']:.4f} "
                      f"(Rr={met['R_random']:.3f} Rd={met['R_targeted_deg']:.3f} "
                      f"Ra={met['R_targeted_adapt']:.3f}) L̄={met['L_bar']:.2f}")

        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / "raw_scan.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "raw_scan.csv", index=False)
    summarize(df)
    print(f"\n[done] {len(df)} 行，{time.time()-t0:.0f}s → {OUT_DIR}")


def summarize(df: pd.DataFrame) -> None:
    agg = (df.groupby(["dataset", "method"])
             .agg(N=("N", "first"), M=("M", "first"), avg_k=("avg_k", "first"),
                  R_bar=("R_bar", "mean"), R_bar_std=("R_bar", "std"),
                  R_random=("R_random", "mean"), R_targeted_deg=("R_targeted_deg", "mean"),
                  R_targeted_adapt=("R_targeted_adapt", "mean"),
                  L_bar=("L_bar", "mean"), J=("J", "mean"),
                  k_max=("k_max", "mean"), degree_gini=("degree_gini", "mean"),
                  n=("R_bar", "size"))
             .reset_index())
    agg.to_csv(OUT_DIR / "summary_by_method.csv", index=False)

    # 相对基线的配对比较（同一 seed 下逐点做差，避免小图噪声）
    rows = []
    for ds, sub in df.groupby("dataset"):
        piv = sub.pivot_table(index="seed", columns="method", values="R_bar")
        if "original" not in piv:
            continue
        for method in piv.columns:
            if method == "original":
                continue
            delta = (piv[method] - piv["original"]).dropna()
            if delta.empty:
                continue
            rows.append({
                "dataset": ds, "method": method,
                "R_bar": float(piv[method].mean()),
                "delta_vs_original": float(delta.mean()),
                "rel_pct": float((piv[method].mean() / piv["original"].mean() - 1) * 100),
                "wins": int((delta > 0).sum()), "n": int(len(delta)),
            })
    pd.DataFrame(rows).to_csv(OUT_DIR / "paired_vs_original.csv", index=False)

    # 排名表
    rank = agg.copy()
    rank["rank_in_dataset"] = rank.groupby("dataset")["R_bar"].rank(ascending=False, method="min")
    rank.to_csv(OUT_DIR / "ranking.csv", index=False)
    (OUT_DIR / "ranking.md").write_text(_to_md(rank), encoding="utf-8")


def _to_md(rank: pd.DataFrame) -> str:
    lines = ["# 同预算 (N, M) 下的拓扑鲁棒性排名", "",
             "R̄ = ½(R_random + R_targeted_deg)，随机攻击 5 次洗牌取均值；"
             "R_adapt 为每步重算最高度的自适应攻击。", ""]
    datasets = list(dict.fromkeys(rank["dataset"]))
    for ds in datasets:
        sub = rank[rank["dataset"] == ds].sort_values("R_bar", ascending=False)
        lines.append(f"## {ds}  (N={int(sub['N'].iloc[0])}, M={int(sub['M'].iloc[0])})")
        lines.append("")
        lines.append("| # | method | R̄ | R_random | R_deg | R_adapt | L̄ | k_max | gini |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(sub.itertuples(), start=1):
            lines.append(f"| {i} | {r.method} | {r.R_bar:.4f} | {r.R_random:.4f} | "
                         f"{r.R_targeted_deg:.4f} | {r.R_targeted_adapt:.4f} | "
                         f"{r.L_bar:.2f} | {r.k_max:.0f} | {r.degree_gini:.3f} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
