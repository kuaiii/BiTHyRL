# -*- coding: utf-8 -*-
"""服务域语义检验：**复制式**（replication）vs **划分式**（partition）。

两种语义
--------
复制式（当前 `src/metrics/metrics.py` 的实现）：
    一个连通分量里只要有一个控制器，该分量内**所有**交换机都由它服务；
    若分量内有 m 个控制器，则每个控制器的服务域 A_j 都等于该分量的交换机数，
    每个交换机能看到的控制器数 k_i = 分量内控制器个数。
      CSA = (1/|V_S|) Σ_i [1 − e^{−α k_i}]      —— 度量"控制路径冗余度"
      H_C = −Σ_j p_j ln p_j,  p_j = A_j / (未控 + Σ_j A_j)

划分式（partition）：
    每个交换机只归属**一个**主控制器，归属按"最近 + 容量均衡"分配；
    主控制器不可达 ⟹ 该交换机失去控制（无论其它控制器是否可达）。
      CSA = |{i : 归属控制器可达}| / |V_S|     —— 度量"覆盖率"，与 GCC 高度共线
      H_C = 各域负载的熵（p_j = A_j / |V_S|）  —— 才真正是"负载均衡度"

本脚本的目的不是选边，而是**量出结论对语义的敏感度**：
如果两种语义下"双峰 vs 基线"的排序一致，语义就不构成审稿风险，可以双口径并列报告；
如果不一致，就必须在论文里明确声明采用的是哪一种，并说明理由。

用法
----
    python scripts/service_domain_semantics.py --datasets GtsCe,BA-200
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.metrics import (  # noqa: E402
    calculate_control_entropy,
    calculate_csa,
    calculate_gcc,
)
from src.topology.bimodal_construction import build_bimodal, max_bimodal_hubs  # noqa: E402
from scripts.topology_vs_baselines import (  # noqa: E402
    rand_regular,
    random_connected,
)

OUT_DIR = ROOT / "results" / "service_domain_semantics"
ALPHA = 0.2


# ---------------------------------------------------------------------------
# 1. 划分式语义
# ---------------------------------------------------------------------------

def hop_distances(g: nx.Graph, centers: Sequence[int]) -> Dict[int, Dict[int, int]]:
    return {c: nx.single_source_shortest_path_length(g, c) for c in centers}


def partition_domains(g: nx.Graph, centers: Sequence[int]) -> Dict[int, List[int]]:
    """最近 + 容量均衡的划分：返回 {controller: [switch, ...]}。

    规则（与 SDN 部署一致，且与拓扑无关，保证可比）：
      1. 每个交换机先归最近控制器（跳数，多源 BFS）；
      2. 若某控制器超额（cap = ⌈|V_S|/m⌉），按"距离近者优先"改派到次近的、仍有余量的
         控制器；仍无处可去则并入当前负载最小的域。
    """
    centers = list(centers)
    switches = [v for v in g.nodes() if v not in set(centers)]
    if not switches or not centers:
        return {c: [] for c in centers}

    dist = hop_distances(g, centers)
    inf = float("inf")
    nearest = {}
    for v in switches:
        best_c, best_d = None, inf
        for c in centers:
            d = dist[c].get(v, inf)
            if d < best_d:
                best_c, best_d = c, d
        nearest[v] = (best_c, best_d)

    cap = math.ceil(len(switches) / len(centers))
    load = {c: 0 for c in centers}
    domains = {c: [] for c in centers}

    for v in sorted(switches, key=lambda x: (nearest[x][1], x)):
        order = sorted(centers, key=lambda c: (dist[c].get(v, inf), c))
        chosen = None
        for c in order:
            if load[c] < cap:
                chosen = c
                break
        if chosen is None:  # 全部满员（理论上不会发生）
            chosen = min(order, key=lambda c: load[c])
        domains[chosen].append(v)
        load[chosen] += 1
    return domains


def metrics_partition(g: nx.Graph, centers: Sequence[int]) -> Dict[str, float]:
    centers = list(centers)
    centers_set = set(centers)
    switches = [v for v in g.nodes() if v not in centers_set]
    n_s = len(switches)
    if n_s == 0:
        return {"CSA": 0.0, "H_C": 0.0, "served": 0, "max_share": 0.0}

    domains = partition_domains(g, centers)
    # 主控制器可达性：控制器必须存活，且与交换机同属一个连通分量
    comp = {}
    for i, c in enumerate(nx.connected_components(g)):
        for v in c:
            comp[v] = i
    served = 0
    load = {}
    for c, members in domains.items():
        alive = g.has_node(c)
        cnt = 0
        for v in members:
            if alive and comp.get(v) == comp.get(c):
                cnt += 1
        load[c] = cnt
        served += cnt

    csa = served / n_s
    total = sum(load.values())
    if total == 0:
        h_c = 0.0
    else:
        p = np.array([x / total for x in load.values() if x > 0], dtype=float)
        h_c = float(-(p * np.log(p)).sum())
    max_share = max(load.values()) / total if total else 0.0
    return {"CSA": float(csa), "H_C": h_c, "served": served, "max_share": float(max_share)}


# ---------------------------------------------------------------------------
# 2. 复制式语义（直接调用现有实现，保证与论文口径一致）
# ---------------------------------------------------------------------------

def metrics_replication(g: nx.Graph, centers: Sequence[int],
                        n_switches0: int) -> Dict[str, float]:
    centers = list(centers)
    csa = calculate_csa(g, centers, alpha=ALPHA, initial_num_switches=n_switches0)
    h_c = calculate_control_entropy(g, centers)
    # 复制式的"重叠度"：Σ_j A_j / |V_S|，>1 表示同一批交换机被多个控制器重复计入
    centers_set = set(centers)
    switches = [v for v in g.nodes() if v not in centers_set]
    overlap = 0
    for comp in nx.connected_components(g):
        m_in = len(comp & centers_set)
        overlap += m_in * len([v for v in comp if v not in centers_set])
    return {"CSA": float(csa), "H_C": float(h_c),
            "overlap": overlap / len(switches) if switches else 0.0}


# ---------------------------------------------------------------------------
# 3. 状态序列：初始 / 摘掉头号 hub / 5% 定点打 / 10% 随机
# ---------------------------------------------------------------------------

def attack_states(g: nx.Graph) -> Dict[str, nx.Graph]:
    """三个状态，全部保留原始节点标签，便于控制器身份对齐。"""
    states = {"initial": g.copy()}
    deg = dict(g.degree())
    top1 = max(deg, key=lambda v: (deg[v], -v))
    h = g.copy()
    h.remove_node(top1)
    states["after_top1"] = h

    h = g.copy()
    for v in sorted(deg, key=lambda v: -deg[v])[: max(1, round(0.05 * g.number_of_nodes()))]:
        if h.has_node(v):
            h.remove_node(v)
    states["after_deg5pct"] = h
    return states


def place_by_degree(g: nx.Graph, ratio: float = 0.1) -> List[int]:
    k = max(1, round(ratio * g.number_of_nodes()))
    deg = dict(g.degree())
    return sorted(g.nodes(), key=lambda v: (-deg[v], v))[:k]


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------

def evaluate_graph(name: str, dataset: str, g: nx.Graph) -> List[Dict]:
    centers0 = place_by_degree(g, 0.1)
    n_s0 = g.number_of_nodes() - len(centers0)
    return [_one_state(name, dataset, state_name, g, h, centers0, n_s0)
            for state_name, h in attack_states(g).items()]


def _one_state(name, dataset, state_name, g0, h, centers0, n_s0) -> Dict:
    # 把控制器标签映射到状态图（attack_states 里做了整数重编号，这里统一用原图标签）
    centers = [c for c in centers0 if c in h]
    rep = metrics_replication(h, centers, n_s0) if centers else {"CSA": 0.0, "H_C": 0.0, "overlap": 0.0}
    par = metrics_partition(h, centers) if centers else {"CSA": 0.0, "H_C": 0.0, "served": 0, "max_share": 0.0}
    return {
        "dataset": dataset, "topology": name, "state": state_name,
        "n_alive": h.number_of_nodes(),
        "gcc_frac": calculate_gcc(h, centers) / g0.number_of_nodes() if centers else 0.0,
        "CSA_rep": rep["CSA"], "H_C_rep": rep["H_C"], "overlap_rep": rep.get("overlap", 0.0),
        "CSA_par": par["CSA"], "H_C_par": par["H_C"], "max_share_par": par.get("max_share", 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="GtsCe,BA-200")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    from src.topology.generators import load_graph

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    for ds in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        path = ROOT / "dataset" / "testdata" / f"{ds}.gml"
        if not path.exists():
            print(f"[skip] {ds}")
            continue
        g0, _ = load_graph(str(path), verbose=False)
        g0 = nx.convert_node_labels_to_integers(g0)
        n, m = g0.number_of_nodes(), g0.number_of_edges()

        variants = {"original": g0.copy(), "rand_regular": rand_regular(n, m, args.seed),
                    "random_connected": random_connected(n, m, args.seed)}
        variants["bimodal_1hub"], _ = build_bimodal(n, m, n_hubs=1, seed=args.seed)
        hi = max_bimodal_hubs(n, m)
        nh = max(2, min(hi, 12))
        variants["bimodal_weak"], _ = build_bimodal(n, m, n_hubs=nh, seed=args.seed)

        print(f"\n[{ds}] N={n} M={m}")
        for name, g in variants.items():
            for r in evaluate_graph(name, ds, g):
                rows.append(r)
            sub = [r for r in rows if r["dataset"] == ds and r["topology"] == name]
            print(f"  {name:17s} " + " | ".join(
                f"{s['state']}: CSA_rep={s['CSA_rep']:.3f} CSA_par={s['CSA_par']:.3f} "
                f"HC_rep={s['H_C_rep']:.2f} HC_par={s['H_C_par']:.2f}" for s in sub))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "semantics_compare.csv", index=False)
    (OUT_DIR / "semantics_compare.md").write_text(_to_md(df), encoding="utf-8")
    print(f"\n[done] {len(df)} 行 → {OUT_DIR}")


def _to_md(df: pd.DataFrame) -> str:
    lines = ["# 服务域语义：复制式 vs 划分式", ""]
    for ds in dict.fromkeys(df["dataset"]):
        sub = df[df["dataset"] == ds]
        lines.append(f"## {ds}")
        lines.append("")
        lines.append("| 拓扑 | 状态 | GCC | CSA(复制) | CSA(划分) | H_C(复制) | H_C(划分) | 复制重叠度 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sub.itertuples():
            lines.append(f"| {r.topology} | {r.state} | {r.gcc_frac:.3f} | {r.CSA_rep:.3f} | "
                         f"{r.CSA_par:.3f} | {r.H_C_rep:.2f} | {r.H_C_par:.2f} | {r.overlap_rep:.2f} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
