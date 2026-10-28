# -*- coding: utf-8 -*-
"""
追踪双峰构造的后处理链，定位 k_min / 边数约束在哪一步被破坏。

create_bimodal_theoretical 的实际执行链（稀疏分支）：
    度序列设计 -> Havel-Hakimi -> _ensure_connectivity -> _adjust_edge_count

本脚本逐阶段记录 (m, k_min, 连通性)，用于说明：
  1) Havel-Hakimi 生成的图通常不连通；
  2) _ensure_connectivity 靠加边连通，破坏边数预算；
  3) _adjust_edge_count 删边回补边数时，只守 degree>1，把 k_min 从 2 降到 1。

输出：results/bimodal_diagnosis/stage_trace.csv
"""
import os
import sys
import warnings

import networkx as nx
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from src.topology.generators import load_graph
from src.topology.reconstruction import (
    _bimodal_degree_sequence_sparse,
    _is_low_density,
    _ensure_connectivity,
    _adjust_edge_count,
    _make_graphical_bimodal_preserving,
)

OUT = os.path.join(ROOT, "results", "bimodal_diagnosis", "stage_trace.csv")


def trace(dataset, path, seed=42):
    G, _ = load_graph(path)
    G = nx.convert_node_labels_to_integers(G)
    n, m = G.number_of_nodes(), G.number_of_edges()
    rows = []

    def rec(stage, H, extra=""):
        degs = [d for _, d in H.degree()]
        rows.append({
            "dataset": dataset,
            "stage": stage,
            "n": H.number_of_nodes(),
            "m": H.number_of_edges(),
            "delta_m": H.number_of_edges() - m,
            "k_min": min(degs) if degs else 0,
            "k_max": max(degs) if degs else 0,
            "connected": nx.is_connected(H) if degs else False,
            "num_isolated": sum(1 for d in degs if d == 0),
            "num_degree1": sum(1 for d in degs if d == 1),
            "note": extra,
        })

    rec("original", G.copy())
    seq = _bimodal_degree_sequence_sparse(n, m, 1, seed=seed)
    rec("degree_seq_design", nx.empty_graph(0), f"k_max={max(seq)},k_min={min(seq)},graphical={nx.is_graphical(seq)}")

    if nx.is_graphical(seq):
        H = nx.havel_hakimi_graph(seq)
    else:
        seq = _make_graphical_bimodal_preserving(seq, 1)
        H = nx.havel_hakimi_graph(seq)
    rec("havel_hakimi", H)

    if not nx.is_connected(H):
        added = _ensure_connectivity(H, seed=seed)
        rec("ensure_connectivity", H, f"added_edges={added}")

    before = H.number_of_edges()
    _adjust_edge_count(H, m, seed=seed)
    rec("adjust_edge_count", H, f"removed_edges={before - H.number_of_edges()}")
    return rows


def main():
    data_dir = os.path.join(ROOT, "dataset", "testdata")
    datasets = ["GtsCe", "Colt", "Chinanet", "UsCarrier", "Cogentco"]
    all_rows = []
    for ds in datasets:
        p = os.path.join(data_dir, f"{ds}.gml")
        if not os.path.exists(p):
            continue
        rows = trace(ds, p)
        all_rows.extend(rows)
        print(f"\n=== {ds} ===")
        for r in rows:
            print(f"  {r['stage']:<22} m={r['m']:<4} (Δ{r['delta_m']:+d})  "
                  f"k_min={r['k_min']}  k_max={r['k_max']}  conn={r['connected']}  "
                  f"deg1={r['num_degree1']}  {r['note']}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(all_rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n明细: {OUT}")


if __name__ == "__main__":
    main()
