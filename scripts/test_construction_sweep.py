# -*- coding: utf-8 -*-
"""构造器健壮性回归：遍历真实拓扑 × hub 数，检查 I1–I4 与失败率。

用法： python scripts/test_construction_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topology.bimodal_construction import build_bimodal, construct_bimodal_sequence  # noqa: E402
from src.topology.generators import load_graph  # noqa: E402

NAMES = ["GtsCe", "Cogentco", "UsCarrier", "Colt", "Chinanet", "BA-200", "BA-500"]
NH = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]


def check(g, spec, n, m, k_min=2):
    deg = dict(g.degree())
    hubs = set(g.graph.get("hub_nodes", []))
    rest = g.subgraph([v for v in g if v not in hubs])
    return {
        "I1": g.number_of_nodes() == n,
        "I2": g.number_of_edges() == m,
        "I3": nx.is_connected(g),
        "I4a": nx.is_connected(rest),
        "I4b": min(deg.values()) >= k_min,
        "I5_bimodal": min(deg[h] for h in hubs) > max(deg[v] for v in g if v not in hubs),
    }


def main():
    total = fail = 0
    for name in NAMES:
        p = ROOT / "dataset" / "testdata" / f"{name}.gml"
        g0, _ = load_graph(str(p), verbose=False)
        if g0 is None:
            print(f"[skip] {name}")
            continue
        g0 = nx.convert_node_labels_to_integers(g0)
        n, m = g0.number_of_nodes(), g0.number_of_edges()
        line = []
        for nh in NH:
            if nh >= n - 2:
                continue
            spec = construct_bimodal_sequence(n, m, n_hubs=nh, k_min=2)
            if not spec.feasible:
                line.append(f"{nh}:infeas")
                continue
            total += 1
            try:
                g, s = build_bimodal(n, m, n_hubs=nh, seed=2026, k_min=2)
            except Exception as exc:  # noqa: BLE001
                fail += 1
                line.append(f"{nh}:ERR({type(exc).__name__})")
                continue
            c = check(g, s, n, m)
            ok = all(c.values())
            if not ok:
                fail += 1
            mark = "ok" if ok else "BAD" + "".join(k for k, v in c.items() if not v)
            line.append(f"{nh}:{mark}/{s.k_max}/{g.graph.get('construction','?')[:6]}")
        print(f"{name:10s} N={n:4d} M={m:4d} | " + "  ".join(line))
    print(f"\n构造 {total} 次，失败 {fail} 次，成功率 {100*(total-fail)/max(total,1):.1f}%")


if __name__ == "__main__":
    main()
