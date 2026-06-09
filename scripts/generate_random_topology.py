# -*- coding: utf-8 -*-
"""
在 dataset/testdata 中生成指定节点数、边数的随机连通拓扑 GML 文件。
"""
import os
import sys
import random
import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_TESTDATA = os.path.join(ROOT, 'dataset', 'testdata')


def random_connected_graph(n: int, m: int, seed=None):
    """
    生成恰好 n 个节点、m 条边的随机连通无向图。
    先建一棵生成树（n-1 条边），再随机添加 m - (n-1) 条边。
    """
    if m < n - 1:
        raise ValueError(f"连通图至少需要 n-1={n - 1} 条边，给定 m={m}")
    if m > n * (n - 1) // 2:
        raise ValueError(f"n 个节点的简单图最多 {n * (n - 1) // 2} 条边，给定 m={m}")
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # 生成树：0-1-2-...-(n-1)
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    need = m - (n - 1)
    # 所有可能的新边（无自环、不重复）
    possible = [(i, j) for i in range(n) for j in range(i + 1, n) if not G.has_edge(i, j)]
    if need > len(possible):
        raise ValueError(f"无法在连通约束下得到恰好 m={m} 条边")
    chosen = rng.sample(possible, need)
    for u, v in chosen:
        G.add_edge(u, v)
    return G


def main():
    os.makedirs(DATA_TESTDATA, exist_ok=True)
    specs = [
        ('Gts1.gml', 149, 193),
        ('Col1.gml', 153, 177),
        ('Chi1.gml', 42, 66),
        ('Cog1.gml', 197, 243),
    ]
    base_seed = 20250201
    for idx, (fname, n, m) in enumerate(specs):
        path = os.path.join(DATA_TESTDATA, fname)
        G = random_connected_graph(n, m, seed=base_seed + idx)
        nx.write_gml(G, path)
        print(f"已生成: {path}  节点={G.number_of_nodes()}, 边={G.number_of_edges()}, 连通={nx.is_connected(G)}")
    print("\n完成。")


if __name__ == '__main__':
    main()
