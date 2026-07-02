#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边数中性归一化工具。

保证所有重构算法输出拓扑的节点数 N 与边数 M 与原图完全一致。
若某方法输出边数不等于目标值，则用最小侵入的中性启发式进行后处理：
- 边数过多：优先删除 betweenness 最低的非桥边，直到 M == target。
- 边数过少：优先在度数最低的节点对之间添加不存在的边，直到 M == target。
"""
import random
import numpy as np
import networkx as nx


def sample_edge_betweenness(G, sample_ratio=0.1, min_k=10, max_k=500, seed=None):
    """对大图采样近似 edge betweenness；小图精确计算。"""
    if seed is not None:
        random.seed(seed)
    n = G.number_of_nodes()
    if n <= 2:
        return {}
    m = G.number_of_edges()
    # 小图直接精确计算
    if n <= 200 or m <= 2000:
        try:
            return nx.edge_betweenness_centrality(G)
        except Exception:
            return {}
    k = min(max_k, max(min_k, int(n * sample_ratio)))
    try:
        return nx.edge_betweenness_centrality(G, k=k)
    except Exception:
        return {}


def normalize_edge_count(G, target_edges, seed=42, preserve_connectivity=True):
    """
    将图 G 的边数调整到 target_edges，节点集保持不变。

    Args:
        G: nx.Graph，会被修改（调用前建议 copy）
        target_edges: int，目标边数
        seed: int
        preserve_connectivity: bool，删除时尽量保持连通

    Returns:
        G_norm: 调整后的图（与输入同一对象）
        info: dict，{
            'edges_before': int,
            'edges_after': int,
            'edges_removed': int,
            'edges_added': int,
            'note': str
        }
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G = nx.Graph(G)
    edges_before = G.number_of_edges()
    info = {
        'edges_before': edges_before,
        'edges_after': edges_before,
        'edges_removed': 0,
        'edges_added': 0,
        'note': '',
    }

    if edges_before == target_edges:
        return G, info

    if edges_before > target_edges:
        need_remove = edges_before - target_edges
        removed = 0
        note_fragments = []

        eb = sample_edge_betweenness(G, seed=seed)
        if not eb:
            note_fragments.append('betweenness computation failed')

        # 候选删除边：按 betweenness 升序（删负载小的边影响小）
        candidates = sorted(
            G.edges(),
            key=lambda e: eb.get(tuple(sorted(e)), 0.0)
        )

        for u, v in candidates:
            if removed >= need_remove:
                break
            if not G.has_edge(u, v):
                continue
            if G.degree(u) <= 1 or G.degree(v) <= 1:
                continue
            if preserve_connectivity:
                G.remove_edge(u, v)
                if not nx.is_connected(G):
                    G.add_edge(u, v)
                    continue
            else:
                G.remove_edge(u, v)
            removed += 1

        info['edges_removed'] = removed
        info['edges_after'] = G.number_of_edges()
        if removed < need_remove:
            note_fragments.append(
                f'could only remove {removed}/{need_remove} edges while preserving connectivity'
            )
        if note_fragments:
            info['note'] = '; '.join(note_fragments)
        return G, info

    else:  # edges_before < target_edges
        need_add = target_edges - edges_before
        added = 0
        nodes = list(G.nodes())
        degrees = dict(G.degree())

        # 候选节点对：优先连接低度节点，保持图结构尽量均匀
        pairs = []
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                if not G.has_edge(u, v):
                    pairs.append((u, v))
        # 按度数和升序排序（优先低-低或低-高度节点连接）
        pairs.sort(key=lambda e: degrees.get(e[0], 0) + degrees.get(e[1], 0))
        random.shuffle(pairs[:min(len(pairs), 1000)])  # 扰动避免完全确定性

        for u, v in pairs:
            if added >= need_add:
                break
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                added += 1

        info['edges_added'] = added
        info['edges_after'] = G.number_of_edges()
        if added < need_add:
            info['note'] = f'could only add {added}/{need_add} edges (no more candidate pairs)'
        return G, info
