# -*- coding: utf-8 -*-
"""
GCC (Giant Connected Component) 及连通分量相关结构指标
"""
import networkx as nx
from typing import List, Tuple


def gcc_size(G: nx.Graph) -> int:
    """
    计算最大连通分量 (GCC) 的节点数。

    Args:
        G: networkx.Graph

    Returns:
        int: 最大连通分量的大小。空图返回 0。
    """
    if G.number_of_nodes() == 0:
        return 0
    components = list(nx.connected_components(G))
    if not components:
        return 0
    return len(max(components, key=len))


def gcc_ratio(G: nx.Graph, initial_nodes: int = None) -> float:
    """
    计算最大连通分量占初始节点总数的比例。

    Args:
        G: networkx.Graph
        initial_nodes: 初始节点总数。若为 None，则使用当前图的节点数。

    Returns:
        float: GCC / initial_nodes，范围 [0, 1]。
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    denom = initial_nodes if initial_nodes is not None else n
    if denom == 0:
        return 0.0
    return gcc_size(G) / denom


def second_lcc_size(G: nx.Graph) -> int:
    """
    计算次大连通分量 (Second LCC) 的节点数。

    Args:
        G: networkx.Graph

    Returns:
        int: 次大连通分量的大小。若连通分量不足 2 个，返回 0。
    """
    if G.number_of_nodes() == 0:
        return 0
    sizes = sorted([len(c) for c in nx.connected_components(G)], reverse=True)
    return sizes[1] if len(sizes) >= 2 else 0


def all_component_sizes(G: nx.Graph) -> List[int]:
    """
    返回所有连通分量的大小列表（按降序排列）。

    Args:
        G: networkx.Graph

    Returns:
        List[int]: 连通分量大小列表。
    """
    if G.number_of_nodes() == 0:
        return []
    return sorted([len(c) for c in nx.connected_components(G)], reverse=True)
