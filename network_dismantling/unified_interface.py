"""
Unified network dismantling interface.
All algorithms accept networkx.Graph and return a dismantling sequence.
"""
import logging
from typing import List, Optional, Callable, Dict
from functools import wraps

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

METHOD_REGISTRY: Dict[str, Callable] = {}


def list_methods() -> List[str]:
    """返回所有已注册的拆解方法名称。"""
    return list(METHOD_REGISTRY.keys())


def register_method(name: str):
    """Decorator to register a dismantling method."""
    def decorator(func: Callable):
        METHOD_REGISTRY[name] = func
        return func
    return decorator


def _standardize_graph(G: nx.Graph) -> nx.Graph:
    """
    Standardize a networkx graph for dismantling:
    - Convert to undirected simple graph (no self-loops, no parallel edges)
    - Relabel nodes to consecutive integers starting from 0
    - Returns the standardized graph and the mapping (new -> old)
    """
    if G.is_directed():
        G = G.to_undirected()
    G = nx.Graph(G)  # remove parallel edges
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # Relabel to consecutive integers 0..n-1
    mapping = {node: i for i, node in enumerate(G.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}
    G = nx.relabel_nodes(G, mapping)
    
    G.graph["_reverse_mapping"] = reverse_mapping
    return G


def _fill_remaining(G: nx.Graph, sequence: List[int]) -> List[int]:
    """Fill remaining nodes (after stop_condition) by degree descending."""
    removed = set(sequence)
    remaining = [v for v in G.nodes() if v not in removed]
    remaining.sort(key=lambda v: G.degree(v), reverse=True)
    return list(sequence) + remaining


def dismantle(G: nx.Graph, method: str, stop_condition: Optional[int] = None, **kwargs) -> List[int]:
    """
    Unified dismantling interface.
    
    Parameters
    ----------
    G : networkx.Graph
        Input network (will be converted to undirected simple graph)
    method : str
        Dismantling method name (e.g., 'degree', 'pagerank', 'betweenness', 
        'eigenvector', 'random', 'brute_force', 'entanglement_small', 
        'entanglement_mid', 'entanglement_large', 'vertex_entanglement')
    stop_condition : int, optional
        Stop dismantling when LCC <= stop_condition. 
        If None, dismantle until all nodes are removed.
    **kwargs : additional method-specific parameters
    
    Returns
    -------
    List[int]
        Dismantling sequence of all nodes (original node IDs)
    """
    if method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method '{method}'. Available: {list(METHOD_REGISTRY.keys())}")
    
    # Standardize graph
    G_std = _standardize_graph(G)
    reverse_mapping = G_std.graph["_reverse_mapping"]
    n = G_std.number_of_nodes()
    
    if stop_condition is None:
        stop_condition = 1  # dismantle all
    
    # Call registered method (stop_condition is a caller-side concern now)
    func = METHOD_REGISTRY[method]
    seq_std = func(G_std, **kwargs)
    
    # Ensure complete sequence
    if len(seq_std) < n:
        seq_std = _fill_remaining(G_std, seq_std)
    
    # Map back to original node IDs
    seq_orig = [reverse_mapping[v] for v in seq_std]
    return seq_orig


# ---------------------------------------------------------------------------
# k-hop rolling-window dismantling (fair comparison: all methods see local)
# ---------------------------------------------------------------------------
def _bfs_khop_nodes(G: nx.Graph, source: int, k_hop: int) -> set:
    """
    通过BFS迭代获取source节点的k-hop范围内的所有邻居节点。
    
    参数:
      G: NetworkX图。
      source: 起始节点。
      k_hop: BFS深度(跳数)。
    
    返回: k-hop范围内所有节点的集合(包含source本身)。
    """
    visited = {source}
    frontier = {source}
    for _ in range(k_hop):
        next_frontier = set()
        for node in frontier:
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        if not next_frontier:
            break
        frontier = next_frontier
    return visited


def dismantle_khop_rolling(
    G: nx.Graph,
    method: str,
    k_hop: int,
    seed: int = 42,
    initial_center: Optional[int] = None,
    **method_kwargs,
) -> List[int]:
    """
    k-hop 滚动窗口拆解方法 (视野逐步扩展版)。
    
    核心逻辑:
      1. G_working 始终保持完整，不物理删除节点。
      2. 维护 visible_nodes 集合表示当前算法"视野范围"。
      3. 每次移除节点 v 后，将 v 的 k-hop 邻居加入 visible_nodes 扩展视野。
      4. 在 visible_nodes 构成的子图上运行 dismantle，取 seq 中第一个不在 removed 中的节点。
      5. 当 visible_nodes 覆盖全图时，直接在 G_working 上运行一次全局 dismantle 补齐剩余序列。
    
    参数:
      G: 原始输入图(不修改)。
      method: 拆解方法名称。
      k_hop: 观察半径(跳数)。
      seed: 随机种子。
      initial_center: 初始中心节点(从外部传入确保各方法可比)。
      **method_kwargs: 额外参数透传给 dismantle(如 ADAPT 的 model_path/device)。
    
    返回: 完整的拆解序列(原始节点ID)。
    """
    # 兼容 ADAPT 的 k_hop 参数命名冲突：dismantle_khop_rolling 的 k_hop 是视野半径，
    # ADAPT 模型的 k_hop 通过 adapt_k 传入，在此处映射回 k_hop 传给 dismantle。
    if "adapt_k" in method_kwargs:
        method_kwargs = dict(method_kwargs)
        method_kwargs["k_hop"] = method_kwargs.pop("adapt_k")

    rng = np.random.default_rng(seed)
    G_working = G.copy()   # 保持完整，不物理删除节点
    removed = []           # 记录已移除节点顺序
    visible_nodes = set()  # 当前视野范围

    # 初始化中心节点
    if initial_center is None or initial_center not in G_working:
        initial_center = int(rng.choice(list(G_working.nodes())))

    # 初始化视野：将 initial_center 的 k-hop 邻居加入 visible_nodes
    visible_nodes.update(_bfs_khop_nodes(G_working, initial_center, k_hop))

    # 迭代拆解过程
    while len(removed) < G.number_of_nodes():
        # 检查视野是否已覆盖全图
        if len(visible_nodes) >= G.number_of_nodes():
            # 视野全局化：直接在完整图上运行 dismantle 补齐剩余序列
            try:
                seq = dismantle(G_working, method=method, **method_kwargs)
            except Exception as e:
                logger.warning(
                    f"dismantle_khop_rolling: global dismantle failed for method '{method}': {e}"
                )
                seq = sorted(G_working.nodes(), key=lambda v: G_working.degree(v), reverse=True)
            # 按顺序追加所有不在 removed 中的节点
            for node in seq:
                if node not in removed:
                    removed.append(node)
            break

        # 在视野范围内构建子图
        subG = G_working.subgraph(visible_nodes).copy()
        if subG.number_of_nodes() == 0:
            break

        # 在局部子图上运行 dismantle
        try:
            seq = dismantle(subG, method=method, **method_kwargs)
        except Exception as e:
            logger.warning(
                f"dismantle_khop_rolling: method '{method}' failed on visible subgraph: {e}"
            )
            # Fallback: 按度降序
            seq = sorted(subG.nodes(), key=lambda v: subG.degree(v), reverse=True)

        # 找到 seq 中第一个不在 removed 中的节点(防御性：seq可能含已移除节点)
        v = None
        for node in seq:
            if node not in removed:
                v = node
                break

        if v is None:
            # seq 中所有节点都已被移除，跳出
            break

        removed.append(v)

        # 扩展视野：将 v 的 k-hop 邻居加入 visible_nodes
        if v in G_working:
            visible_nodes.update(_bfs_khop_nodes(G_working, v, k_hop))

    # 防御性：确保返回完整序列(补齐任何遗漏的节点)
    remaining = [v for v in G.nodes() if v not in removed]
    if remaining:
        remaining.sort(key=lambda v: G.degree(v), reverse=True)
        removed.extend(remaining)

    return removed


# ---------------------------------------------------------------------------
# Heuristics (static scores)
# ---------------------------------------------------------------------------
from network_dismantling.heuristics.sorters_nx import (
    degree_scores,
    pagerank_scores,
    betweenness_scores,
    eigenvector_scores,
    random_scores,
)


# Import ADAPT to register it in the unified interface
try:
    import network_dismantling.ADAPT  # noqa: F401
except Exception as _e:
    logger.warning(f"ADAPT not available for unified interface: {_e}")

@register_method("degree")
def _degree_dismantler(G: nx.Graph, **kwargs) -> List[int]:
    return degree_scores(G)


@register_method("pagerank")
def _pagerank_dismantler(G: nx.Graph, **kwargs) -> List[int]:
    return pagerank_scores(G)


@register_method("betweenness")
def _betweenness_dismantler(G: nx.Graph, **kwargs) -> List[int]:
    return betweenness_scores(G)


@register_method("eigenvector")
def _eigenvector_dismantler(G: nx.Graph, **kwargs) -> List[int]:
    return eigenvector_scores(G)


@register_method("random")
def _random_dismantler(G: nx.Graph, seed: int = None, **kwargs) -> List[int]:
    return random_scores(G, seed=seed)


# ---------------------------------------------------------------------------
# Brute Force (for small networks)
# ---------------------------------------------------------------------------
@register_method("brute_force")
def _brute_force_dismantler(G: nx.Graph, max_k: int = None, **kwargs) -> List[int]:
    """
    Brute-force dismantler for small networks.
    Finds the smallest set of nodes whose removal reduces LCC to <= 1.
    Remaining nodes are sorted by degree descending.
    """
    from itertools import combinations
    
    n = G.number_of_nodes()
    if max_k is None:
        max_k = min(n, 10)
    
    nodes = list(G.nodes())
    best_sequence = None
    
    for k in range(1, max_k + 1):
        best_score = n
        best_combo = None
        
        for combo in combinations(nodes, k):
            temp = G.copy()
            temp.remove_nodes_from(combo)
            if temp.number_of_nodes() > 0:
                components = list(nx.connected_components(temp))
                lcc = max(len(c) for c in components) if components else 0
            else:
                lcc = 0
            
            if lcc < best_score:
                best_score = lcc
                best_combo = combo
            
            if best_score <= 1:
                break
        
        if best_score <= 1 and best_combo is not None:
            best_sequence = list(best_combo)
            break
    
    if best_sequence is None:
        # Fallback to degree
        best_sequence = sorted(nodes, key=lambda v: G.degree(v), reverse=True)[:max_k]
    
    remaining = [v for v in nodes if v not in best_sequence]
    remaining.sort(key=lambda v: G.degree(v), reverse=True)
    return best_sequence + remaining


# ---------------------------------------------------------------------------
# Multiscale Entanglement (networkx version)
# ---------------------------------------------------------------------------
from network_dismantling.multiscale_entanglement.original_entanglement_functions import (
    entanglement_small as _entanglement_small,
    entanglement_mid as _entanglement_mid,
    entanglement_large as _entanglement_large,
)


@register_method("entanglement_small")
def _entanglement_small_dismantler(G: nx.Graph, **kwargs) -> List[int]:
    ent = _entanglement_small(G)
    scores = np.zeros(G.number_of_nodes())
    for v, val in ent.items():
        scores[v] = val
    nodes = list(G.nodes())
    nodes.sort(key=lambda v: scores[v], reverse=True)
    return nodes


@register_method("entanglement_mid")
def _entanglement_mid_dismantler(G: nx.Graph, **kwargs) -> List[int]:
    ent = _entanglement_mid(G)
    scores = np.zeros(G.number_of_nodes())
    for v, val in ent.items():
        scores[v] = val
    nodes = list(G.nodes())
    nodes.sort(key=lambda v: scores[v], reverse=True)
    return nodes


@register_method("entanglement_large")
def _entanglement_large_dismantler(G: nx.Graph, **kwargs) -> List[int]:
    ent = _entanglement_large(G)
    scores = np.zeros(G.number_of_nodes())
    for v, val in ent.items():
        scores[v] = val
    nodes = list(G.nodes())
    nodes.sort(key=lambda v: scores[v], reverse=True)
    return nodes


# ---------------------------------------------------------------------------
# CI (Collective Influence) - C executable wrapper
# ---------------------------------------------------------------------------
from network_dismantling.CI.ci_wrapper_nx import ci_dismantle_nx


@register_method("CI_L1")
def _ci_l1_dismantler(G: nx.Graph, **kwargs) -> List[int]:
    return ci_dismantle_nx(G, l=1, stop_condition=1)


@register_method("CI_L2")
def _ci_l2_dismantler(G: nx.Graph, **kwargs) -> List[int]:
    return ci_dismantle_nx(G, l=2, stop_condition=1)


@register_method("CI_L3")
def _ci_l3_dismantler(G: nx.Graph, **kwargs) -> List[int]:
    return ci_dismantle_nx(G, l=3, stop_condition=1)


# ---------------------------------------------------------------------------
# CoreHD (Python reimplementation)
# ---------------------------------------------------------------------------
from network_dismantling.CoreHD.corehd_nx import corehd_dismantle


@register_method("CoreHD")
def _corehd_dismantler(G: nx.Graph, seed: int = None, **kwargs) -> List[int]:
    seq = corehd_dismantle(G, stop_condition=1, seed=seed)
    # Fill remaining nodes by degree descending
    return _fill_remaining(G, seq)


# ---------------------------------------------------------------------------
# GND / EGND (Python reimplementation)
# ---------------------------------------------------------------------------
from network_dismantling.GND.gnd_nx import gnd_dismantle


@register_method("GND")
def _gnd_dismantler(G: nx.Graph, remove_strategy: int = 3, seed: int = None, **kwargs) -> List[int]:
    seq = gnd_dismantle(G, stop_condition=1, remove_strategy=remove_strategy, seed=seed)
    return _fill_remaining(G, seq)


# ---------------------------------------------------------------------------
# EI (Python reimplementation)
# ---------------------------------------------------------------------------
from network_dismantling.EI.ei_nx import ei_dismantle


@register_method("EI_s1")
def _ei_s1_dismantler(G: nx.Graph, kk: int = 1000, seed: int = None, **kwargs) -> List[int]:
    return ei_dismantle(G, stop_condition=1, sigma=1, kk=kk, seed=seed)


@register_method("EI_s2")
def _ei_s2_dismantler(G: nx.Graph, kk: int = 1000, seed: int = None, **kwargs) -> List[int]:
    return ei_dismantle(G, stop_condition=1, sigma=2, kk=kk, seed=seed)


@register_method("EGND")
def _egnd_dismantler(G: nx.Graph, runs: int = 10, remove_strategy: int = 3, seed: int = None, **kwargs) -> List[int]:
    """
    Ensemble GND: run GND multiple times with different seeds and pick the best result
    (minimum number of removed nodes).
    """
    rng = np.random.default_rng(seed)
    best_seq = None
    best_len = float('inf')
    
    for i in range(runs):
        run_seed = rng.integers(0, 2**31)
        seq = gnd_dismantle(G, stop_condition=1, remove_strategy=remove_strategy, seed=run_seed)
        if len(seq) < best_len:
            best_len = len(seq)
            best_seq = seq
    
    return _fill_remaining(G, best_seq)


# ---------------------------------------------------------------------------
# Vertex Entanglement (networkx version)
# ---------------------------------------------------------------------------
from network_dismantling.vertex_entanglement.vertex_entanglement_nx import VertexEnt_nx


@register_method("vertex_entanglement")
def _vertex_entanglement_dismantler(G: nx.Graph, **kwargs) -> List[int]:
    """
    Vertex Entanglement dismantler.
    Uses numpy spectral computation on networkx graph.
    """
    # VE returns array where lower value = more important for dismantling
    ve = VertexEnt_nx(G, perturb_strategy='default')
    # Invert: lower VE -> higher priority
    scores = -ve
    nodes = list(G.nodes())
    nodes.sort(key=lambda v: scores[v], reverse=True)
    return nodes


# ---------------------------------------------------------------------------
# GDM (Graph Dismantling Machine) - networkx reimplementation
# ---------------------------------------------------------------------------
from network_dismantling.GDM.predictors_nx import gdm_dismantle_from_path


@register_method("GDM")
def _gdm_dismantler(G: nx.Graph, **kwargs) -> List[int]:
    """
    GDM (Graph Dismantling Machine) dismantler.
    Uses a pre-trained GAT model to predict node importance scores.
    """
    model_path = kwargs.get("model_path", "network_dismantling/GDM/models_newpg/gdm_nx_best.pth")
    device = kwargs.get("device", "cuda" if __import__("torch").cuda.is_available() else "cpu")
    seq = gdm_dismantle_from_path(G, model_path, stop_condition=1, device=device)
    return _fill_remaining(G, seq)


@register_method("GDM+R")
def _gdm_reinsertion_dismantler(G: nx.Graph, **kwargs) -> List[int]:
    """
    GDM + Reinsertion dismantler.
    Uses GDM for initial prediction, then applies greedy reinsertion optimization.
    """
    model_path = kwargs.get("model_path", "network_dismantling/GDM/models_newpg/gdm_nx_best.pth")
    device = kwargs.get("device", "cuda" if __import__("torch").cuda.is_available() else "cpu")
    seq = gdm_dismantle_from_path(G, model_path, stop_condition=1, device=device, use_reinsertion=True)
    return _fill_remaining(G, seq)


# ---------------------------------------------------------------------------
# FINDER_ND (Deep RL dismantler)
# ---------------------------------------------------------------------------
try:
    from network_dismantling.FINDER_ND.python_interface_nx import finder_nd_dismantle
    _FINDER_AVAILABLE = True
except Exception as _e:
    _FINDER_AVAILABLE = False
    logger.warning(f"FINDER_ND not available: {_e}")


@register_method("FINDER")
def _finder_dismantler(G: nx.Graph, **kwargs) -> List[int]:
    """FINDER_ND dismantler (no reinsertion)."""
    if not _FINDER_AVAILABLE:
        raise RuntimeError("FINDER_ND is not available. Check FINDER_ND installation.")
    n = G.number_of_nodes()
    stop_frac = 1.0 / n if n > 0 else 0.01
    return finder_nd_dismantle(G, stop_condition=stop_frac, reinsertion=False, **kwargs)


@register_method("FINDER+R")
def _finder_reinsertion_dismantler(G: nx.Graph, **kwargs) -> List[int]:
    """FINDER_ND dismantler with reinsertion."""
    if not _FINDER_AVAILABLE:
        raise RuntimeError("FINDER_ND is not available. Check FINDER_ND installation.")
    n = G.number_of_nodes()
    stop_frac = 1.0 / n if n > 0 else 0.01
    return finder_nd_dismantle(G, stop_condition=stop_frac, reinsertion=True, **kwargs)
