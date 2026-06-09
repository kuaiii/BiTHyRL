# -*- coding: utf-8 -*-
"""
网络重构算法统一接口
======================================
所有的拓扑构造/重构算法都接受network.Graph，并返回至少包含键“G_constructed”的字典。
用法
-----
>>> from network_construction.unified_interface import construct, list_algorithms
>>> G2 = construct(G, algorithm='Onion', max_iterations=2000)
>>> print(G2.number_of_nodes(), G2.number_of_edges())
"""
import logging
from typing import Dict, Callable, Optional, Any
import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
CONSTRUCTION_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {}


def register_algorithm(name: str):
    """注册网络重构算法装饰器"""
    def decorator(func: Callable):
        CONSTRUCTION_REGISTRY[name] = func
        return func
    return decorator


def list_algorithms():
    """列出所有已经注册的算法"""
    return list(CONSTRUCTION_REGISTRY.keys())


def construct(G: nx.Graph, algorithm: str = "baseline", **kwargs) -> nx.Graph:
    """
    统一单层拓扑重构算法接口。
    参数：
    ----------
    G : networkx.Graph
        Input network.
    algorithm : str
        重构算法名字.  Default is ``baseline`` (returns a copy).
    **kwargs : 额外算法所需要的一些参数.

    返回值：
    -------
    networkx.Graph
        重构 / 优化后的网络拓扑.
    """
    if algorithm not in CONSTRUCTION_REGISTRY:
        raise ValueError(
            f"Unknown construction algorithm '{algorithm}'. "
            f"Available: {list_algorithms()}"
        )
    # 调用注册的算法函数，传入图和额外参数
    result = CONSTRUCTION_REGISTRY[algorithm](G, **kwargs)
    # 如果注册的函数返回一个字典，必须包含键 'G_constructed'，否则抛出错误
    if isinstance(result, dict):
        if "G_constructed" not in result:
            raise RuntimeError(
                f"Algorithm '{algorithm}' returned a dict without 'G_constructed'."
            )
        # 返回构造好的图
        return result["G_constructed"]
    elif isinstance(result, nx.Graph):
        return result
    else:
        raise RuntimeError(
            f"Algorithm '{algorithm}' returned unsupported type {type(result)}."
        )


# ---------------------------------------------------------------------------
# GA (Genetic Algorithm)
# ---------------------------------------------------------------------------
try:
    from src.topology.genetic_optimization import solution

    @register_algorithm("GA")
    def _ga(G: nx.Graph, **kwargs) -> Dict[str, Any]:
        nodes_num = G.number_of_nodes()
        edges_num = G.number_of_edges()
        best_solution, _ = solution(nodes_num, edges_num)
        G_opt = nx.from_numpy_array(best_solution)
        added = list(set(G_opt.edges()) - set(G.edges()))
        return {"G_constructed": G_opt, "added_edges": added, "algorithm": "GA"}
except Exception as _e:
    logger.warning(f"GA not available for unified interface: {_e}")


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
@register_algorithm("baseline")
def _baseline(G: nx.Graph, **kwargs) -> Dict[str, Any]:
    """返回一个G的副本，不进行任何的修改。"""
    return {"G_constructed": G.copy(), "added_edges": [], "algorithm": "baseline"}


# ---------------------------------------------------------------------------
# DAiMo
# ---------------------------------------------------------------------------
try:
    from .Daimo.Daimo import DAiMoOptimizer
    @register_algorithm("DAiMo")
    def _daimo(G: nx.Graph, local_generations: int = 50, num_local_workers: int = 6, **kwargs) -> Dict[str, Any]:
        optimizer = DAiMoOptimizer(
            local_generations=local_generations,
            num_local_workers=num_local_workers
        )
        G_opt = optimizer.optimize(G)
        added = list(set(G_opt.edges()) - set(G.edges()))
        return {"G_constructed": G_opt, "added_edges": added, "algorithm": "DAiMo"}
except Exception as _e:
    logger.warning(f"DAiMo not available for unified interface: {_e}")


# ---------------------------------------------------------------------------
# ROMEN
# ---------------------------------------------------------------------------
try:
    from .ROMEN.ROMEN import optimize_network_robustness

    @register_algorithm("ROMEN")
    def _romen(G: nx.Graph, population_size: int = 3, generations: int = 50, **kwargs) -> Dict[str, Any]:
        import random
        import numpy as np
        # Generate simple random positions if missing
        pos = {}
        for n in G.nodes():
            if 'pos' in G.nodes[n]:
                p = G.nodes[n]['pos']
                if isinstance(p, (tuple, list)) and len(p) >= 2:
                    pos[n] = (float(p[0]), float(p[1]))
        if len(pos) != G.number_of_nodes():
            rng = np.random.default_rng(kwargs.get('seed', 42))
            pos = {n: (rng.random() * 1000, rng.random() * 1000) for n in G.nodes()}

        result = optimize_network_robustness(
            initial_graph=G,
            positions=pos,
            population_size=population_size,
            generations=generations,
            verbose=False
        )
        G_opt = result.get('optimized_graph', G)
        added = list(set(G_opt.edges()) - set(G.edges()))
        return {"G_constructed": G_opt, "added_edges": added, "algorithm": "ROMEN"}
except Exception as _e:
    logger.warning(f"ROMEN not available for unified interface: {_e}")


# ---------------------------------------------------------------------------
# UNITY
# ---------------------------------------------------------------------------
try:
    from .UNITY.UNITY import UNITY

    @register_algorithm("UNITY")
    def _unity(G: nx.Graph, seed: int = 3, **kwargs) -> Dict[str, Any]:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        unity = UNITY(area_diameter=500, comm_range=200)
        G_opt = unity.generate_matched_topology(G)
        # Fallback: if edge count mismatch, try wider comm range
        if G_opt.number_of_edges() < G.number_of_edges():
            unity_wide = UNITY(area_diameter=500, comm_range=1000)
            random.seed(seed)
            np.random.seed(seed)
            G_opt = unity_wide.generate_matched_topology(G)
        added = list(set(G_opt.edges()) - set(G.edges()))
        return {"G_constructed": G_opt, "added_edges": added, "algorithm": "UNITY"}
except Exception as _e:
    logger.warning(f"UNITY not available for unified interface: {_e}")


# ---------------------------------------------------------------------------
# ONION
# ---------------------------------------------------------------------------
try:
    from .ONION.onion import OnionNetworkOptimizer

    @register_algorithm("Onion")
    def _onion(G: nx.Graph, max_iterations: int = 2000, **kwargs) -> Dict[str, Any]:
        optimizer = OnionNetworkOptimizer(G)
        G_opt = optimizer.optimize_network(max_iterations=max_iterations, verbose=False)
        added = list(set(G_opt.edges()) - set(G.edges()))
        return {"G_constructed": G_opt, "added_edges": added, "algorithm": "Onion"}
except Exception as _e:
    logger.warning(f"Onion not available for unified interface: {_e}")


# ---------------------------------------------------------------------------
# FRED-ABL
# ---------------------------------------------------------------------------
try:
    from .FRED_ABL.fred_abl_optimization import IIoTEnv, FRED_ABL

    @register_algorithm("FRED_ABL")
    def _fred_abl(G: nx.Graph, iterations: int = 15, comm_range: float = 0.4, **kwargs) -> Dict[str, Any]:
        import random
        import numpy as np
        # Normalise positions to [0, 1]
        pos = {}
        for n in G.nodes():
            if 'pos' in G.nodes[n]:
                p = G.nodes[n]['pos']
                if isinstance(p, (tuple, list)) and len(p) >= 2:
                    pos[n] = (float(p[0]), float(p[1]))
        if len(pos) != G.number_of_nodes():
            rng = np.random.default_rng(kwargs.get('seed', 42))
            pos = {n: (rng.random(), rng.random()) for n in G.nodes()}

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        norm_pos = {}
        G_norm = nx.Graph()
        G_norm.add_nodes_from(range(G.number_of_nodes()))
        for i, n in enumerate(G.nodes()):
            norm_pos[i] = ((pos[n][0] - min(xs)) / scale, (pos[n][1] - min(ys)) / scale)
            G_norm.nodes[i]['pos'] = norm_pos[i]
        node_map = {n: i for i, n in enumerate(G.nodes())}
        for u, v in G.edges():
            G_norm.add_edge(node_map[u], node_map[v])

        env = IIoTEnv(G=G_norm, comm_range=comm_range)
        fred = FRED_ABL(env)
        best_adj = fred.run(iterations=iterations, initial_samples=3, verbose=False)
        G_opt = nx.from_numpy_array(best_adj)
        if 'pos' in G_norm.nodes[0]:
            for i in G_opt.nodes():
                if i in norm_pos:
                    G_opt.nodes[i]['pos'] = norm_pos[i]
        added = list(set(G_opt.edges()) - set(G_norm.edges()))
        return {"G_constructed": G_opt, "added_edges": added, "algorithm": "FRED_ABL"}
except Exception as _e:
    logger.warning(f"FRED_ABL not available for unified interface: {_e}")


# ---------------------------------------------------------------------------
# QDLM
# ---------------------------------------------------------------------------
try:
    from .QDLM.qdlm_optimization import QuantumEnvironment, QuantumLearningModel

    @register_algorithm("QDLM")
    def _qdlm(G: nx.Graph, pop_size: int = 20, max_iterations: int = 30, comm_range: float = 0.4, **kwargs) -> Dict[str, Any]:
        import random
        import numpy as np
        pos = {}
        for n in G.nodes():
            if 'pos' in G.nodes[n]:
                p = G.nodes[n]['pos']
                if isinstance(p, (tuple, list)) and len(p) >= 2:
                    pos[n] = (float(p[0]), float(p[1]))
        if len(pos) != G.number_of_nodes():
            rng = np.random.default_rng(kwargs.get('seed', 42))
            pos = {n: (rng.random(), rng.random()) for n in G.nodes()}

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        norm_pos = {}
        G_norm = nx.Graph()
        G_norm.add_nodes_from(range(G.number_of_nodes()))
        for i, n in enumerate(G.nodes()):
            norm_pos[i] = ((pos[n][0] - min(xs)) / scale, (pos[n][1] - min(ys)) / scale)
            G_norm.nodes[i]['pos'] = norm_pos[i]
        node_map = {n: i for i, n in enumerate(G.nodes())}
        for u, v in G.edges():
            G_norm.add_edge(node_map[u], node_map[v])

        env = QuantumEnvironment(G=G_norm, comm_range=comm_range)
        if env.m == 0:
            return {"G_constructed": G_norm, "added_edges": [], "algorithm": "QDLM"}
        qlm = QuantumLearningModel(env, pop_size=pop_size, max_iterations=max_iterations)
        best_adj = qlm.run(verbose=False)
        G_opt = nx.from_numpy_array(best_adj)
        added = list(set(G_opt.edges()) - set(G_norm.edges()))
        return {"G_constructed": G_opt, "added_edges": added, "algorithm": "QDLM"}
except Exception as _e:
    logger.warning(f"QDLM not available for unified interface: {_e}")


# ---------------------------------------------------------------------------
# Q-Robust
# ---------------------------------------------------------------------------
try:
    from .other.Q_Robust import Q_Robust_method

    @register_algorithm("Q_Robust")
    def _qrobust(G: nx.Graph, max_gen: int = 10, **kwargs) -> Dict[str, Any]:
        G_opt = Q_Robust_method(G, max_gen=max_gen, use_optimized=True)
        added = list(set(G_opt.edges()) - set(G.edges()))
        return {"G_constructed": G_opt, "added_edges": added, "algorithm": "Q_Robust"}
except Exception as _e:
    logger.warning(f"Q_Robust not available for unified interface: {_e}")


# ---------------------------------------------------------------------------
# SmartTRO
# ---------------------------------------------------------------------------
try:
    from .other.smartTRO import SmartTRO

    @register_algorithm("SmartTRO")
    def _smarttro(G: nx.Graph, model_path: str = "models/smarttro_final.pt", max_actions: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        import os
        edge_density = max(1, int(2 * G.number_of_edges() / G.number_of_nodes()))
        agent = SmartTRO(num_nodes=G.number_of_nodes(), edge_density=edge_density, num_agents=1)
        if max_actions is not None:
            agent.env.max_actions = max_actions
        if os.path.exists(model_path):
            agent.load_model(model_path)
        G_opt = agent.optimize_topology(G)
        added = list(set(G_opt.edges()) - set(G.edges()))
        return {"G_constructed": G_opt, "added_edges": added, "algorithm": "SmartTRO"}
except Exception as _e:
    logger.warning(f"SmartTRO not available for unified interface: {_e}")


# ---------------------------------------------------------------------------
# TEAM
# ---------------------------------------------------------------------------
try:
    from .other.team_optimization import IoTNetwork, TEAM_Engine

    @register_algorithm("TEAM")
    def _team(G: nx.Graph, **kwargs) -> Dict[str, Any]:
        iot_net = IoTNetwork(G=G)
        engine = TEAM_Engine(iot_net)
        engine.run(verbose=False)
        G_opt = iot_net.graph.copy()
        added = list(set(G_opt.edges()) - set(G.edges()))
        return {"G_constructed": G_opt, "added_edges": added, "algorithm": "TEAM"}
except Exception as _e:
    logger.warning(f"TEAM not available for unified interface: {_e}")


# ---------------------------------------------------------------------------
# BiT-HyRL bimodal topology (via src.topology.reconstruction)
# ---------------------------------------------------------------------------
try:
    import sys
    import os
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from src.topology.reconstruction import create_bimodal_network_exact

    @register_algorithm("BiT-HyRL")
    def _bit_hyrl(G: nx.Graph, num_hubs: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        hub_num = num_hubs if num_hubs is not None else max(1, G.number_of_nodes() // 10)
        G_opt = create_bimodal_network_exact(G, hub_num=hub_num)
        added = list(set(G_opt.edges()) - set(G.edges()))
        return {"G_constructed": G_opt, "added_edges": added, "algorithm": "BiT-HyRL"}
except Exception as _e:
    logger.warning(f"BiT-HyRL bimodal topology not available for unified interface: {_e}")
