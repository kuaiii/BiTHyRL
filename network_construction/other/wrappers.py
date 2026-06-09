"""
统一算法包装器
将现有独立实现的算法包装为符合 NetworkReconstructionAlgorithm 基类的接口
"""

import sys
import os
import copy
import random
import numpy as np
import networkx as nx

from .base import NetworkReconstructionAlgorithm
from .utils.metrics import compute_all_metrics


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------
def _generate_random_positions(G, area_size=1000, seed=None):
    """为图节点生成随机位置"""
    if seed is not None:
        random.seed(seed)
    return {n: (random.uniform(0, area_size), random.uniform(0, area_size)) for n in G.nodes()}


def _extract_positions(G):
    """从图节点属性中提取位置，如果没有则返回None"""
    pos = {}
    for n in G.nodes():
        if 'pos' in G.nodes[n]:
            p = G.nodes[n]['pos']
            if isinstance(p, (tuple, list)) and len(p) >= 2:
                pos[n] = (float(p[0]), float(p[1]))
        elif 'x' in G.nodes[n] and 'y' in G.nodes[n]:
            pos[n] = (float(G.nodes[n]['x']), float(G.nodes[n]['y']))
    return pos if len(pos) == G.number_of_nodes() else None


def _adj_to_graph(adj, pos=None):
    """邻接矩阵转无向图"""
    G = nx.from_numpy_array(np.array(adj))
    if pos is not None:
        for n in G.nodes():
            if n in pos:
                G.nodes[n]['pos'] = pos[n]
    return G


def _high_degree_attack_sequence(G):
    """生成动态高度数攻击序列"""
    seq = []
    H = G.copy()
    while H.number_of_nodes() > 0:
        d = dict(H.degree())
        if not d:
            break
        target = max(d, key=d.get)
        seq.append(target)
        H.remove_node(target)
    return seq


# ------------------------------------------------------------------
# 基线包装器
# ------------------------------------------------------------------

class RandomAttackWrapper(NetworkReconstructionAlgorithm):
    def __init__(self, seed=None, config=None):
        super().__init__(config)
        self.seed = seed

    def dismantle(self, G, stop_condition='threshold', stop_threshold=0.5):
        from .random_attack import RandomAttack
        return RandomAttack(seed=self.seed, config=self.config).dismantle(G, stop_condition, stop_threshold)

    def construct(self, G, target_edges=None, **kwargs):
        return {'G_constructed': G.copy(), 'added_edges': [], 'improvement_auc': 0.0, 'improvement_pc': 0.0}


class HighDegreeAttackWrapper(NetworkReconstructionAlgorithm):
    def __init__(self, config=None):
        super().__init__(config)

    def dismantle(self, G, stop_condition='threshold', stop_threshold=0.5):
        from .high_degree_attack import HighDegreeAttack
        return HighDegreeAttack(config=self.config).dismantle(G, stop_condition, stop_threshold)

    def construct(self, G, target_edges=None, **kwargs):
        return {'G_constructed': G.copy(), 'added_edges': [], 'improvement_auc': 0.0, 'improvement_pc': 0.0}


# ------------------------------------------------------------------
# DAiMo 包装器 (2025_TMC_DAiMo)
# ------------------------------------------------------------------

class DAiMoWrapper(NetworkReconstructionAlgorithm):
    """DAiMo: 基于模体分析的遗传算法优化"""

    def __init__(self, config=None):
        super().__init__(config)
        self.local_generations = config.get('daimo_local_generations', 50) if config else 50
        self.num_local_workers = config.get('daimo_local_workers', 6) if config else 6

    def construct(self, G, target_edges=None, **kwargs):
        from .Daimo import DAiMoOptimizer
        optimizer = DAiMoOptimizer(
            local_generations=self.local_generations,
            num_local_workers=self.num_local_workers
        )
        G_opt = optimizer.optimize(G)
        added = list(set(G_opt.edges()) - set(G.edges()))
        return {
            'G_constructed': G_opt,
            'added_edges': added,
            'improvement_auc': 0.0,
            'improvement_pc': 0.0
        }

    def dismantle(self, G, stop_condition='threshold', stop_threshold=0.5):
        result = self.construct(G)
        Gc = result['G_constructed']
        seq = _high_degree_attack_sequence(Gc)
        gcc = self.run_attack(Gc, seq)
        metrics = compute_all_metrics(gcc, Gc.number_of_nodes(), threshold=stop_threshold)
        return {
            'attack_sequence': seq,
            'gcc_sizes': gcc,
            **metrics
        }


# ------------------------------------------------------------------
# Q-Robust 包装器 (2024_TMC_QRobust)
# ------------------------------------------------------------------

class QRobustWrapper(NetworkReconstructionAlgorithm):
    """Q-Robust: 量子启发式鲁棒性优化"""

    def __init__(self, config=None):
        super().__init__(config)
        self.max_gen = config.get('qrobust_max_gen', 10) if config else 10

    def construct(self, G, target_edges=None, **kwargs):
        from .Q_Robust import Q_Robust_method
        G_opt = Q_Robust_method(G, max_gen=self.max_gen, use_optimized=True)
        added = list(set(G_opt.edges()) - set(G.edges()))
        return {
            'G_constructed': G_opt,
            'added_edges': added,
            'improvement_auc': 0.0,
            'improvement_pc': 0.0
        }

    def dismantle(self, G, stop_condition='threshold', stop_threshold=0.5):
        result = self.construct(G)
        Gc = result['G_constructed']
        seq = _high_degree_attack_sequence(Gc)
        gcc = self.run_attack(Gc, seq)
        metrics = compute_all_metrics(gcc, Gc.number_of_nodes(), threshold=stop_threshold)
        return {
            'attack_sequence': seq,
            'gcc_sizes': gcc,
            **metrics
        }


# ------------------------------------------------------------------
# ROMEN 包装器 (2024_ToN_ROMEM)
# ------------------------------------------------------------------

class ROMENWrapper(NetworkReconstructionAlgorithm):
    """ROMEN: 基于SAC的拓扑优化"""

    def __init__(self, config=None):
        super().__init__(config)
        self.population_size = config.get('romen_population_size', 5) if config else 5
        self.generations = config.get('romen_generations', 50) if config else 50
        self.steps = config.get('romen_steps', 30) if config else 30

    def construct(self, G, target_edges=None, **kwargs):
        from .ROMEN import optimize_network_robustness
        pos = _extract_positions(G)
        if pos is None:
            pos = _generate_random_positions(G, area_size=1000, seed=42)
        result = optimize_network_robustness(
            initial_graph=G,
            positions=pos,
            population_size=self.population_size,
            generations=self.generations,
            verbose=False
        )
        G_opt = result.get('optimized_graph', G)
        added = list(set(G_opt.edges()) - set(G.edges()))
        return {
            'G_constructed': G_opt,
            'added_edges': added,
            'improvement_auc': 0.0,
            'improvement_pc': 0.0
        }

    def dismantle(self, G, stop_condition='threshold', stop_threshold=0.5):
        result = self.construct(G)
        Gc = result['G_constructed']
        seq = _high_degree_attack_sequence(Gc)
        gcc = self.run_attack(Gc, seq)
        metrics = compute_all_metrics(gcc, Gc.number_of_nodes(), threshold=stop_threshold)
        return {
            'attack_sequence': seq,
            'gcc_sizes': gcc,
            **metrics
        }


# ------------------------------------------------------------------
# UNITY 包装器 (2025_ToC_UNITY)
# ------------------------------------------------------------------

class UNITYWrapper(NetworkReconstructionAlgorithm):
    """UNITY: 统一鲁棒网络拓扑生成（边数归一化版）"""

    def __init__(self, config=None):
        super().__init__(config)
        self.seed = config.get('unity_seed', 3) if config else 3

    def construct(self, G, target_edges=None, **kwargs):
        import random, numpy as np
        from .UNITY import UNITY
        # 使用固定seed以产生与论文一致的典型拓扑
        random.seed(self.seed)
        np.random.seed(self.seed)
        unity = UNITY(area_diameter=500, comm_range=200)
        G_opt = unity.generate_matched_topology(G)
        # 若通信范围限制导致边数不足，放宽范围重试
        if G_opt.number_of_edges() < G.number_of_edges():
            unity_wide = UNITY(area_diameter=500, comm_range=1000)
            random.seed(self.seed)
            np.random.seed(self.seed)
            G_opt = unity_wide.generate_matched_topology(G)
        # 强制约束检查
        assert G_opt.number_of_nodes() == G.number_of_nodes(), \
            f"UNITY节点数不匹配: {G_opt.number_of_nodes()} != {G.number_of_nodes()}"
        assert G_opt.number_of_edges() == G.number_of_edges(), \
            f"UNITY边数不匹配: {G_opt.number_of_edges()} != {G.number_of_edges()}"
        added = list(set(G_opt.edges()) - set(G.edges()))
        return {
            'G_constructed': G_opt,
            'added_edges': added,
            'improvement_auc': 0.0,
            'improvement_pc': 0.0
        }

    def dismantle(self, G, stop_condition='threshold', stop_threshold=0.5):
        result = self.construct(G)
        Gc = result['G_constructed']
        seq = _high_degree_attack_sequence(Gc)
        gcc = self.run_attack(Gc, seq)
        metrics = compute_all_metrics(gcc, Gc.number_of_nodes(), threshold=stop_threshold)
        return {
            'attack_sequence': seq,
            'gcc_sizes': gcc,
            **metrics
        }


# ------------------------------------------------------------------
# FRED-ABL 包装器 (2025_TMC_FRED-ABL)
# ------------------------------------------------------------------

class FREDABLWrapper(NetworkReconstructionAlgorithm):
    """FRED-ABL: 贝叶斯优化鲁棒拓扑"""

    def __init__(self, config=None):
        super().__init__(config)
        self.iterations = config.get('fredabl_iterations', 15) if config else 15

    def construct(self, G, target_edges=None, **kwargs):
        from .fred_abl_optimization import IIoTEnv, FRED_ABL
        pos = _extract_positions(G)
        if pos is None:
            pos = _generate_random_positions(G, area_size=1.0, seed=42)
        # FRED-ABL需要归一化位置[0,1]
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        scale = max(max_x - min_x, max_y - min_y, 1.0)
        norm_pos = {}
        G_norm = nx.Graph()
        G_norm.add_nodes_from(range(G.number_of_nodes()))
        for i, n in enumerate(G.nodes()):
            norm_pos[i] = ((pos[n][0] - min_x) / scale, (pos[n][1] - min_y) / scale)
            G_norm.nodes[i]['pos'] = norm_pos[i]
        # 添加边（重编号）
        node_map = {n: i for i, n in enumerate(G.nodes())}
        for u, v in G.edges():
            G_norm.add_edge(node_map[u], node_map[v])
        env = IIoTEnv(G=G_norm, comm_range=0.4)
        fred = FRED_ABL(env)
        best_adj = fred.run(iterations=self.iterations, initial_samples=3, verbose=False)
        G_opt = _adj_to_graph(best_adj, norm_pos)
        G_opt = nx.Graph(G_opt)
        G_opt.remove_edges_from(nx.selfloop_edges(G_opt))
        # 确保连通性：如果不连通，提取最大连通分量并从possible_edges补充边
        if not nx.is_connected(G_opt) and G_opt.number_of_nodes() > 0:
            gcc = max(nx.connected_components(G_opt), key=len)
            G_opt = G_opt.subgraph(gcc).copy()
            # 补充边以达到目标边数
            target_edges = G_norm.number_of_edges()
            while G_opt.number_of_edges() < target_edges:
                # 从possible_edges中随机添加不在图中的边
                candidates = [e for e in env.possible_edges if not G_opt.has_edge(e[0], e[1])]
                if not candidates:
                    break
                u, v = random.choice(candidates)
                G_opt.add_edge(u, v)
        added = list(set(G_opt.edges()) - set(G_norm.edges()))
        return {
            'G_constructed': G_opt,
            'added_edges': added,
            'improvement_auc': 0.0,
            'improvement_pc': 0.0
        }

    def dismantle(self, G, stop_condition='threshold', stop_threshold=0.5):
        result = self.construct(G)
        Gc = result['G_constructed']
        seq = _high_degree_attack_sequence(Gc)
        gcc = self.run_attack(Gc, seq)
        metrics = compute_all_metrics(gcc, Gc.number_of_nodes(), threshold=stop_threshold)
        return {
            'attack_sequence': seq,
            'gcc_sizes': gcc,
            **metrics
        }


# ------------------------------------------------------------------
# Onion 包装器 (2024_TMC_no)
# ------------------------------------------------------------------

class OnionWrapper(NetworkReconstructionAlgorithm):
    """Onion: 洋葱网络结构优化"""

    def __init__(self, config=None):
        super().__init__(config)

    def construct(self, G, target_edges=None, **kwargs):
        from .onion import OnionNetworkOptimizer
        optimizer = OnionNetworkOptimizer(G)
        G_opt = optimizer.optimize_network(max_iterations=2000, verbose=False)
        added = list(set(G_opt.edges()) - set(G.edges()))
        return {
            'G_constructed': G_opt,
            'added_edges': added,
            'improvement_auc': 0.0,
            'improvement_pc': 0.0
        }

    def dismantle(self, G, stop_condition='threshold', stop_threshold=0.5):
        result = self.construct(G)
        Gc = result['G_constructed']
        seq = _high_degree_attack_sequence(Gc)
        gcc = self.run_attack(Gc, seq)
        metrics = compute_all_metrics(gcc, Gc.number_of_nodes(), threshold=stop_threshold)
        return {
            'attack_sequence': seq,
            'gcc_sizes': gcc,
            **metrics
        }


# ------------------------------------------------------------------
# QDLM 包装器 (量子驱动学习模型)
# ------------------------------------------------------------------

class QDLMWrapper(NetworkReconstructionAlgorithm):
    """QDLM: 量子驱动高效学习模型"""

    def __init__(self, config=None):
        super().__init__(config)
        self.pop_size = config.get('qdlm_pop_size', 20) if config else 20
        self.max_iterations = config.get('qdlm_max_iterations', 30) if config else 30

    def construct(self, G, target_edges=None, **kwargs):
        from .qdlm_optimization import QuantumEnvironment, QuantumLearningModel
        pos = _extract_positions(G)
        if pos is None:
            pos = _generate_random_positions(G, area_size=1.0, seed=42)
        # QDLM需要归一化位置[0,1]
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        scale = max(max_x - min_x, max_y - min_y, 1.0)
        norm_pos = {}
        G_norm = nx.Graph()
        G_norm.add_nodes_from(range(G.number_of_nodes()))
        for i, n in enumerate(G.nodes()):
            norm_pos[i] = ((pos[n][0] - min_x) / scale, (pos[n][1] - min_y) / scale)
            G_norm.nodes[i]['pos'] = norm_pos[i]
        node_map = {n: i for i, n in enumerate(G.nodes())}
        for u, v in G.edges():
            G_norm.add_edge(node_map[u], node_map[v])
        env = QuantumEnvironment(G=G_norm, comm_range=0.4)
        if env.m == 0:
            # 无有效边，直接返回原图
            return {
                'G_constructed': G_norm,
                'added_edges': [],
                'improvement_auc': 0.0,
                'improvement_pc': 0.0
            }
        qlm = QuantumLearningModel(env, pop_size=self.pop_size, max_iterations=self.max_iterations)
        best_adj = qlm.run(verbose=False)
        G_opt = _adj_to_graph(best_adj, norm_pos)
        G_opt = nx.Graph(G_opt)
        G_opt.remove_edges_from(nx.selfloop_edges(G_opt))
        added = list(set(G_opt.edges()) - set(G_norm.edges()))
        return {
            'G_constructed': G_opt,
            'added_edges': added,
            'improvement_auc': 0.0,
            'improvement_pc': 0.0
        }

    def dismantle(self, G, stop_condition='threshold', stop_threshold=0.5):
        result = self.construct(G)
        Gc = result['G_constructed']
        seq = _high_degree_attack_sequence(Gc)
        gcc = self.run_attack(Gc, seq)
        metrics = compute_all_metrics(gcc, Gc.number_of_nodes(), threshold=stop_threshold)
        return {
            'attack_sequence': seq,
            'gcc_sizes': gcc,
            **metrics
        }


# ------------------------------------------------------------------
# SmartTRO 包装器 (2022_CN_SmartTRO)
# ------------------------------------------------------------------

class SmartTROWrapper(NetworkReconstructionAlgorithm):
    """SmartTRO: 基于GCN+DRL的拓扑优化"""

    def __init__(self, config=None):
        super().__init__(config)
        self.num_envs = config.get('smarttro_num_envs', 1) if config else 1
        self.max_actions = config.get('smarttro_max_actions', None) if config else None
        self.model_path = config.get('smarttro_model_path', 'models/smarttro_final.pt') if config else 'models/smarttro_final.pt'

    def construct(self, G, target_edges=None, **kwargs):
        from .smartTRO import SmartTRO
        import os
        edge_density = max(1, int(2 * G.number_of_edges() / G.number_of_nodes()))
        agent = SmartTRO(num_nodes=G.number_of_nodes(), edge_density=edge_density, num_agents=1)
        if self.max_actions is not None:
            agent.env.max_actions = self.max_actions
        # 加载预训练模型（如果存在）
        if os.path.exists(self.model_path):
            print(f"[SmartTRO] Loading pretrained model from {self.model_path}")
            agent.load_model(self.model_path)
        else:
            print(f"[SmartTRO] No pretrained model found at {self.model_path}, using random init")
        G_opt = agent.optimize_topology(G)
        added = list(set(G_opt.edges()) - set(G.edges()))
        return {
            'G_constructed': G_opt,
            'added_edges': added,
            'improvement_auc': 0.0,
            'improvement_pc': 0.0
        }

    def dismantle(self, G, stop_condition='threshold', stop_threshold=0.5):
        result = self.construct(G)
        Gc = result['G_constructed']
        seq = _high_degree_attack_sequence(Gc)
        gcc = self.run_attack(Gc, seq)
        metrics = compute_all_metrics(gcc, Gc.number_of_nodes(), threshold=stop_threshold)
        return {
            'attack_sequence': seq,
            'gcc_sizes': gcc,
            **metrics
        }


# ------------------------------------------------------------------
# TEAM 包装器 (2025_TMC_TEAM)
# ------------------------------------------------------------------

class TEAMWrapper(NetworkReconstructionAlgorithm):
    """TEAM: 多目标遗传算法拓扑优化"""

    def __init__(self, config=None):
        super().__init__(config)

    def construct(self, G, target_edges=None, **kwargs):
        from .team_optimization import IoTNetwork, TEAM_Engine
        # 从G构建IoTNetwork
        iot_net = IoTNetwork(G=G)
        engine = TEAM_Engine(iot_net)
        history = engine.run(verbose=False)
        # TEAM返回的是历史记录，需要从中提取最优解
        # 取最后一个作为最终拓扑
        G_opt = iot_net.graph.copy()
        added = list(set(G_opt.edges()) - set(G.edges()))
        return {
            'G_constructed': G_opt,
            'added_edges': added,
            'improvement_auc': 0.0,
            'improvement_pc': 0.0,
            'history': history
        }

    def dismantle(self, G, stop_condition='threshold', stop_threshold=0.5):
        result = self.construct(G)
        Gc = result['G_constructed']
        seq = _high_degree_attack_sequence(Gc)
        gcc = self.run_attack(Gc, seq)
        metrics = compute_all_metrics(gcc, Gc.number_of_nodes(), threshold=stop_threshold)
        return {
            'attack_sequence': seq,
            'gcc_sizes': gcc,
            **metrics
        }
