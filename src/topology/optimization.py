# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np

# 导入拆分后的模块
from src.topology.genetic_optimization import solution
from src.topology.onion_optimization import OnionOptimizer
from src.topology.romen_optimization import ROHEMOptimizer
from src.topology.unity_optimization import UNITY

# ----------------- Helper Functions (保持兼容性) -----------------

def construct_GA(G, controllers_rate):
    nodes_num = G.number_of_nodes()
    edges_num = G.number_of_edges()
    best_solution, _ = solution(nodes_num, edges_num)
    G_optimized = nx.from_numpy_array(best_solution)
    return G_optimized

def construct_onion(G, max_iter=100, verbose=False):
    optimizer = OnionOptimizer(G) 
    G_onion = optimizer.optimize_network(max_iterations=max_iter, verbose=verbose)
    return G_onion

def construct_romen(G, seed=None, verbose=False):
    """
    构建ROMEN优化拓扑。
    
    Args:
        G: 初始图
        seed: 随机种子（用于可重复性）
        verbose: 是否打印详细信息
    """
    import random
    import numpy as np
    
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    optimizer = ROHEMOptimizer(
        population_size=2,  # 减少种群大小
        generations=15,     # 减少代数
        verbose=verbose
    )
    G_romen = optimizer.optimize_topology(
        initial_graph=G,
        area_size=500,
        comm_distance=200
    )
    return G_romen

def construct_unity(G):
    unity = UNITY()
    G_unity = unity.generate_matched_topology(G)
    return G_unity

def _adjust_edge_count_to_target(G_optimized, target_edges, G_original, seed=None, comm_range=None):
    """
    调整优化后的图边数到目标值，同时考虑通信半径限制。
    
    Args:
        G_optimized: 优化后的图
        target_edges: 目标边数（原始图的边数）
        G_original: 原始图（用于获取节点位置和通信范围）
        seed: 随机种子
        comm_range: 通信半径（如果提供，只添加在通信范围内的边）
    
    Returns:
        调整后的图
    """
    import random
    import numpy as np
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    current_edges = G_optimized.number_of_edges()
    n = G_optimized.number_of_nodes()
    
    if current_edges == target_edges:
        return G_optimized
    
    # 获取节点位置（用于计算距离）
    node_positions = {}
    for i in G_optimized.nodes():
        if 'pos' in G_optimized.nodes[i]:
            node_positions[i] = G_optimized.nodes[i]['pos']
        elif i < G_original.number_of_nodes() and 'pos' in G_original.nodes[i]:
            node_positions[i] = G_original.nodes[i]['pos']
            G_optimized.nodes[i]['pos'] = G_original.nodes[i]['pos']
    
    if current_edges < target_edges:
        # 需要添加边
        need_more = target_edges - current_edges
        
        # 获取所有可能的边（不在当前图中的）
        all_possible = []
        for u in range(n):
            for v in range(u + 1, n):
                if not G_optimized.has_edge(u, v):
                    # 如果提供了通信范围，检查距离
                    if comm_range is not None and node_positions:
                        if u in node_positions and v in node_positions:
                            dist = np.linalg.norm(
                                np.array(node_positions[u]) - np.array(node_positions[v])
                            )
                            if dist > comm_range:
                                continue  # 跳过超出通信范围的边
                    all_possible.append((u, v))
        
        # 随机选择边添加
        if len(all_possible) >= need_more:
            edges_to_add = random.sample(all_possible, need_more)
            G_optimized.add_edges_from(edges_to_add)
        else:
            # 如果可能的边不够，添加所有可能的边
            G_optimized.add_edges_from(all_possible)
            # 如果还不够，忽略通信范围限制添加边
            remaining = target_edges - G_optimized.number_of_edges()
            if remaining > 0:
                all_remaining = [(u, v) for u in range(n) for v in range(u + 1, n) 
                               if not G_optimized.has_edge(u, v)]
                if len(all_remaining) >= remaining:
                    edges_to_add = random.sample(all_remaining, remaining)
                    G_optimized.add_edges_from(edges_to_add)
    
    elif current_edges > target_edges:
        # 需要删除边
        need_remove = current_edges - target_edges
        
        # 优先删除度数较低的节点之间的边（保持图的鲁棒性）
        degrees = dict(G_optimized.degree())
        edges_list = list(G_optimized.edges())
        
        # 按边的度数总和排序，优先删除连接低度节点的边
        def edge_priority(edge):
            u, v = edge
            return degrees.get(u, 0) + degrees.get(v, 0)
        
        edges_list.sort(key=edge_priority)  # 从小到大排序，优先删除低度边
        
        # 删除边，但要保持图的连通性
        removed = 0
        for u, v in edges_list:
            if removed >= need_remove:
                break
            # 检查删除这条边后图是否仍然连通
            G_optimized.remove_edge(u, v)
            if nx.is_connected(G_optimized):
                removed += 1
            else:
                # 如果不连通，恢复这条边
                G_optimized.add_edge(u, v)
        
        # 如果还有需要删除的边，强制删除（可能破坏连通性）
        if removed < need_remove:
            remaining_edges = list(G_optimized.edges())
            random.shuffle(remaining_edges)
            for u, v in remaining_edges[:need_remove - removed]:
                G_optimized.remove_edge(u, v)
    
    return G_optimized


def _ensure_node_positions(G, pos_dict=None):
    """
    确保图节点有位置属性。
    如果图没有位置属性，使用 spring_layout 生成。
    
    Args:
        G: NetworkX 图
        pos_dict: 可选的位置字典（从 load_graph 返回）
    
    Returns:
        G: 带有位置属性的图
    """
    G = G.copy()
    n = G.number_of_nodes()
    
    # 检查是否已有位置属性
    has_pos = all('pos' in G.nodes[i] for i in G.nodes())
    
    if not has_pos:
        # 如果有外部位置字典，使用它
        if pos_dict is not None:
            for node in G.nodes():
                if node in pos_dict and pos_dict[node] != (None, None):
                    G.nodes[node]['pos'] = pos_dict[node]
                else:
                    # 如果没有位置，使用随机位置
                    G.nodes[node]['pos'] = (np.random.random(), np.random.random())
        else:
            # 使用 spring_layout 生成位置
            pos = nx.spring_layout(G, scale=1.0, seed=42)
            for node in G.nodes():
                G.nodes[node]['pos'] = tuple(pos[node])
    
    # 确保节点ID是连续的0到n-1
    if sorted(G.nodes()) != list(range(n)):
        mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(G.nodes()))}
        G = nx.relabel_nodes(G, mapping)
        # 重新设置位置属性
        for old_id, new_id in mapping.items():
            if 'pos' in G.nodes[new_id]:
                pass  # 位置已经随节点重命名
    
    return G

def construct_fred_abl(G, seed=None, comm_range=0.3, iterations=15, initial_samples=3, verbose=False):
    """
    构建 FRED-ABL 优化拓扑（优化版本：减少迭代次数，使用采样加速）。
    
    Args:
        G: 初始图
        seed: 随机种子
        comm_range: 通信半径（归一化到[0,1]范围）
        iterations: 优化迭代次数（默认15，已优化）
        initial_samples: 初始采样数（默认3，已优化）
        verbose: 是否打印详细信息
    
    Returns:
        G_optimized: 优化后的图
    """
    import random
    import numpy as np
    
    # 检查 sklearn 是否可用
    try:
        import sklearn
    except ImportError:
        raise ImportError(
            "FRED-ABL 算法需要 scikit-learn 库。请运行以下命令安装：\n"
            "pip install scikit-learn>=1.3.0"
        )
    
    from src.topology.fred_abl_optimization import IIoTEnv, FRED_ABL
    
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 确保图有位置属性
    G_with_pos = _ensure_node_positions(G)
    
    # 创建环境
    env = IIoTEnv(G=G_with_pos, comm_range=comm_range, seed=seed if seed else 42)
    
    # 初始化优化器
    fred = FRED_ABL(env)
    
    # 运行算法（使用优化参数）
    best_adj = fred.run(iterations=iterations, initial_samples=initial_samples, verbose=verbose)
    
    # 转换为 NetworkX 图
    G_optimized = nx.from_numpy_array(best_adj)
    
    # 保留位置属性
    for i in G_optimized.nodes():
        if i < G_with_pos.number_of_nodes() and 'pos' in G_with_pos.nodes[i]:
            G_optimized.nodes[i]['pos'] = G_with_pos.nodes[i]['pos']
    
    # 调整边数以匹配原始图（考虑通信范围）
    target_edges = G.number_of_edges()
    G_optimized = _adjust_edge_count_to_target(
        G_optimized, target_edges, G_with_pos, seed=seed, comm_range=comm_range
    )
    
    return G_optimized

def construct_team(G, seed=None, verbose=False):
    """
    构建 TEAM 优化拓扑（优化版本：减少迭代次数，使用采样加速）。
    
    Args:
        G: 初始图
        seed: 随机种子
        verbose: 是否打印详细信息
    
    Returns:
        G_optimized: 优化后的图
    """
    import random
    import numpy as np
    from src.topology.team_optimization import IoTNetwork, TEAM_Engine, CONF
    
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 确保图有位置属性
    G_with_pos = _ensure_node_positions(G)
    
    # 创建网络环境
    iot_net = IoTNetwork(G=G_with_pos)
    
    # 检查是否有超级节点
    if len(iot_net.super_nodes) == 0:
        # 如果没有超级节点，返回原图
        return G_with_pos
    
    # 临时减少迭代次数以加速（如果默认值太大）
    original_max_gen = CONF['MAX_GEN']
    CONF['MAX_GEN'] = min(20, CONF['MAX_GEN'])  # 最多20代
    
    try:
        # 初始化算法引擎
        team_solver = TEAM_Engine(iot_net)
        
        # 运行演化
        team_solver.run(verbose=verbose)
    finally:
        # 恢复原始配置
        CONF['MAX_GEN'] = original_max_gen
    
    # 获取重构后的图
    G_optimized = team_solver.get_reconstructed_graph()
    
    # 如果返回None，使用原图
    if G_optimized is None:
        G_optimized = G_with_pos.copy()
    
    # 保留位置属性
    for i in G_optimized.nodes():
        if i < G_with_pos.number_of_nodes() and 'pos' in G_with_pos.nodes[i]:
            G_optimized.nodes[i]['pos'] = G_with_pos.nodes[i]['pos']
    
    # 调整边数以匹配原始图
    target_edges = G.number_of_edges()
    G_optimized = _adjust_edge_count_to_target(G_optimized, target_edges, G_with_pos, seed=seed)
    
    return G_optimized

def construct_qdlm(G, seed=None, comm_range=0.3, pop_size=20, max_iterations=25, verbose=False):
    """
    构建 QDLM (Quantum Learning Model) 优化拓扑（优化版本：减少迭代次数和种群大小，使用采样加速）。
    
    Args:
        G: 初始图
        seed: 随机种子
        comm_range: 通信半径（归一化到[0,1]范围）
        pop_size: 种群大小（默认20，已优化）
        max_iterations: 最大迭代次数（默认25，已优化）
        verbose: 是否打印详细信息
    
    Returns:
        G_optimized: 优化后的图
    """
    import random
    import numpy as np
    from src.topology.qdlm_optimization import QuantumEnvironment, QuantumLearningModel
    
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 确保图有位置属性
    G_with_pos = _ensure_node_positions(G)
    
    # 创建环境
    env = QuantumEnvironment(G=G_with_pos, comm_range=comm_range, seed=seed if seed else 42)
    
    # 初始化量子学习模型（使用优化参数）
    qlm = QuantumLearningModel(env, pop_size=pop_size, max_iterations=max_iterations)
    
    # 运行优化
    best_adj = qlm.run(verbose=verbose)
    
    # 转换为 NetworkX 图
    G_optimized = nx.from_numpy_array(best_adj)
    
    # 保留位置属性
    for i in G_optimized.nodes():
        if i < G_with_pos.number_of_nodes() and 'pos' in G_with_pos.nodes[i]:
            G_optimized.nodes[i]['pos'] = G_with_pos.nodes[i]['pos']
    
    # 调整边数以匹配原始图（考虑通信范围）
    target_edges = G.number_of_edges()
    G_optimized = _adjust_edge_count_to_target(
        G_optimized, target_edges, G_with_pos, seed=seed, comm_range=comm_range
    )
    
    return G_optimized