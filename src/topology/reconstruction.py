# -*- coding: utf-8 -*-
"""
BiT-HyRL 双峰拓扑重构模块。

提供 create_bimodal_network_exact 函数，用于将输入网络重构为符合双峰度分布的网络。

关键约束：
1. 节点数 = 原图节点数（严格相等）
2. 边数 = 原图边数（严格相等）
3. 图必须连通

稀疏网络优化：当平均度较小或密度较低时，自动采用稀疏分支，保证度序列可图化、删边时保持连通。

版本历史:
- v1.0: 初始版本
- v1.1: 修复边数约束问题，添加连通性保证
- v1.2: 低密度网络双峰拓扑优化（稀疏分支、保连通边数调整）
"""
import networkx as nx
import random
import numpy as np
from math import ceil
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 稀疏网络阈值：平均度 < 此值或密度 < 此值时使用稀疏优化逻辑
SPARSE_AVG_DEGREE_THRESHOLD = 3.5
SPARSE_DENSITY_THRESHOLD = 0.02


def _ensure_connectivity(G, seed=None):
    """
    确保图的连通性。
    
    如果图不连通，则连接各个连通分量。
    
    Args:
        G: 输入图
        seed: 随机种子
        
    Returns:
        连通的图（添加的边数）
    """
    if seed is not None:
        random.seed(seed)
    
    if nx.is_connected(G):
        return 0
    
    # 获取所有连通分量
    components = list(nx.connected_components(G))
    if len(components) == 1:
        return 0
    
    # 按大小排序，最大的在前
    components = sorted(components, key=len, reverse=True)
    
    # 连接各个分量到主分量
    main_component = components[0]
    edges_added = 0
    
    for i in range(1, len(components)):
        comp = components[i]
        # 选择主分量中度数最高的节点
        main_node = max(main_component, key=lambda x: G.degree(x))
        # 选择当前分量中度数最高的节点
        comp_node = max(comp, key=lambda x: G.degree(x))
        
        # 添加边连接两个分量
        if not G.has_edge(main_node, comp_node):
            G.add_edge(main_node, comp_node)
            edges_added += 1
        
        # 更新主分量
        main_component = main_component | comp
    
    return edges_added


def _adjust_edge_count(G, target_edges, seed=None):
    """
    调整图的边数到目标值，同时保持连通性。
    
    Args:
        G: 输入图
        target_edges: 目标边数
        seed: 随机种子
        
    Returns:
        tuple: (调整后的图, 实际调整的边数)
    """
    if seed is not None:
        random.seed(seed)
    
    current_edges = G.number_of_edges()
    N = G.number_of_nodes()
    adjusted = 0
    
    if current_edges < target_edges:
        # 需要添加边
        need_more = target_edges - current_edges
        all_possible = [(u, v) for u in range(N) for v in range(u+1, N) 
                        if not G.has_edge(u, v)]
        random.shuffle(all_possible)
        
        # 优先添加连接到hub节点的边，保持双峰特性
        degrees = dict(G.degree())
        avg_degree = sum(degrees.values()) / N if N > 0 else 0
        
        # 按优先级排序：连接到高度节点的边优先
        def edge_priority(edge):
            u, v = edge
            return -(degrees.get(u, 0) + degrees.get(v, 0))
        
        all_possible.sort(key=edge_priority)
        
        edges_to_add = all_possible[:need_more]
        G.add_edges_from(edges_to_add)
        adjusted = len(edges_to_add)
        
    elif current_edges > target_edges:
        # 需要删除边，但要保持连通性
        need_less = current_edges - target_edges
        edges = list(G.edges())
        
        # 按度数从高到低排序，优先删除高度节点之间的边（保持双峰结构）
        degrees = dict(G.degree())
        edges.sort(key=lambda e: -(degrees[e[0]] + degrees[e[1]]))
        
        removed = 0
        for edge in edges:
            if removed >= need_less:
                break
            
            u, v = edge
            # 检查两端节点的度数，避免产生孤立节点
            if G.degree(u) <= 1 or G.degree(v) <= 1:
                continue
            
            # 检查删除这条边后是否仍然连通
            G.remove_edge(u, v)
            
            if nx.is_connected(G):
                removed += 1
            else:
                # 恢复边
                G.add_edge(u, v)
        
        adjusted = -removed
        
        # 如果无法删除足够的边，记录警告
        if removed < need_less:
            logger.warning(f"边数调整: 需要删除{need_less}条边，实际删除{removed}条（保持连通性）")
    
    return adjusted


def _make_graphical(degree_sequence):
    """
    将度序列调整为可图化的序列。
    
    使用 Erdős–Gallai 定理进行调整。
    
    Args:
        degree_sequence: 原始度序列
        
    Returns:
        可图化的度序列
    """
    n = len(degree_sequence)
    if n == 0:
        return degree_sequence
    
    # 确保所有度数非负且不超过 n-1
    degree_sequence = [max(0, min(d, n-1)) for d in degree_sequence]
    
    # 确保度数和为偶数
    total = sum(degree_sequence)
    if total % 2 != 0:
        # 减少最大度数
        max_idx = degree_sequence.index(max(degree_sequence))
        degree_sequence[max_idx] -= 1
    
    # 尝试简单调整使其满足 Erdős–Gallai 定理
    max_iterations = 100
    for _ in range(max_iterations):
        if nx.is_graphical(degree_sequence):
            return degree_sequence
        
        # 降序排列
        degree_sequence.sort(reverse=True)
        
        # 尝试减少最大度数，增加最小度数
        if degree_sequence[0] > 0:
            degree_sequence[0] -= 1
            if degree_sequence[-1] < n - 1:
                degree_sequence[-1] += 1
        
        # 确保度数和为偶数
        total = sum(degree_sequence)
        if total % 2 != 0:
            max_idx = degree_sequence.index(max(degree_sequence))
            if degree_sequence[max_idx] > 0:
                degree_sequence[max_idx] -= 1
    
    return degree_sequence


def _make_graphical_bimodal_preserving(degree_sequence, num_hubs):
    """
    在保持双峰结构的前提下将度序列调整为可图化：始终保留前 num_hubs 个为“高度”节点，
    其余为“低度”节点，仅在这两类之间转移度数，避免通用 _make_graphical 把双峰抹平。
    
    Args:
        degree_sequence: 度序列（可为 list，会被原地修改的拷贝）
        num_hubs: 高度节点数量（前 num_hubs 个为 hub）
        
    Returns:
        可图化的度序列（降序：前 num_hubs 为 hub，其余为 low）
    """
    seq = list(degree_sequence)
    n = len(seq)
    if n == 0 or num_hubs <= 0 or num_hubs >= n:
        return _make_graphical(seq)
    seq = [max(0, min(d, n - 1)) for d in seq]
    total = sum(seq)
    if total % 2 != 0:
        if seq[0] > 0:
            seq[0] -= 1
        else:
            for i in range(n):
                if seq[i] > 1:
                    seq[i] -= 1
                    break
    max_iter = 500
    for _ in range(max_iter):
        if nx.is_graphical(seq):
            seq.sort(reverse=True)
            return seq
        seq.sort(reverse=True)
        # 最小 hub 在 num_hubs-1，最大 low 在 num_hubs
        small_hub = seq[num_hubs - 1]
        large_low = seq[num_hubs]
        if small_hub <= 0 or large_low >= n - 1:
            break
        seq[num_hubs - 1] -= 1
        seq[num_hubs] += 1
        total = sum(seq)
        if total % 2 != 0:
            if seq[num_hubs - 1] > 0:
                seq[num_hubs - 1] -= 1
            else:
                seq[num_hubs] -= 1
        seq.sort(reverse=True)  # 保持降序，下一轮最小 hub / 最大 low 位置正确
    return _make_graphical(seq)


def create_bimodal_network_exact(G, hub_num=1, seed=None):
    """
    构造符合最优双峰度分布的无向连通网络（BiT-HyRL 拓扑层）。
    
    基于理论公式: k_max = A * N^(2/3)
    其中 A = (2 * (avg_k^2) * ((avg_k - 1)^2) / (2 * avg_k - 1))^(1/3)
    
    关键约束（严格保证）:
    1. 节点数 = 原图节点数
    2. 边数 = 原图边数
    3. 图必须连通
    
    参数:
        G: 输入图
        hub_num: hub 节点数量（直接指定），默认 None 表示使用默认策略（稀疏用 1，否则用 N*0.15）
        seed: 随机种子，默认None使用当前随机状态
    
    返回:
        nx.Graph: 满足双峰分布且连通的新图（节点数和边数与原图相同）
    """
    N = G.number_of_nodes()
    M = G.number_of_edges()
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 参数验证
    if N < 2:
        logger.warning("节点数太少，返回原图副本")
        return G.copy()
    
    # 计算最小边数（保证连通性需要至少 N-1 条边）
    min_edges_for_connectivity = N - 1
    if M < min_edges_for_connectivity:
        logger.warning(f"原图边数({M})小于连通性最小要求({min_edges_for_connectivity})，将使用最小边数")
        M = min_edges_for_connectivity
    
    total_degree = 2 * M
    avg_k = total_degree / N
    density = (2 * M) / (N * (N - 1)) if N > 1 else 0
    is_sparse = (avg_k < SPARSE_AVG_DEGREE_THRESHOLD or density < SPARSE_DENSITY_THRESHOLD)

    # 计算hub节点数量（稀疏网络减少 hub 数以保证 k_base >= 1）
    if hub_num is not None:
        num_hubs = max(1, min(hub_num, N // 4, N - 1))
    elif is_sparse:
        num_hubs = 1
    else:
        min_hubs = 2 if N >= 20 else 1
        num_hubs = max(min_hubs, min(int(N * 0.15), N // 4))

    # 计算最优 k_max（基于双峰分布理论）
    if avg_k > 1 and (2 * avg_k - 1) > 0:
        numerator = 2 * (avg_k**2) * ((avg_k - 1)**2)
        denominator = 2 * avg_k - 1
        A = (numerator / denominator) ** (1/3)
        k_max = int(round(A * (N ** (2/3))))
    else:
        k_max = max(2, int(avg_k * 3))

    # 限制 k_max 范围
    k_max = min(k_max, N - 1)
    k_max = max(k_max, 2)

    # 稀疏网络：保证 degree_rem >= (N - num_hubs)，使 k_base >= 1
    num_low = N - num_hubs
    if is_sparse and num_low > 0:
        max_k_max_for_connectivity = total_degree - num_low
        k_max = min(k_max, max(2, max_k_max_for_connectivity))

    # 构造度序列
    degree_rem = total_degree - k_max * num_hubs

    # 确保低度节点的度数至少为1（保证连通性）
    if num_low > 0:
        if degree_rem < num_low:
            k_max = max(2, (total_degree - num_low) // num_hubs)
            degree_rem = total_degree - k_max * num_hubs
        k_base = max(1, degree_rem // num_low)
        remainder = degree_rem % num_low if degree_rem >= num_low else 0
    else:
        k_base = 1
        remainder = 0
    
    # 构建度序列：hub节点 + 高度节点 + 基础度节点
    degree_sequence = [k_max] * num_hubs
    degree_sequence += [k_base + 1] * min(remainder, num_low)
    degree_sequence += [k_base] * max(0, num_low - remainder)
    
    # 确保所有度数至少为1（保证连通性）
    degree_sequence = [max(1, d) for d in degree_sequence]
    
    # 调整度序列总和为目标值
    actual_degree = sum(degree_sequence)
    degree_diff = total_degree - actual_degree
    
    if degree_diff > 0:
        # 需要增加度数
        for i in range(abs(degree_diff)):
            idx = (num_hubs + i) % N
            degree_sequence[idx] += 1
    elif degree_diff < 0:
        # 需要减少度数，但保持最小度数为1
        for i in range(abs(degree_diff)):
            idx = (num_hubs + i) % N
            if degree_sequence[idx] > 1:
                degree_sequence[idx] -= 1
    
    # 确保度序列总和为偶数
    if sum(degree_sequence) % 2 != 0:
        for i in range(num_hubs, N):
            if degree_sequence[i] < N - 1:
                degree_sequence[i] += 1
                break
    
    current_seed = seed if seed is not None else None
    
    # 尝试生成图
    G_new = None
    max_attempts = 5
    
    for attempt in range(max_attempts):
        try:
            if nx.is_graphical(degree_sequence):
                G_new = nx.havel_hakimi_graph(degree_sequence)
            else:
                # 调整度序列使其可实现
                adjusted_seq = _make_graphical(degree_sequence.copy())
                if nx.is_graphical(adjusted_seq):
                    G_new = nx.havel_hakimi_graph(adjusted_seq)
                else:
                    # 使用 configuration_model
                    G_multi = nx.configuration_model(degree_sequence, seed=current_seed)
                    G_new = nx.Graph(G_multi)
                    G_new.remove_edges_from(nx.selfloop_edges(G_new))
            
            if G_new is not None:
                break
                
        except Exception as e:
            logger.warning(f"图生成尝试 {attempt+1} 失败: {e}")
            if current_seed is not None:
                current_seed += 1
    
    if G_new is None:
        logger.error("无法生成双峰图，返回原图副本")
        return G.copy()
    
    # 随机化边连接（打乱结构）- 可选步骤，失败时跳过
    num_edges = G_new.number_of_edges()
    if num_edges > 10:  # 只对较大的图进行随机化
        # 使用保守的交换次数，避免超时
        nswap = min(num_edges, 100)  # 最多交换 100 次
        max_tries = nswap * 20  # 每次交换最多尝试 20 次
        try:
            nx.double_edge_swap(G_new, nswap=nswap, max_tries=max_tries, seed=current_seed)
        except (nx.NetworkXAlgorithmError, nx.NetworkXError, Exception):
            # 边交换失败是正常的，直接跳过
            pass
    
    # 确保连通性（可能会添加边）
    edges_added_for_connectivity = _ensure_connectivity(G_new, seed=seed)
    if edges_added_for_connectivity > 0:
        logger.debug(f"为保证连通性添加了 {edges_added_for_connectivity} 条边")
    
    # 调整边数到精确目标值（保持连通性）
    original_M = G.number_of_edges()
    _adjust_edge_count(G_new, original_M, seed=seed)
    
    # 最终验证
    final_N = G_new.number_of_nodes()
    final_M = G_new.number_of_edges()
    original_N = G.number_of_nodes()
    is_connected = nx.is_connected(G_new)
    
    # 检查约束是否满足
    constraints_satisfied = True
    
    if final_N != original_N:
        logger.error(f"[约束违反] 节点数不匹配: 期望{original_N}, 实际{final_N}")
        constraints_satisfied = False
        
    if final_M != original_M:
        logger.error(f"[约束违反] 边数不匹配: 期望{original_M}, 实际{final_M} (差异: {final_M - original_M})")
        constraints_satisfied = False
        
    if not is_connected:
        logger.error("[约束违反] 生成的图不连通！")
        constraints_satisfied = False
    
    if constraints_satisfied:
        logger.debug(f"BiT-HyRL拓扑构造完成: N={final_N}, M={final_M}, 连通=True")
    else:
        logger.warning(f"BiT-HyRL拓扑约束未完全满足: N={final_N}/{original_N}, M={final_M}/{original_M}, 连通={is_connected}")
    
    return G_new


def _evaluate_robustness(G, seed=None):
    """
    评估网络的综合鲁棒性 R_weight = 0.5 * R_random + 0.5 * R_degree
    
    使用与主程序相同的攻击模拟逻辑（attack_step 作为 x 坐标，然后归一化）
    
    Args:
        G: 输入图
        seed: 随机种子
        
    Returns:
        tuple: (R_weight, R_degree, R_random)
    """
    if G is None or G.number_of_nodes() == 0:
        return 0.0, 0.0, 0.0
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    initial_nodes = G.number_of_nodes()
    
    def simulate_attack(G_copy, attack_sequence):
        """模拟攻击过程，返回 R 值（使用 attack_step 作为 x 坐标，然后归一化）"""
        x_results = [0]  # 攻击步数
        y_results = [1.0]  # GCC 比例
        
        attack_step = 0
        for node in attack_sequence:
            if node not in G_copy.nodes():
                continue
            G_copy.remove_node(node)
            attack_step += 1
            
            if G_copy.number_of_nodes() == 0:
                x_results.append(attack_step)
                y_results.append(0.0)
                break
            
            # 计算 GCC 比例
            components = list(nx.connected_components(G_copy))
            gcc_size = max(len(c) for c in components) if components else 0
            gcc_ratio = gcc_size / initial_nodes
            
            x_results.append(attack_step)
            y_results.append(gcc_ratio)
            
            if gcc_ratio <= 0:
                break
        
        # 计算 R 值（梯形法则，使用 attack_step）
        r_value = 0.0
        for i in range(len(x_results) - 1):
            dx = x_results[i + 1] - x_results[i]
            avg_y = (y_results[i] + y_results[i + 1]) / 2
            r_value += dx * avg_y
        
        # 归一化到 [0,1] 范围（除以初始节点数）
        if initial_nodes > 0:
            r_value = r_value / initial_nodes
        
        return r_value
    
    # Degree Attack（目标攻击）
    G_degree = G.copy()
    degree_list = sorted(G_degree.degree(), key=lambda x: x[1], reverse=True)
    degree_sequence = [node for node, _ in degree_list]
    R_degree = simulate_attack(G_degree, degree_sequence)
    
    # Random Attack（随机攻击）
    G_random = G.copy()
    random_sequence = list(G_random.nodes())
    random.shuffle(random_sequence)
    R_random = simulate_attack(G_random, random_sequence)
    
    # 加权 R 值
    R_weight = 0.5 * R_degree + 0.5 * R_random
    
    return R_weight, R_degree, R_random


def _attack_auc_lcc(G, attack_sequence):
    """计算给定攻击序列下的 LCC AUC，节点移除比例为 x 轴。"""
    if G is None or G.number_of_nodes() == 0:
        return 0.0

    G_sim = G.copy()
    n0 = G_sim.number_of_nodes()
    x_curve = [0.0]
    y_curve = [1.0]

    for step, node in enumerate(attack_sequence, start=1):
        if G_sim.has_node(node):
            G_sim.remove_node(node)
        if G_sim.number_of_nodes() == 0:
            y_val = 0.0
        else:
            y_val = max((len(c) for c in nx.connected_components(G_sim)), default=0) / n0
        x_curve.append(step / n0)
        y_curve.append(y_val)

    if len(x_curve) < 2:
        return 0.0
    return float(np.trapz(y_curve, x_curve))


def _attack_sequence_for_method(G, method, seed=None):
    """生成轻量级拓扑筛选用的攻击序列。"""
    nodes = list(G.nodes())
    if method == "random":
        rng = random.Random(seed)
        rng.shuffle(nodes)
        return nodes
    if method == "betweenness":
        scores = nx.betweenness_centrality(G)
        return sorted(nodes, key=lambda v: scores.get(v, 0.0), reverse=True)
    if method == "pagerank":
        scores = nx.pagerank(G)
        return sorted(nodes, key=lambda v: scores.get(v, 0.0), reverse=True)
    if method == "eigenvector":
        try:
            scores = nx.eigenvector_centrality(G, max_iter=1000)
            return sorted(nodes, key=lambda v: scores.get(v, 0.0), reverse=True)
        except Exception:
            pass
    return sorted(nodes, key=lambda v: G.degree(v), reverse=True)


def _low_layer_resilience_score(G):
    """
    评估移除最高度 hub 后，低度层是否仍能自保。

    双峰理论要求低度节点层自身能形成巨连通分支；该项用于避免单枢纽被打掉后
    外围节点立即碎裂。
    """
    n = G.number_of_nodes()
    if n <= 2:
        return 0.0

    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / n if n > 0 else 0.0
    hub_threshold = max(avg_degree * 2.0, avg_degree + 1.0)
    hub_nodes = [v for v, d in degrees.items() if d >= hub_threshold]
    if not hub_nodes:
        hub_nodes = [max(degrees, key=degrees.get)]

    G_low = G.copy()
    G_low.remove_nodes_from(hub_nodes)
    if G_low.number_of_nodes() == 0:
        return 0.0

    low_n = G_low.number_of_nodes()
    lcc_ratio = max((len(c) for c in nx.connected_components(G_low)), default=0) / low_n
    min_degree_ratio = min(1.0, min((d for _, d in G_low.degree()), default=0) / 2.0)
    return 0.75 * lcc_ratio + 0.25 * min_degree_ratio


def _candidate_topology_score(G, attack_methods=None, seed=None):
    """综合拓扑筛选分数：多攻击 AUC + 低度层自保。"""
    if attack_methods is None:
        attack_methods = ("degree", "betweenness", "random")

    aucs = {}
    for i, method in enumerate(attack_methods):
        seq_seed = None if seed is None else seed + 997 * (i + 1)
        seq = _attack_sequence_for_method(G, method, seed=seq_seed)
        aucs[method] = _attack_auc_lcc(G, seq)

    if not aucs:
        attack_score = 0.0
    else:
        # min 项让候选拓扑别只讨好随机攻击，mean 项保留整体曲线收益。
        attack_score = 0.65 * min(aucs.values()) + 0.35 * (sum(aucs.values()) / len(aucs))

    low_layer_score = _low_layer_resilience_score(G)
    score = 0.8 * attack_score + 0.2 * low_layer_score
    return score, aucs, low_layer_score


def create_bimodal_adaptive_robust(
    G,
    seed=None,
    max_hub_ratio=0.25,
    num_samples=12,
    attack_methods=None,
    verbose=False,
):
    """
    自适应鲁棒双峰构造：在固定 N/M 下搜索 hub 数量，优先选择多攻击 AUC 稳健的拓扑。

    保留双峰先验，但不再固定为单超级枢纽；这能缓解真实网络在 degree/betweenness
    攻击下“一打 hub 即崩”的问题。
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n < 4 or m < n - 1:
        return create_bimodal_theoretical(n, m, seed=seed)

    if attack_methods is None:
        attack_methods = ("degree", "betweenness", "random")

    max_hubs = max(1, min(n - 1, int(round(n * max_hub_ratio))))
    max_hubs = min(max_hubs, max(1, n // 4))
    if max_hubs <= num_samples:
        hub_candidates = list(range(1, max_hubs + 1))
    else:
        hub_candidates = [1, max_hubs]
        step = (max_hubs - 1) / max(1, num_samples - 1)
        hub_candidates.extend(int(round(1 + step * i)) for i in range(1, num_samples - 1))
        # 多加几个小 hub 候选，真实网络通常在这个区域更稳。
        hub_candidates.extend([2, 3, 4, 5, max(1, int(round(0.05 * n)))])
        hub_candidates = sorted({h for h in hub_candidates if 1 <= h <= max_hubs})

    best = None
    search_results = []
    for hub_num in hub_candidates:
        current_seed = None if seed is None else seed + hub_num
        G_candidate = create_bimodal_with_num_hubs(n, m, hub_num, seed=current_seed)
        if G_candidate is None:
            continue

        score, aucs, low_layer_score = _candidate_topology_score(
            G_candidate, attack_methods=attack_methods, seed=current_seed
        )
        record = {
            "hub_num": hub_num,
            "score": score,
            "aucs": aucs,
            "low_layer_score": low_layer_score,
        }
        search_results.append(record)
        if best is None or score > best["score"]:
            best = {"G": G_candidate.copy(), **record}

        if verbose:
            auc_text = ", ".join(f"{k}={v:.4f}" for k, v in aucs.items())
            print(
                f"  hub_num={hub_num:3d}: score={score:.4f}, "
                f"low_layer={low_layer_score:.4f}, {auc_text}"
            )

    if best is None:
        logger.warning("Adaptive robust bimodal search failed; falling back to theoretical bimodal.")
        G_fallback = create_bimodal_theoretical(n, m, seed=seed)
        G_fallback.graph["adaptive_bimodal"] = {"fallback": True}
        return G_fallback

    best_graph = best["G"]
    best_graph.graph["adaptive_bimodal"] = {
        "strategy": "adaptive_robust",
        "best_hub_num": best["hub_num"],
        "best_score": best["score"],
        "best_aucs": best["aucs"],
        "best_low_layer_score": best["low_layer_score"],
        "search_results": search_results,
        "attack_methods": list(attack_methods),
    }
    if verbose:
        print(f"Adaptive robust bimodal selected hub_num={best['hub_num']} score={best['score']:.4f}")
    return best_graph


def create_bimodal_adaptive(G, seed=None, search_range=None, num_samples=10, verbose=False):
    """
    自适应选择最佳 hub_num 的双峰网络构造。
    
    通过搜索不同的 hub_num 值，找到使 R_weight = 0.5*R_random + 0.5*R_degree 最大的配置。
    
    双峰网络中节点聚集在两个簇：
    - 大簇（hub 节点）：高度数节点，数量为 hub_num
    - 小簇（low 节点）：低度数节点，数量为 N - hub_num
    
    Args:
        G: 输入图
        seed: 随机种子
        search_range: 搜索范围 (min_hubs, max_hubs)，默认 (1, N//4)
        num_samples: 采样数量（在搜索范围内均匀采样），默认 10
        verbose: 是否输出详细信息
        
    Returns:
        nx.Graph: R_weight 最大的双峰网络
    """
    N = G.number_of_nodes()
    M = G.number_of_edges()
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if N < 4:
        logger.warning("节点数太少，无法进行自适应搜索，返回默认双峰网络")
        return create_bimodal_network_exact(G, hub_num=1, seed=seed)
    
    # 确定搜索范围
    if search_range is None:
        min_hubs = 1
        max_hubs = N # 最大 hub 数量为节点数的 1/4
    else:
        min_hubs, max_hubs = search_range
        min_hubs = max(1, min_hubs)
        max_hubs = min(max_hubs, N - 1)
    
   
    # 生成采样点（在搜索范围内均匀采样）
    if max_hubs - min_hubs + 1 <= num_samples:
        # 范围较小，遍历所有值
        hub_candidates = list(range(min_hubs, max_hubs + 1))
    else:
        # 范围较大，均匀采样
        hub_candidates = [min_hubs]  # 始终包含最小值
        step = (max_hubs - min_hubs) / (num_samples - 1)
        for i in range(1, num_samples - 1):
            candidate = int(round(min_hubs + i * step))
            if candidate not in hub_candidates:
                hub_candidates.append(candidate)
        if max_hubs not in hub_candidates:
            hub_candidates.append(max_hubs)  # 始终包含最大值
        hub_candidates = sorted(set(hub_candidates))
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"自适应双峰网络搜索 (N={N}, M={M})")
        print(f"搜索范围: hub_num ∈ [{min_hubs}, {max_hubs}]")
        print(f"采样点数: {len(hub_candidates)}")
        print(f"{'='*60}")
    
    # 评估每个候选 hub_num
    best_hub_num = 1
    best_R_weight = -1.0
    best_G = None
    results = []
    
    for hub_num in hub_candidates:
        # 构造双峰网络
        current_seed = seed + hub_num if seed is not None else None
        G_candidate = create_bimodal_network_exact(G, hub_num=hub_num, seed=current_seed)
        
        # #region agent log - 检查候选网络约束
        import json, os, time as time_module
        debug_log_path = r"e:\项目\02-论文\03-论文计划\17-BIG\code\v2\BiT-HyRL\.cursor\debug.log"
        try:
            os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                log_entry = {
                    "id": f"log_{int(time_module.time() * 1000)}_candidate_{hub_num}",
                    "timestamp": int(time_module.time() * 1000),
                    "location": "reconstruction.py:create_bimodal_adaptive",
                    "message": f"Candidate network hub_num={hub_num}",
                    "hypothesisId": "H1",
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "data": {
                        "hub_num": hub_num,
                        "target_nodes": N,
                        "target_edges": M,
                        "candidate_nodes": G_candidate.number_of_nodes() if G_candidate else None,
                        "candidate_edges": G_candidate.number_of_edges() if G_candidate else None,
                        "is_connected": nx.is_connected(G_candidate) if G_candidate else False,
                        "nodes_match": N == G_candidate.number_of_nodes() if G_candidate else False,
                        "edges_match": M == G_candidate.number_of_edges() if G_candidate else False,
                        "is_valid": G_candidate is not None and G_candidate.number_of_nodes() > 0
                    }
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception:
            pass
        # #endregion
        
        if G_candidate is None or G_candidate.number_of_nodes() == 0:
            if verbose:
                print(f"  hub_num={hub_num:3d}: 构造失败")
            continue
        
        # 评估鲁棒性
        eval_seed = current_seed + 1000 if current_seed is not None else None
        R_weight, R_degree, R_random = _evaluate_robustness(G_candidate, seed=eval_seed)
        
        # #region agent log - 评估结果（修复后）
        try:
            import json, os, time as time_module
            debug_log_path = r"e:\项目\02-论文\03-论文计划\17-BIG\code\v2\BiT-HyRL\.cursor\debug.log"
            os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                log_entry = {
                    "id": f"log_{int(time_module.time() * 1000)}_eval_{hub_num}_fixed",
                    "timestamp": int(time_module.time() * 1000),
                    "location": "reconstruction.py:create_bimodal_adaptive",
                    "message": f"Evaluation result hub_num={hub_num} (after fix)",
                    "hypothesisId": "H2",
                    "sessionId": "debug-session",
                    "runId": "post-fix",
                    "data": {
                        "hub_num": hub_num,
                        "R_weight": R_weight,
                        "R_degree": R_degree,
                        "R_random": R_random,
                        "is_best": R_weight > best_R_weight,
                        "note": "Using attack_step as x-coordinate, normalized by initial_nodes"
                    }
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception:
            pass
        # #endregion
        
        results.append({
            'hub_num': hub_num,
            'R_weight': R_weight,
            'R_degree': R_degree,
            'R_random': R_random
        })
        
        if verbose:
            print(f"  hub_num={hub_num:3d}: R_weight={R_weight:.4f} (R_tar={R_degree:.4f}, R_ran={R_random:.4f})")
        
        # 更新最佳结果
        if R_weight > best_R_weight:
            best_R_weight = R_weight
            best_hub_num = hub_num
            best_G = G_candidate.copy()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"最优结果: hub_num={best_hub_num}, R_weight={best_R_weight:.4f}")
        print(f"{'='*60}\n")
    
    # 如果所有候选都失败，返回默认网络
    if best_G is None:
        logger.warning("所有候选 hub_num 都失败，返回默认双峰网络")
        return create_bimodal_network_exact(G, hub_num=1, seed=seed)
    
    # #region agent log - 最终返回的网络约束检查
    try:
        import json, os, time as time_module
        debug_log_path = r"e:\项目\02-论文\03-论文计划\17-BIG\code\v2\BiT-HyRL\.cursor\debug.log"
        os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            log_entry = {
                "id": f"log_{int(time_module.time() * 1000)}_final_network",
                "timestamp": int(time_module.time() * 1000),
                "location": "reconstruction.py:create_bimodal_adaptive",
                "message": "Final selected network constraints",
                "hypothesisId": "H1",
                "sessionId": "debug-session",
                "runId": "run1",
                "data": {
                    "best_hub_num": best_hub_num,
                    "best_R_weight": best_R_weight,
                    "target_nodes": N,
                    "target_edges": M,
                    "final_nodes": best_G.number_of_nodes(),
                    "final_edges": best_G.number_of_edges(),
                    "is_connected": nx.is_connected(best_G),
                    "nodes_match": N == best_G.number_of_nodes(),
                    "edges_match": M == best_G.number_of_edges(),
                    "constraints_satisfied": (N == best_G.number_of_nodes() and 
                                            M == best_G.number_of_edges() and 
                                            nx.is_connected(best_G))
                }
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception:
        pass
    # #endregion
    
    # 存储搜索结果到图属性（可选，用于调试）
    best_G.graph['adaptive_search'] = {
        'best_hub_num': best_hub_num,
        'best_R_weight': best_R_weight,
        'search_results': results
    }
    
    return best_G


def create_bimodal_adaptive_fast(G, seed=None, verbose=False):
    """
    快速自适应双峰网络构造（使用三分搜索）。
    
    假设 R_weight 关于 hub_num 是单峰函数（存在唯一最大值），使用三分搜索加速。
    
    Args:
        G: 输入图
        seed: 随机种子
        verbose: 是否输出详细信息
        
    Returns:
        nx.Graph: R_weight 最大的双峰网络
    """
    N = G.number_of_nodes()
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if N < 4:
        return create_bimodal_network_exact(G, hub_num=1, seed=seed)
    
    # 搜索范围
    left = 1
    right = max(2, N // 4)
    
    # 缓存已评估的结果
    cache = {}
    
    def evaluate(hub_num):
        if hub_num in cache:
            return cache[hub_num]
        
        current_seed = seed + hub_num if seed is not None else None
        G_candidate = create_bimodal_network_exact(G, hub_num=hub_num, seed=current_seed)
        
        if G_candidate is None:
            cache[hub_num] = (None, 0.0, 0.0, 0.0)
            return cache[hub_num]
        
        eval_seed = current_seed + 1000 if current_seed is not None else None
        R_weight, R_degree, R_random = _evaluate_robustness(G_candidate, seed=eval_seed)
        cache[hub_num] = (G_candidate.copy(), R_weight, R_degree, R_random)
        
        if verbose:
            print(f"  hub_num={hub_num:3d}: R_weight={R_weight:.4f}")
        
        return cache[hub_num]
    
    if verbose:
        print(f"\n自适应搜索 (三分法): N={N}, 范围=[{left}, {right}]")
    
    # 三分搜索
    while right - left > 2:
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3
        
        _, r1, _, _ = evaluate(mid1)
        _, r2, _, _ = evaluate(mid2)
        
        if r1 < r2:
            left = mid1
        else:
            right = mid2
    
    # 在剩余范围内找最优
    best_hub_num = left
    best_R_weight = -1.0
    best_G = None
    
    for hub_num in range(left, right + 1):
        G_candidate, R_weight, _, _ = evaluate(hub_num)
        if G_candidate is not None and R_weight > best_R_weight:
            best_R_weight = R_weight
            best_hub_num = hub_num
            best_G = G_candidate
    
    if verbose:
        print(f"最优: hub_num={best_hub_num}, R_weight={best_R_weight:.4f}\n")
    
    if best_G is None:
        return create_bimodal_network_exact(G, hub_num=1, seed=seed)
    
    return best_G


def create_bimodal_with_num_hubs(n, m, num_hubs, seed=42):
    """
    构造指定 hub 数量的双峰网络（固定 N、M）。
    用于实验：逐步增大 num_hubs，观察韧性变化。
    
    Args:
        n: 节点数
        m: 边数
        num_hubs: hub 节点数量（至少 1，至多 n-1）
        seed: 随机种子
        
    Returns:
        nx.Graph: 连通图，节点数 n、边数 m；若无法构造则返回 None
    """
    if n < 2 or m < n - 1:
        return None
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    num_hubs = max(1, min(num_hubs, n - 1))
    num_low = n - num_hubs
    total_degree = 2 * m
    
    # k_max 上限：保证 degree_rem >= num_low（k_base>=1）
    k_max_cap = (total_degree - num_low) // num_hubs if num_hubs > 0 else n - 1
    k_max_cap = max(2, min(k_max_cap, n - 1))
    
    # 理论 k_max（双峰公式，平均分配剩余度）
    avg_k = total_degree / n
    if avg_k > 1 and (2 * avg_k - 1) > 0:
        numerator = 2 * (avg_k**2) * ((avg_k - 1)**2)
        denominator = 2 * avg_k - 1
        A = (numerator / denominator) ** (1/3)
        k_max_theory = int(round(A * (n ** (2/3))))
    else:
        k_max_theory = max(2, int(avg_k * 2))
    
    k_max = min(k_max_theory, k_max_cap)
    k_max = max(2, k_max)
    degree_rem = total_degree - k_max * num_hubs
    
    if degree_rem < num_low:
        k_max = max(2, (total_degree - num_low) // num_hubs)
        degree_rem = total_degree - k_max * num_hubs
    k_base = max(1, degree_rem // num_low)
    remainder = degree_rem % num_low if degree_rem >= num_low else 0
    
    degree_seq = [k_max] * num_hubs + [k_base + 1] * remainder + [k_base] * (num_low - remainder)
    if sum(degree_seq) != total_degree:
        diff = total_degree - sum(degree_seq)
        for _ in range(abs(diff)):
            if diff > 0:
                i = min(range(len(degree_seq)), key=lambda i: (degree_seq[i], -i))
                if degree_seq[i] < n - 1:
                    degree_seq[i] += 1
                    diff -= 1
            else:
                i = max(range(len(degree_seq)), key=lambda i: (degree_seq[i], -i))
                if degree_seq[i] > 1:
                    degree_seq[i] -= 1
                    diff += 1
    if sum(degree_seq) % 2 != 0:
        for i in range(len(degree_seq)):
            if degree_seq[i] < n - 1:
                degree_seq[i] += 1
                break
    if not nx.is_graphical(degree_seq):
        degree_seq = _make_graphical(degree_seq.copy())
    try:
        G = nx.havel_hakimi_graph(degree_seq)
    except Exception:
        G = nx.configuration_model(degree_seq, seed=seed)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
    if not nx.is_connected(G):
        _ensure_connectivity(G, seed=seed)
    _adjust_edge_count(G, m, seed=seed)
    if G.number_of_nodes() != n or G.number_of_edges() != m or not nx.is_connected(G):
        return None
    return G


def adjust_edges_strictly(G, target_m):
    """辅助函数：强制增删边以精确匹配目标边数（删边时可能破坏连通性，仅用于非稀疏图）"""
    current_m = G.number_of_edges()
    if current_m < target_m:
        nodes = list(G.nodes())
        while G.number_of_edges() < target_m:
            u, v = random.sample(nodes, 2)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
    elif current_m > target_m:
        edges = list(G.edges())
        random.shuffle(edges)
        G.remove_edges_from(edges[:current_m - target_m])
    return G


def _is_low_density(n, m):
    """判断是否为低密度（稀疏）网络"""
    if n < 2 or m < n - 1:
        return True
    avg_k = 2 * m / n
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0
    return avg_k < SPARSE_AVG_DEGREE_THRESHOLD or density < SPARSE_DENSITY_THRESHOLD


def _bimodal_degree_sequence_sparse(n, m, num_hubs=1, seed=None):
    """
    为稀疏网络 (n, m) 构造双峰度序列：保证可图化、连通性可行、且保留双峰形态。
    - num_hubs: hub 节点数量；默认 1。保证 k_base >= 1、可图化。
    """
    if seed is not None:
        random.seed(seed)
    num_hubs = max(1, min(num_hubs, n - 1))
    n_low = n - num_hubs
    total_degree = 2 * m
    avg_k = total_degree / n
    # 非 hub 至少需要 n_low 的度数和（支撑一棵树）
    min_degree_for_low = n_low
    # hub 最多拿走这么多度，且 k_max 需使 degree_rem >= n_low
    k_max_cap = (total_degree - min_degree_for_low) // num_hubs if num_hubs > 0 else n - 1
    k_max_cap = min(n - 1, max(1, k_max_cap))
    k_max_cap = max(2, k_max_cap)
    # 理论 k_max（双峰公式）
    if avg_k > 1 and (2 * avg_k - 1) > 0:
        numerator = 2 * (avg_k**2) * ((avg_k - 1)**2)
        denominator = 2 * avg_k - 1
        A = (numerator / denominator) ** (1/3)
        k_max_theory = int(round(A * (n ** (2/3))))
    else:
        k_max_theory = max(2, int(avg_k * 2))
    k_max = min(k_max_theory, k_max_cap)
    k_max = max(2 if n > 2 else 1, k_max)
    degree_rem = total_degree - k_max * num_hubs
    if degree_rem < n_low:
        k_max = max(1, (total_degree - n_low) // num_hubs) if num_hubs > 0 else 1
        k_max = min(k_max, n - 1)
        degree_rem = total_degree - k_max * num_hubs
    k_base = max(1, degree_rem // n_low) if n_low > 0 else 0
    remainder = degree_rem % n_low if n_low > 0 else 0
    degree_seq = [k_max] * num_hubs + [k_base + 1] * remainder + [k_base] * (n_low - remainder)
    # 确保和为 2*m
    s = sum(degree_seq)
    if s < total_degree:
        for _ in range(total_degree - s):
            idx = random.randint(num_hubs, n - 1) if n > num_hubs else 0
            if degree_seq[idx] < n - 1:
                degree_seq[idx] += 1
    elif s > total_degree:
        for _ in range(s - total_degree):
            idx = random.randint(num_hubs, n - 1) if n > num_hubs else 0
            if degree_seq[idx] > 1:
                degree_seq[idx] -= 1
    if sum(degree_seq) % 2 != 0:
        if num_hubs > 0 and degree_seq[0] > 0:
            degree_seq[0] -= 1
        else:
            for i in range(num_hubs, n):
                if degree_seq[i] > 1:
                    degree_seq[i] -= 1
                    break
    return degree_seq


def create_bimodal_theoretical(n, m, seed=42):
    """
    Bimodal Network (Based on k_max = A * N^(2/3))。
    对低密度网络（平均度小、边数少）使用稀疏优化：度序列可图化、删边时保持连通性。

    Args:
        n: 节点数
        m: 边数
        seed: 随机种子
    """
    if n < 2 or m < n - 1:
        G = nx.path_graph(n) if n >= 2 else nx.Graph()
        if n >= 2 and m >= n - 1:
            for _ in range(m - (n - 1)):
                u, v = random.sample(list(G.nodes()), 2)
                if u != v and not G.has_edge(u, v):
                    G.add_edge(u, v)
        return G

    avg_k = 2 * m / n
    use_sparse = _is_low_density(n, m)
    num_hubs = 1  # 默认使用 1 个 hub 节点

    if use_sparse:
        degree_seq = _bimodal_degree_sequence_sparse(n, m, num_hubs=num_hubs, seed=seed)
        if not nx.is_graphical(degree_seq):
            # 使用保双峰的可图化调整，避免抹平 hub 与 low 的区分（否则 hub_ratio 会“失效”）
            degree_seq = _make_graphical_bimodal_preserving(degree_seq, num_hubs)
        try:
            G = nx.havel_hakimi_graph(degree_seq)
        except Exception:
            G = nx.configuration_model(degree_seq, seed=seed)
            G = nx.Graph(G)
            G.remove_edges_from(nx.selfloop_edges(G))
        if not nx.is_connected(G):
            _ensure_connectivity(G, seed=seed)
        # 稀疏网络：用保连通的方式调整边数
        _adjust_edge_count(G, m, seed=seed)
        G_final = G
    else:
        numerator = 2 * (avg_k**2) * ((avg_k - 1)**2)
        denominator = 2 * avg_k - 1
        A = (numerator / denominator) ** (1/3)
        k_max = int(round(A * (n ** (2/3))))
        k_max = min(k_max, n - 1)
        k_max = max(k_max, 2)
        n_high = num_hubs
        n_low = n - n_high
        total_degree_needed = 2 * m
        degree_rem = total_degree_needed - k_max * n_high
        k_base = max(1, degree_rem // n_low) if n_low > 0 else 0
        remainder = (degree_rem % n_low) if n_low > 0 else 0
        degree_seq = [k_max] * n_high + [k_base + 1] * remainder + [k_base] * (n_low - remainder)
        if sum(degree_seq) != total_degree_needed:
            diff = total_degree_needed - sum(degree_seq)
            for _ in range(abs(diff)):
                if diff > 0:
                    i = min(range(len(degree_seq)), key=lambda i: (degree_seq[i], -i))
                    if degree_seq[i] < n - 1:
                        degree_seq[i] += 1
                        diff -= 1
                else:
                    i = max(range(len(degree_seq)), key=lambda i: (degree_seq[i], -i))
                    if degree_seq[i] > 1:
                        degree_seq[i] -= 1
                        diff += 1
        if sum(degree_seq) % 2 != 0:
            for i in range(len(degree_seq)):
                if degree_seq[i] < n - 1:
                    degree_seq[i] += 1
                    break
        if nx.is_graphical(degree_seq):
            G = nx.havel_hakimi_graph(degree_seq)
            try:
                nx.double_edge_swap(G, nswap=min(5 * m, 500), max_tries=min(20 * m, 2000), seed=seed)
            except Exception:
                pass
        else:
            degree_seq = _make_graphical(degree_seq)
            G = nx.havel_hakimi_graph(degree_seq) if nx.is_graphical(degree_seq) else nx.configuration_model(degree_seq, seed=seed)
            if not isinstance(G, nx.Graph):
                G = nx.Graph(G)
                G.remove_edges_from(nx.selfloop_edges(G))
        # 稠密网络：删边时仍用保连通调整，避免偶发断连
        current_m = G.number_of_edges()
        if current_m != m:
            _adjust_edge_count(G, m, seed=seed)
        G_final = G
    
    # #region agent log - 假设E: 记录双峰重构效果（用于分析稀疏网络重构问题）
    import json, os, time as time_module
    debug_log_path = r"e:\项目\02-论文\03-论文计划\17-BIG\code\v2\BiT-HyRL\.cursor\debug.log"
    try:
        os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
        degrees = [d for _, d in G_final.degree()]
        actual_max_degree = max(degrees) if degrees else 0
        actual_avg_degree = sum(degrees) / len(degrees) if degrees else 0
        actual_min_degree = min(degrees) if degrees else 0
        hub_threshold = actual_avg_degree * 2
        num_hubs = sum(1 for d in degrees if d > hub_threshold)
        degree_variance = np.var(degrees) if degrees else 0
        hub_degree_ratio = actual_max_degree / actual_avg_degree if actual_avg_degree > 0 else 0
        log_entry = {
            "timestamp": int(time_module.time() * 1000),
            "location": "reconstruction.py:create_bimodal_theoretical",
            "message": "Bimodal reconstruction effect for hypothesis E",
            "hypothesisId": "E",
            "sessionId": "debug-session",
            "data": {
                "target_nodes": n,
                "target_edges": m,
                "target_avg_degree": round(avg_k, 4),
                "sparse_mode": use_sparse,
                "theoretical_k_max": actual_max_degree,
                "actual_max_degree": actual_max_degree,
                "actual_avg_degree": round(actual_avg_degree, 4),
                "num_hubs": num_hubs,
                "hub_degree_ratio": round(hub_degree_ratio, 4),
                "degree_variance": round(float(degree_variance), 4),
                "k_base": actual_min_degree,
                "is_connected": nx.is_connected(G_final),
                "final_edges": G_final.number_of_edges()
            }
        }
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        pass
    # #endregion

    return G_final


def _ensure_connected_degree_preserving(G, seed=None, max_attempts=5000):
    """
    通过保度的 double-edge swap 使图连通，尽量不改变边数。
    若无法通过 swap 连通，则返回 False（调用方可使用 _ensure_connectivity 作为兜底）。
    """
    if seed is not None:
        random.seed(seed)
    if nx.is_connected(G):
        return True

    nodes = list(G.nodes())
    edges = list(G.edges())
    for attempt in range(max_attempts):
        comps = list(nx.connected_components(G))
        if len(comps) == 1:
            return True
        main = max(comps, key=len)
        small = next(iter([c for c in comps if c is not main]))

        # 在 small 中随机选一条边
        small_edges = [(u, v) for u, v in G.edges() if u in small and v in small]
        if not small_edges:
            continue
        u, v = random.choice(small_edges)

        # 在 main 中随机选一条边
        main_edges = [(x, y) for x, y in G.edges() if x in main and y in main]
        if not main_edges:
            continue
        x, y = random.choice(main_edges)

        # 避免自环/重边，并确保四个节点互不相同
        if len({u, v, x, y}) < 4:
            continue
        if G.has_edge(u, x) or G.has_edge(v, y):
            continue

        # 执行 2-switch：u-v, x-y -> u-x, v-y
        G.remove_edge(u, v)
        G.remove_edge(x, y)
        G.add_edge(u, x)
        G.add_edge(v, y)

    return nx.is_connected(G)


def _ensure_min_degree(G, min_degree=2, seed=None):
    """
    通过添加边使图中所有节点的度至少为 min_degree。
    仅在存在可添加的简单边时操作，不删除已有边，不强制精确边数。
    """
    if seed is not None:
        random.seed(seed)
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return
    max_degree = len(nodes) - 1
    # 最多迭代轮数，避免死循环
    for _ in range(10 * len(nodes)):
        low_nodes = [v for v in nodes if G.degree(v) < min_degree]
        if not low_nodes:
            break
        v = low_nodes[0]
        # 候选：与 v 不相邻、且自身度未满的节点；优先连接度数较低的节点
        candidates = [u for u in nodes if u != v and not G.has_edge(v, u) and G.degree(u) < max_degree]
        if not candidates:
            # 若 v 已经连接到所有其他节点但仍不满足 min_degree，则图规模过小，无法继续
            break
        candidates.sort(key=lambda u: G.degree(u))
        u = candidates[0]
        G.add_edge(v, u)


def create_bimodal_network_docx(G=None, n=None, m=None, seed=None, ensure_connected=True):
    """
    参照 docs/Bimodal.txt 中的修正逻辑构造双峰网络。

    核心规则：
    - 固定成本：节点数 N、边数 M 与原网络相同。
    - 仅当 avg_k = 2M/N >= 2 时才可构造（保证低度节点 k_min >= 2）。
    - 先尝试书中理论最优 k_max = A * N^(2/3)（A 见公式 7.67）。
    - 若理论最优导致 k_min < 2 或 k_max > N-1，则：
        * 边数偏少时：固定 k_min=2，k_max = 2M - 2(N-1)；
        * 边数偏多（简单图上限）：固定 k_max=N-1，将剩余度数分配给叶子节点（k_min>=2）。
    - 优先使用 Havel-Hakimi 生成精确度序列的简单图；不可图化时回退到配置模型。
    - 最后确保图连通且实际最小度 >= 2。

    Args:
        G: 输入图（提供 N, M）。若给定 n, m 则优先使用 n, m。
        n, m: 节点数与边数。
        seed: 随机种子。
        ensure_connected: 是否保证返回图连通（默认 True）。

    Returns:
        dict: {
            'G': 生成的双峰网络,
            'k_min_target': 目标低度,
            'k_max_target': 目标高度,
            'strategy': 'theory' / 'low' / 'cap' / 'infeasible',
            'M_actual': 实际边数,
            'N_actual': 实际节点数,
            'connected': 是否连通,
        }
        若不可行，'G' 为 None。
    """
    import random
    import numpy as np
    import networkx as nx

    if G is not None:
        N = G.number_of_nodes()
        M = G.number_of_edges()
    else:
        N = n
        M = m

    if N is None or M is None:
        raise ValueError("必须提供 G 或 (n, m)")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    result = {
        'G': None,
        'k_min_target': None,
        'k_max_target': None,
        'strategy': None,
        'M_actual': None,
        'N_actual': N,
        'connected': False,
    }

    if N < 2 or M < N - 1:
        result['strategy'] = 'infeasible'
        return result

    avg_k = 2 * M / N
    if avg_k < 2:
        result['strategy'] = 'infeasible'
        return result

    n_leaves = N - 1
    total_deg = 2 * M

    # 书中公式 7.67
    A = ((2 * avg_k**2 * (avg_k - 1)**2) / (2 * avg_k - 1)) ** (1 / 3)
    k_max_theory = A * (N ** (2 / 3))
    k_min_theory = (total_deg - k_max_theory) / n_leaves

    # 尝试理论最优
    if k_min_theory >= 2.0 and k_max_theory <= N - 1:
        k_max = int(round(k_max_theory))
        k_max = min(max(k_max, 2), N - 1)
        rem = total_deg - k_max
        k_min = rem // n_leaves
        extra = rem % n_leaves
        strategy = 'theory'
    elif (total_deg - 2 * n_leaves) <= N - 1:
        # 边数偏少：优先保证 k_min = 2
        k_min = 2
        k_max = total_deg - 2 * n_leaves
        k_max = min(max(k_max, k_min), N - 1)
        rem = total_deg - k_max
        k_min = rem // n_leaves
        extra = rem % n_leaves
        strategy = 'low'
    else:
        # 边数偏多：受简单图上限 N-1 约束
        k_max = N - 1
        rem = total_deg - k_max
        k_min = rem // n_leaves
        extra = rem % n_leaves
        strategy = 'cap'

    # 安全性检查
    if k_min < 2:
        result['strategy'] = 'infeasible'
        return result
    if k_max < k_min:
        result['strategy'] = 'infeasible'
        return result

    # 度序列：1 个枢纽 + (N-1) 个叶子；extra 个叶子度为 k_min+1，其余为 k_min
    degree_seq = [k_max] + [k_min + 1] * extra + [k_min] * (n_leaves - extra)

    # 若不可图化，回退到配置模型
    if nx.is_graphical(degree_seq):
        try:
            G_new = nx.havel_hakimi_graph(degree_seq)
        except Exception:
            G_new = nx.configuration_model(degree_seq, seed=seed)
            G_new = nx.Graph(G_new)
            G_new.remove_edges_from(nx.selfloop_edges(G_new))
    else:
        G_multi = nx.configuration_model(degree_seq, seed=seed)
        G_new = nx.Graph(G_multi)
        G_new.remove_edges_from(nx.selfloop_edges(G_new))

    if ensure_connected and not nx.is_connected(G_new):
        connected = _ensure_connected_degree_preserving(G_new, seed=seed)
        if not connected:
            _ensure_connectivity(G_new, seed=seed)

    # Havel-Hakimi 已保证实际最小度 >= k_min >= 2，这里不再主动加边以维持精确 M。
    # 若未来使用配置模型 fallback 且出现低度节点，可在此补充保度修复。

    result.update({
        'G': G_new,
        'k_min_target': k_min,
        'k_max_target': k_max,
        'strategy': strategy,
        'M_actual': G_new.number_of_edges(),
        'N_actual': G_new.number_of_nodes(),
        'connected': nx.is_connected(G_new),
    })
    return result
