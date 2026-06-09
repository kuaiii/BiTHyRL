# -*- coding: utf-8 -*-
"""BiT-HyRL 双峰拓扑重构。"""
import random
import numpy as np
import networkx as nx
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _ensure_connectivity(G, target_edges, seed=None):
    """
    确保图的连通性，同时尽量保持目标边数。
    
    Args:
        G: 输入图
        target_edges: 目标边数
        seed: 随机种子
        
    Returns:
        连通的图
    """
    if seed is not None:
        random.seed(seed)
    
    if nx.is_connected(G):
        return G
    
    # 获取所有连通分量
    components = list(nx.connected_components(G))
    if len(components) == 1:
        return G
    
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
    
    return G


def _adjust_edge_count(G, target_edges, seed=None):
    """
    调整图的边数到目标值，同时保持连通性。
    
    Args:
        G: 输入图
        target_edges: 目标边数
        seed: 随机种子
        
    Returns:
        调整后的图
    """
    if seed is not None:
        random.seed(seed)
    
    current_edges = G.number_of_edges()
    N = G.number_of_nodes()
    
    if current_edges < target_edges:
        # 需要添加边
        need_more = target_edges - current_edges
        all_possible = [(u, v) for u in range(N) for v in range(u+1, N) 
                        if not G.has_edge(u, v)]
        random.shuffle(all_possible)
        
        # 优先添加连接到hub节点的边，保持双峰特性
        degrees = dict(G.degree())
        avg_degree = sum(degrees.values()) / N if N > 0 else 0
        hub_threshold = avg_degree * 2
        
        # 按优先级排序：连接到高度节点的边优先
        def edge_priority(edge):
            u, v = edge
            return -(degrees.get(u, 0) + degrees.get(v, 0))
        
        all_possible.sort(key=edge_priority)
        
        edges_to_add = all_possible[:need_more]
        G.add_edges_from(edges_to_add)
        
    elif current_edges > target_edges:
        # 需要删除边，但要保持连通性
        need_less = current_edges - target_edges
        edges = list(G.edges())
        random.shuffle(edges)
        
        removed = 0
        for edge in edges:
            if removed >= need_less:
                break
            
            # 检查删除这条边后是否仍然连通
            u, v = edge
            G.remove_edge(u, v)
            
            if nx.is_connected(G):
                removed += 1
            else:
                # 恢复边
                G.add_edge(u, v)
        
        # 如果无法删除足够的边，记录警告
        if removed < need_less:
            logger.warning(f"无法删除足够的边保持连通性: 需要删除{need_less}条，实际删除{removed}条")
    
    return G


def create_bimodal_network_exact(G, hub_ratio=0.15, seed=None):
    """
    构造符合最优双峰度分布的无向连通网络（BiT-HyRL 拓扑层）。
    基于理论公式 k_max = A * N^(2/3)。
    
    保证约束：
    1. 节点数 = 原图节点数
    2. 边数 = 原图边数（尽可能接近）
    3. 图必须连通
    
    Args:
        G: 原始图
        hub_ratio: hub节点比例，默认0.15
        seed: 随机种子
        
    Returns:
        满足双峰分布且连通的新图
    """
    N = G.number_of_nodes()
    M = G.number_of_edges()
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 参数验证
    if N < 2:
        logger.warning("节点数太少，返回原图")
        return G.copy()
    
    # 计算最小边数（保证连通性需要至少 N-1 条边）
    min_edges_for_connectivity = N - 1
    if M < min_edges_for_connectivity:
        logger.warning(f"原图边数({M})小于连通性最小要求({min_edges_for_connectivity})，将增加边数")
        M = min_edges_for_connectivity
    
    total_degree = 2 * M
    avg_k = total_degree / N
    
    # 计算hub节点数量
    if hub_ratio <= 0.01:
        num_hubs = 1
    else:
        min_hubs = 2 if N >= 20 else 1
        num_hubs = max(min_hubs, min(int(N * hub_ratio), N // 4))
    
    # 计算最优 k_max（基于双峰分布理论）
    if avg_k > 1 and (2 * avg_k - 1) > 0:
        A = (2 * (avg_k**2) * ((avg_k - 1)**2) / (2 * avg_k - 1)) ** (1/3)
        k_max = int(round(A * (N ** (2/3))))
    else:
        k_max = max(2, int(avg_k * 3))
    
    # 限制 k_max 范围
    k_max = min(k_max, N - 1)
    k_max = max(k_max, 2)
    
    # 构造度序列
    num_low = N - num_hubs
    degree_rem = total_degree - k_max * num_hubs
    
    # 确保低度节点的度数至少为1（保证连通性）
    if num_low > 0:
        k_base = max(1, degree_rem // num_low)
        remainder = degree_rem % num_low if degree_rem >= num_low else 0
    else:
        k_base = 1
        remainder = 0
    
    # 构建度序列：hub节点 + 高度节点 + 基础度节点
    degree_sequence = [k_max] * num_hubs
    degree_sequence += [k_base + 1] * min(remainder, num_low)
    degree_sequence += [k_base] * max(0, num_low - remainder)
    
    # 确保所有度数至少为1
    degree_sequence = [max(1, d) for d in degree_sequence]
    
    # 调整度序列总和为偶数
    actual_degree = sum(degree_sequence)
    degree_diff = total_degree - actual_degree
    
    if degree_diff > 0:
        # 需要增加度数
        for i in range(abs(degree_diff)):
            idx = (num_hubs + i) % N
            degree_sequence[idx] += 1
    elif degree_diff < 0:
        # 需要减少度数
        for i in range(abs(degree_diff)):
            idx = (num_hubs + i) % N
            if degree_sequence[idx] > 1:
                degree_sequence[idx] -= 1
    
    # 确保度序列总和为偶数
    if sum(degree_sequence) % 2 != 0:
        # 找一个非hub节点增加度数
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
                # 使用 havel_hakimi 算法
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
    
    # 确保连通性
    G_new = _ensure_connectivity(G_new, M, seed=seed)
    
    # 调整边数到目标值
    target_M = G.number_of_edges()
    G_new = _adjust_edge_count(G_new, target_M, seed=seed)
    
    # 最终验证
    final_N = G_new.number_of_nodes()
    final_M = G_new.number_of_edges()
    original_N = G.number_of_nodes()
    original_M = G.number_of_edges()
    is_connected = nx.is_connected(G_new)
    
    if final_N != original_N:
        logger.error(f"节点数不匹配: 期望{original_N}, 实际{final_N}")
    if final_M != original_M:
        logger.warning(f"边数略有差异: 期望{original_M}, 实际{final_M} (差异: {final_M - original_M})")
    if not is_connected:
        logger.error("生成的图不连通！")
    
    logger.debug(f"BiT-HyRL拓扑构造完成: N={final_N}, M={final_M}, 连通={is_connected}")
    
    return G_new


def _make_graphical(degree_sequence):
    """
    将度序列调整为可图化的序列。
    
    使用 Erdős–Gallai 定理进行调整。
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
        
        # 尝试减少最大度数
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
