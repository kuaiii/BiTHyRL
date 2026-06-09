"""
IoT 网络拓扑生成器
基于各论文中描述的 scale-free IoT 拓扑生成方式

共同设置（来自论文分析）：
- N=100, M=2 是最常见的测试配置
- 区域大小：500×500m² 或 1000×1000m²
- 通信半径：200m
- 节点随机均匀部署
- 优先连接机制（类似 BA 模型，但受通信半径约束）
"""

import random
import numpy as np
import networkx as nx


def generate_iot_topology(N, M, area_size=500, comm_radius=200, seed=None):
    """
    生成 scale-free IoT 拓扑（通用版本）
    
    参数:
        N: 节点数量
        M: 边密度（每个节点的平均连接数）
        area_size: 区域边长（正方形区域）
        comm_radius: 通信半径
        seed: 随机种子
    
    返回:
        networkx.Graph，节点带有 'pos' 属性
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 随机部署节点位置
    positions = {}
    for i in range(N):
        positions[i] = (random.uniform(0, area_size), random.uniform(0, area_size))
    
    G = nx.Graph()
    G.add_nodes_from(range(N))
    nx.set_node_attributes(G, positions, 'pos')
    
    # 计算节点间距离
    def dist(i, j):
        x1, y1 = positions[i]
        x2, y2 = positions[j]
        return np.hypot(x1 - x2, y1 - y2)
    
    # 生成候选边（通信半径内）
    candidate_edges = {}
    for i in range(N):
        candidate_edges[i] = [j for j in range(N) if i != j and dist(i, j) <= comm_radius]
    
    # 使用优先连接机制构建网络
    # 初始连通：连接前 min(M+1, N) 个节点形成初始团
    init_nodes = min(M + 1, N)
    for i in range(init_nodes):
        for j in range(i + 1, init_nodes):
            if j in candidate_edges[i]:
                G.add_edge(i, j)
    
    # 优先连接添加剩余节点
    for i in range(init_nodes, N):
        candidates = [j for j in candidate_edges[i] if G.has_node(j)]
        if not candidates:
            continue
        
        # 按度数加权选择（优先连接）
        degrees = [G.degree(j) + 1 for j in candidates]  # +1 避免度数为0
        total = sum(degrees)
        probs = [d / total for d in degrees]
        
        # 选择 M 个邻居（不放回）
        targets = []
        temp_candidates = candidates.copy()
        temp_degrees = degrees.copy()
        for _ in range(min(M, len(temp_candidates))):
            total = sum(temp_degrees)
            if total == 0:
                break
            p = [d / total for d in temp_degrees]
            idx = np.random.choice(len(temp_candidates), p=p)
            targets.append(temp_candidates[idx])
            temp_candidates.pop(idx)
            temp_degrees.pop(idx)
        
        for j in targets:
            G.add_edge(i, j)
    
    # 确保连通性：如果不连通，提取最大连通分量
    if not nx.is_connected(G):
        gcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(gcc).copy()
    
    # 重新编号节点为 0..n-1，确保连续性
    G = nx.convert_node_labels_to_integers(G)
    
    return G


def generate_iot_topology_qiu(N, M, area_size=500, comm_radius=200, seed=None):
    """
    生成天津大学 Qiu 团队论文中使用的 IoT 拓扑
    
    特点：
    - 节点按距离中心排序
    - 初始节点全连接
    - 后续节点优先连接度数高的邻居（在通信半径内）
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    positions = {}
    for i in range(N):
        positions[i] = (random.uniform(0, area_size), random.uniform(0, area_size))
    
    G = nx.Graph()
    G.add_nodes_from(range(N))
    nx.set_node_attributes(G, positions, 'pos')
    
    center = (area_size / 2, area_size / 2)
    
    def dist(i, j):
        x1, y1 = positions[i]
        x2, y2 = positions[j]
        return np.hypot(x1 - x2, y1 - y2)
    
    def dist_to_center(i):
        x, y = positions[i]
        return np.hypot(x - center[0], y - center[1])
    
    # 按距离中心排序
    nodes_by_dist = sorted(range(N), key=dist_to_center)
    
    # 初始全连接前 M+1 个节点
    init = min(M + 1, N)
    for i in range(init):
        for j in range(i + 1, init):
            if dist(nodes_by_dist[i], nodes_by_dist[j]) <= comm_radius:
                G.add_edge(nodes_by_dist[i], nodes_by_dist[j])
    
    # 按距离顺序添加剩余节点
    for idx in range(init, N):
        node = nodes_by_dist[idx]
        candidates = [j for j in G.nodes() if j != node and dist(node, j) <= comm_radius and G.degree(j) < N - 1]
        if not candidates:
            continue
        
        degrees = [G.degree(j) + 1 for j in candidates]
        total = sum(degrees)
        probs = [d / total for d in degrees]
        
        num_to_connect = min(M, len(candidates))
        targets = np.random.choice(candidates, size=num_to_connect, replace=False, p=probs)
        for j in targets:
            G.add_edge(node, j)
    
    if not nx.is_connected(G):
        gcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(gcc).copy()
    
    return G


def add_positions_to_graph(G, area_size=500, seed=None):
    """
    为纯拓扑图添加随机位置属性
    用于那些需要位置信息但没有位置属性的图
    """
    if seed is not None:
        random.seed(seed)
    G = G.copy()
    for n in G.nodes():
        G.nodes[n]['pos'] = (random.uniform(0, area_size), random.uniform(0, area_size))
    return G
