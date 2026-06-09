# -*- coding: utf-8 -*-
import networkx as nx
import math
import numpy as np

# 尝试导入 GPU 工具
try:
    from src.utils.gpu_utils import (
        is_gpu_available, to_tensor, to_numpy, TORCH_AVAILABLE, CUDA_AVAILABLE, DEVICE
    )
    if TORCH_AVAILABLE:
        import torch
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    
    def is_gpu_available():
        return False

# 计算有权网络的最短路径
def get_shortest_dist(G, source, dest):
    try:
        # 检查边是否有 'weight' 属性，如果没有，默认权重为 1
        # 使用 nx.shortest_path_length 直接计算距离，比手动累加更高效且兼容无权图
        # 如果图中有权重，nx.shortest_path_length 会自动处理
        try:
            # 尝试作为加权图计算
            dist = nx.shortest_path_length(G, source, dest, weight='weight')
        except TypeError:
             # 如果失败（例如混合类型），则退回无权计算
            dist = nx.shortest_path_length(G, source, dest)
            
        return dist
    except nx.NetworkXNoPath:
        return float('inf')

# 覆盖率计算
# 每个节点都有固定的控制器
def coverage_compute1(G, centers, assignments):
    # communities = list(nx.connected_components(G))
    count = 0
    center_num = len(centers)
    for node in G.nodes():
        if assignments[node] < center_num:
            if nx.has_path(G, node, centers[assignments[node]]):
                count += 1
    return count / G.number_of_nodes()

# 覆盖率计算
# 原理：基于连通分量。只要一个连通分量中包含至少一个控制器，
# 该分量内的所有节点都被视为“已覆盖”（可达）。
def coverage_compute2(G, centers):
    current_total_nodes = G.number_of_nodes()
    if current_total_nodes == 0:
        return 0.0
        
    # 获取图中所有的连通分量
    communities = list(nx.connected_components(G))
    centers_set = set(centers)
    count = 0
    
    # 遍历每个连通分量
    for community in communities:
        # isdisjoint返回True表示无交集。
        # not isdisjoint 表示有交集，即该社区内至少有一个控制器
        if not centers_set.isdisjoint(community):
            count += len(community)
            
    # 修正：返回绝对数量，以便在主程序中统一进行归一化处理（除以初始节点数）
    # 这样可以保证单调性
    return count

# 网络传输效率计算
# 定义为：(节点到最近控制器的效率之和) + (控制器之间的互联效率之和)
# 效率 = 1 / 最短路径距离
# 注意：此函数返回的是累加值，未进行归一化处理。
def weight_efficiency_compute(G, centers):
    efficiency = 0.
    
    # 1. 计算【节点-控制器】通信效率
    # 对于每个非控制器节点，找到离它最近的控制器，计算距离倒数
    for node in G.nodes():
        if node not in centers:
            min_dis = float('inf')
            
            # 遍历所有控制器，找最近的一个
            for center in centers:
                # 优化：先尝试获取距离，捕获异常，比先has_path再计算更快
                try:
                    # 使用 helper 函数计算带权距离
                    # 若无路经，get_shortest_dist 应返回 float('inf')
                    if nx.has_path(G, node, center):
                        temp_dis = get_shortest_dist(G, node, center)
                        if temp_dis < min_dis:
                            min_dis = temp_dis
                except Exception:
                    pass
            
            # 只有当节点能连接到至少一个控制器时，才计算效率
            if min_dis != 0 and min_dis != float('inf'):
                efficiency += 1 / min_dis

    # 2. 计算【控制器-控制器】同步效率
    # 计算所有控制器对之间的距离倒数
    center_num = len(centers)
    temp = 0.
    for i in range(center_num):
        source = centers[i]
        # 只计算 j > i 的部分，避免重复计算 (i, j) 和 (j, i)
        for j in range(i + 1, center_num):
            dest = centers[j]
            if nx.has_path(G, source, dest):
                d = get_shortest_dist(G, source, dest)
                if d != 0:
                    temp += 1 / d
                    
    # 返回两部分效率之和
    return efficiency + temp

# 计算含控制器的最大联通子图大小
def max_component_size_compute(G, centers):
    if G.number_of_nodes() == 0:
        return 0
    communities = list(sorted(nx.connected_components(G), key=len, reverse=True))
    for community in communities:
        if not community.isdisjoint(centers):
            return len(community)
    return 0

# 统一指标接口
def calculate_gcc(G, centers):
    """
    计算最大连通子图 (GCC) 指标
    注意：这里计算的是绝对节点数，归一化在主循环中进行
    """
    return max_component_size_compute(G, centers)

def calculate_efficiency(G, centers):
    """
    计算网络加权效率指标
    """
    return weight_efficiency_compute(G, centers)

def calculate_coverage(G, centers):
    """
    计算覆盖率指标 (基于连通分量)
    """
    return coverage_compute2(G, centers)

def calculate_gcc(G, centers):
    """
    计算包含控制器的最大连通子图 (GCC - Giant Connected Component with Controllers)
    
    在 SDN 网络中，只有包含控制器的连通分量才是有效的（可控制的）。
    没有控制器的分量会发生级联失效。
    
    Args:
        G (nx.Graph): 网络拓扑
        centers (list): 控制器节点列表
        
    Returns:
        int: 包含控制器的最大连通子图的节点数
    """
    if G.number_of_nodes() == 0:
        return 0
    
    components = list(nx.connected_components(G))
    if not components:
        return 0
    
    centers_set = set(centers) if centers else set()
    
    # 如果没有控制器，返回0（网络完全失控）
    if not centers_set:
        return 0
    
    # 找出包含控制器的所有连通分量
    controlled_components = []
    for comp in components:
        # 检查该分量是否包含至少一个控制器
        if not centers_set.isdisjoint(comp):
            controlled_components.append(comp)
    
    # 如果没有包含控制器的分量，返回0
    if not controlled_components:
        return 0
    
    # 返回包含控制器的最大连通分量的大小
    return len(max(controlled_components, key=len))

# -----------------------------------------------------------
# 新增的指标计算函数 (CSA, H_C, WCP)
# -----------------------------------------------------------

def calculate_csa(G, centers, alpha=0.2, initial_num_switches=None):
    """
    计算控制供给可用性 (Control Supply Availability, CSA)
    CSA = 1/|V_S| * sum(1 - e^(-alpha * k_i))
    
    支持 GPU 加速计算
    
    修改说明：
    1. 如果提供了initial_num_switches，分母固定为初始交换节点总数
    2. 否则，分母使用当前剩余交换节点数
    
    Args:
        G (nx.Graph): 网络拓扑
        centers (list): 控制器节点列表
        alpha (float): 衰减因子，默认 0.2
        initial_num_switches (int, optional): 初始交换节点总数，如果提供则使用固定分母
        
    Returns:
        float: CSA 值
    """
    if G.number_of_nodes() == 0:
        return 0.0
        
    centers_set = set(centers)
    switch_nodes = [n for n in G.nodes() if n not in centers_set]
    
    if not switch_nodes:
        return 0.0
    
    # 检查是否使用 GPU 加速
    use_gpu = is_gpu_available() and TORCH_AVAILABLE and len(switch_nodes) > 100
    
    # 预计算连通分量及其包含的控制器数量
    components = list(nx.connected_components(G))
    node_to_k = {}
    
    for comp in components:
        # 计算该分量内的控制器数量
        num_centers_in_comp = len(comp.intersection(centers_set))
        for node in comp:
            node_to_k[node] = num_centers_in_comp
    
    if use_gpu:
        # GPU 加速版本
        k_values = np.array([node_to_k.get(i, 0) for i in switch_nodes], dtype=np.float32)
        k_tensor = torch.tensor(k_values, device=DEVICE)
        
        # 计算 1 - exp(-alpha * k_i)
        csa_tensor = 1 - torch.exp(-alpha * k_tensor)
        total_csa = torch.sum(csa_tensor).item()
    else:
        # CPU 版本
        total_csa = 0.0
        for i in switch_nodes:
            k_i = node_to_k.get(i, 0)
            term = 1 - math.exp(-alpha * k_i)
            total_csa += term
    
    # 确定分母：如果提供了initial_num_switches，使用固定值；否则使用当前交换节点数
    if initial_num_switches is not None:
        denominator = initial_num_switches
    else:
        denominator = len(switch_nodes)
    
    if denominator == 0:
        return 0.0
    
    return total_csa / denominator

def calculate_control_entropy(G, centers, initial_sum_A=None):
    """
    计算控制熵 (Control Entropy, H_C)
    H_C = - sum(p_j * ln(p_j))
    p_j = A_j / D
    A_j: 控制器 j 的服务域大小 (连通的交换节点数)
    D: 分母，如果提供了initial_sum_A，则使用该固定值；否则使用当前状态的动态和
    
    修改说明：
    1. 如果提供了initial_sum_A，分母D固定为初始状态的sum(A_m_initial)ss
    2. 否则，分母D = 未被控制的节点数 + sum(A_m_current)
    
    Args:
        G (nx.Graph): 网络拓扑
        centers (list): 控制器节点列表
        initial_sum_A (float, optional): 初始状态的sum(A_m)值，如果提供则使用固定分母
        
    Returns:
        float: H_C 值
    """
    if not centers:
        return 0.0
        
    centers_set = set(centers)
    switch_nodes = [n for n in G.nodes() if n not in centers_set]
    
    if not switch_nodes:
        return 0.0
        
    # 计算每个控制器的服务域 A_j
    # A_j: 能连通到控制器 j 的交换节点数量
    # 实际上，就是控制器 j 所在连通分量中的交换节点数量
    
    # 预计算连通分量
    components = list(nx.connected_components(G))
    
    # 优化算法：
    # 1. 遍历所有连通分量
    # 2. 对每个分量，计算其中 switch_nodes 的数量 (size_sw)
    # 3. 该分量中的所有控制器，其 A_j 都等于 size_sw
    # 4. 统计未被控制的交换节点数（不在任何包含控制器的分量中的交换节点）
    
    A_map = {} # center -> A_j
    controlled_switches = set()  # 被控制的交换节点集合
    
    for comp in components:
        # 计算该分量内的交换节点数量
        switches_in_comp = [n for n in comp if n not in centers_set]
        size_sw = len(switches_in_comp)
        
        # 找出该分量内的所有控制器
        centers_in_comp = [n for n in comp if n in centers_set]
        
        if centers_in_comp:
            # 该分量包含控制器，其中的交换节点都被控制
            for switch in switches_in_comp:
                controlled_switches.add(switch)
            
            # 该分量中的所有控制器，其 A_j 都等于 size_sw
            for c in centers_in_comp:
                A_map[c] = size_sw
    
    # 计算未被控制的交换节点数
    uncontrolled_switches = len(switch_nodes) - len(controlled_switches)
    
    # 提取 A_j 列表 (按照 centers 的顺序或任意顺序，求和不影响)
    # 注意：如果有控制器是孤立的且没有交换节点连接，A_j = 0
    A_list = [A_map.get(c, 0) for c in centers]
    sum_A = sum(A_list)
    
    # 确定分母D：如果提供了initial_sum_A，使用固定值；否则使用当前状态的动态和
    if initial_sum_A is not None:
        # 使用初始状态的固定分母
        total_denominator = initial_sum_A
    else:
        # 使用当前状态的动态和
        total_denominator = uncontrolled_switches + sum_A
    
    if total_denominator == 0:
        return 0.0
        
    H_C = 0.0
    
    # 计算控制器的熵贡献
    for A_j in A_list:
        if A_j > 0:
            p_j = A_j / total_denominator
            H_C += p_j * math.log(p_j)
    
    # 计算未被控制节点的熵贡献（如果存在且使用动态分母）
    # 注意：如果使用固定分母initial_sum_A，未被控制节点不应该参与熵计算
    # 因为它们不在初始的sum(A_m)中
    if initial_sum_A is None and uncontrolled_switches > 0:
        p_uncontrolled = uncontrolled_switches / total_denominator
        H_C += p_uncontrolled * math.log(p_uncontrolled)
            
    return -H_C

def calculate_wcp(G, centers, beta=1.0, initial_num_switches=None):
    """
    计算加权控制势能 (Weighted Control Potential, WCP)
    WCP = 1/|V_S| * sum_i ( sum_j ( 1 / (d_ij)^beta ) )
    
    支持 GPU 加速计算
    
    修改说明：
    1. 如果提供了initial_num_switches，分母固定为初始交换节点总数
    2. 否则，分母使用当前剩余交换节点数
    3. 使用固定分母可以保证WCP成为单调递减指标
    
    Args:
        G (nx.Graph): 网络拓扑
        centers (list): 控制器节点列表
        beta (float): 距离衰减因子
        initial_num_switches (int, optional): 初始交换节点总数，如果提供则使用固定分母
        
    Returns:
        float: WCP 值
    """
    centers_set = set(centers)
    switch_nodes = [n for n in G.nodes() if n not in centers_set]
    num_switches = len(switch_nodes)
    
    if num_switches == 0:
        return 0.0
    
    # 检查是否使用 GPU 加速
    use_gpu = is_gpu_available() and TORCH_AVAILABLE and num_switches > 50
    
    if use_gpu:
        # GPU 加速版本
        nodes_list = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes_list)}
        switch_indices = [node_to_idx[n] for n in switch_nodes if n in node_to_idx]
        
        # 预计算从所有控制器到所有节点的距离
        num_centers = len(centers)
        num_nodes = len(nodes_list)
        
        # 距离矩阵: centers x nodes
        dist_matrix = np.full((num_centers, num_nodes), np.inf, dtype=np.float32)
        
        for i, center in enumerate(centers):
            try:
                lengths = nx.single_source_shortest_path_length(G, center)
                for target, dist in lengths.items():
                    if target in node_to_idx:
                        dist_matrix[i, node_to_idx[target]] = dist
            except Exception:
                continue
        
        # 转移到 GPU
        dist_tensor = torch.tensor(dist_matrix, dtype=torch.float32, device=DEVICE)
        
        # 提取交换节点的距离
        switch_dists = dist_tensor[:, switch_indices]  # centers x switches
        
        # 计算 1/dist^beta，处理 inf 和 0
        valid_mask = (switch_dists > 0) & (switch_dists < float('inf'))
        inv_dists = torch.where(valid_mask, 1.0 / (switch_dists ** beta), torch.zeros_like(switch_dists))
        
        # 对每个交换节点求和所有控制器的贡献
        switch_potentials = torch.sum(inv_dists, dim=0)  # shape: (switches,)
        
        # 计算总 WCP
        total_wcp = torch.sum(switch_potentials).item()
        
    else:
        # CPU 版本 (原始实现)
        # 累加器：每个交换机 i 的 sum_inv_dist
        node_potentials = {n: 0.0 for n in switch_nodes}
        
        # 使用多源 Dijkstra 或 BFS 来处理
        # 为了准确性和兼容性，我们检查图是否有权重
        is_weighted = nx.is_weighted(G)
        
        for center in centers:
            # 计算从 center 到所有节点的最短路径
            try:
                if is_weighted:
                    lengths = nx.single_source_dijkstra_path_length(G, center, weight='weight')
                else:
                    lengths = nx.single_source_shortest_path_length(G, center)
            except Exception:
                continue
                
            for target, dist in lengths.items():
                if target in node_potentials:
                    if dist > 0:
                        val = 1.0 / (dist ** beta)
                        node_potentials[target] += val
                    
        # 对所有交换机求和
        total_wcp = sum(node_potentials.values())
    
    # 确定分母：如果提供了initial_num_switches，使用固定值；否则使用当前交换节点数
    if initial_num_switches is not None:
        denominator = initial_num_switches
    else:
        denominator = num_switches
    
    if denominator == 0:
        return 0.0
    
    return total_wcp / denominator

def get_metric_function(metric_name):
    """
    根据名称获取指标计算函数
    """
    if metric_name == "efficiency":
        return calculate_efficiency
    elif metric_name == "coverage":
        return calculate_coverage
    else:
        # 默认为 GCC
        return calculate_gcc


# -----------------------------------------------------------
# R值计算函数（统一x网格插值方法）
# -----------------------------------------------------------

def calculate_r_value_interpolated(x_curve, y_curve, num_points=101):
    """
    计算R值（鲁棒性指标）：使用统一x网格插值后计算曲线下面积
    
    问题背景：
    由于多层网络拆解中存在级联失效，不同方法的x_curve采样点分布不同：
    1. 采样点不均匀（每次攻击可能导致多个节点失效，x值跳跃式增长）
    2. 曲线长度不同（某方法可能提前崩溃）
    
    解决方案：
    将所有曲线重采样到统一的x网格（0到1之间num_points个点），然后计算面积
    
    Args:
        x_curve (list): x轴数据（移除节点比例）
        y_curve (list): y轴数据（韧性指标值）
        num_points (int): 统一x网格的采样点数，默认101（对应0.01的间隔）
        
    Returns:
        float: R值（归一化后的曲线下面积）
    """
    if len(x_curve) < 2 or len(y_curve) < 2:
        return 0.0
    
    # 1. 确保数据有效
    x_arr = np.array(x_curve, dtype=np.float64)
    y_arr = np.array(y_curve, dtype=np.float64)
    
    # 2. 处理数据：确保x是单调递增的（移除重复点）
    # 找到唯一的x值及其对应的y值
    unique_indices = []
    seen_x = set()
    for i, x in enumerate(x_arr):
        if x not in seen_x:
            unique_indices.append(i)
            seen_x.add(x)
    
    if len(unique_indices) < 2:
        return 0.0
    
    x_arr = x_arr[unique_indices]
    y_arr = y_arr[unique_indices]
    
    # 3. 确保x范围从0开始
    if x_arr[0] > 0:
        # 在开头插入(0, y_arr[0])
        x_arr = np.concatenate([[0.0], x_arr])
        y_arr = np.concatenate([[y_arr[0]], y_arr])
    
    # 4. 确保x范围到1结束（曲线可能提前崩溃）
    if x_arr[-1] < 1.0:
        # 在末尾插入(1.0, y_arr[-1])或(1.0, 0.0)
        # 如果最后的y值已经是0或接近0，则填充0
        last_y = y_arr[-1] if y_arr[-1] > 1e-10 else 0.0
        x_arr = np.concatenate([x_arr, [1.0]])
        y_arr = np.concatenate([y_arr, [last_y]])
    
    # 5. 创建统一的x网格
    x_uniform = np.linspace(0.0, 1.0, num_points)
    
    # 6. 线性插值到统一网格
    y_interpolated = np.interp(x_uniform, x_arr, y_arr)
    
    # 7. 使用梯形法则计算面积
    r_value = np.trapz(y_interpolated, x_uniform)
    
    return r_value


def calculate_r_value_simple(x_curve, y_curve):
    """
    简单R值计算方法：直接使用梯形法则
    
    注意：此方法在不同曲线的x采样点不同时可能不够准确
    建议使用 calculate_r_value_interpolated 替代
    
    Args:
        x_curve (list): x轴数据
        y_curve (list): y轴数据
        
    Returns:
        float: R值
    """
    if len(x_curve) < 2:
        return 0.0
    return np.trapz(y_curve, x_curve)
