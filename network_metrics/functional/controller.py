# -*- coding: utf-8 -*-
"""
控制器相关功能指标：CSA、CCE、WCP
"""
import math
import numpy as np
import networkx as nx
from typing import List, Tuple

# 尝试导入 GPU 工具
try:
    from src.utils.gpu_utils import is_gpu_available, TORCH_AVAILABLE, CUDA_AVAILABLE, DEVICE
    if TORCH_AVAILABLE:
        import torch
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

    def is_gpu_available():
        return False


def calculate_csa(G, centers, alpha=0.2, initial_num_switches=None):
    """
    控制供给可用性 (Control Supply Availability, CSA)
    CSA = 1/|V_S| * sum(1 - e^(-alpha * k_i))

    Args:
        G: networkx.Graph
        centers: 控制器节点列表
        alpha: 衰减因子，默认 0.2
        initial_num_switches: 初始交换节点总数（固定分母），可选

    Returns:
        float: CSA 值
    """
    if G.number_of_nodes() == 0:
        return 0.0

    centers_set = set(centers)
    switch_nodes = [n for n in G.nodes() if n not in centers_set]
    if not switch_nodes:
        return 0.0

    use_gpu = is_gpu_available() and TORCH_AVAILABLE and len(switch_nodes) > 100

    components = list(nx.connected_components(G))
    node_to_k = {}
    for comp in components:
        num_centers = len(comp.intersection(centers_set))
        for node in comp:
            node_to_k[node] = num_centers

    if use_gpu:
        k_values = np.array([node_to_k.get(i, 0) for i in switch_nodes], dtype=np.float32)
        k_tensor = torch.tensor(k_values, device=DEVICE)
        csa_tensor = 1 - torch.exp(-alpha * k_tensor)
        total_csa = torch.sum(csa_tensor).item()
    else:
        total_csa = 0.0
        for i in switch_nodes:
            k_i = node_to_k.get(i, 0)
            total_csa += 1 - math.exp(-alpha * k_i)

    denominator = initial_num_switches if initial_num_switches is not None else len(switch_nodes)
    if denominator == 0:
        return 0.0
    return total_csa / denominator


def calculate_cce(G, centers, initial_sum_A=None):
    """
    控制集中度熵 (Control Centralization Entropy, CCE)
    H_C = - sum(p_j * ln(p_j))

    Args:
        G: networkx.Graph
        centers: 控制器节点列表
        initial_sum_A: 初始 sum(A_m) 值（固定分母），可选

    Returns:
        float: CCE 值
    """
    if not centers:
        return 0.0

    centers_set = set(centers)
    switch_nodes = [n for n in G.nodes() if n not in centers_set]
    if not switch_nodes:
        return 0.0

    components = list(nx.connected_components(G))
    A_map = {}
    controlled_switches = set()

    for comp in components:
        switches_in_comp = [n for n in comp if n not in centers_set]
        size_sw = len(switches_in_comp)
        centers_in_comp = [n for n in comp if n in centers_set]
        if centers_in_comp:
            for switch in switches_in_comp:
                controlled_switches.add(switch)
            for c in centers_in_comp:
                A_map[c] = size_sw

    uncontrolled = len(switch_nodes) - len(controlled_switches)
    A_list = [A_map.get(c, 0) for c in centers]
    sum_A = sum(A_list)

    total_denominator = initial_sum_A if initial_sum_A is not None else (uncontrolled + sum_A)
    if total_denominator == 0:
        return 0.0

    H_C = 0.0
    for A_j in A_list:
        if A_j > 0:
            p_j = A_j / total_denominator
            H_C += p_j * math.log(p_j)

    if initial_sum_A is None and uncontrolled > 0:
        p_uncontrolled = uncontrolled / total_denominator
        H_C += p_uncontrolled * math.log(p_uncontrolled)

    return -H_C


def calculate_wcp(G, centers, beta=1.0, initial_num_switches=None):
    """
    加权控制势能 (Weighted Control Potential, WCP)
    WCP = 1/|V_S| * sum_i ( sum_j ( 1 / (d_ij)^beta ) )

    Args:
        G: networkx.Graph
        centers: 控制器节点列表
        beta: 距离衰减因子
        initial_num_switches: 初始交换节点总数（固定分母），可选

    Returns:
        float: WCP 值
    """
    centers_set = set(centers)
    switch_nodes = [n for n in G.nodes() if n not in centers_set]
    num_switches = len(switch_nodes)
    if num_switches == 0:
        return 0.0

    use_gpu = is_gpu_available() and TORCH_AVAILABLE and num_switches > 50

    if use_gpu:
        nodes_list = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes_list)}
        switch_indices = [node_to_idx[n] for n in switch_nodes if n in node_to_idx]
        num_centers = len(centers)
        num_nodes = len(nodes_list)
        dist_matrix = np.full((num_centers, num_nodes), np.inf, dtype=np.float32)

        for i, center in enumerate(centers):
            try:
                lengths = nx.single_source_shortest_path_length(G, center)
                for target, dist in lengths.items():
                    if target in node_to_idx:
                        dist_matrix[i, node_to_idx[target]] = dist
            except Exception:
                continue

        dist_tensor = torch.tensor(dist_matrix, dtype=torch.float32, device=DEVICE)
        switch_dists = dist_tensor[:, switch_indices]
        valid_mask = (switch_dists > 0) & (switch_dists < float('inf'))
        inv_dists = torch.where(valid_mask, 1.0 / (switch_dists ** beta), torch.zeros_like(switch_dists))
        switch_potentials = torch.sum(inv_dists, dim=0)
        total_wcp = torch.sum(switch_potentials).item()
    else:
        node_potentials = {n: 0.0 for n in switch_nodes}
        is_weighted = nx.is_weighted(G)
        for center in centers:
            try:
                if is_weighted:
                    lengths = nx.single_source_dijkstra_path_length(G, center, weight='weight')
                else:
                    lengths = nx.single_source_shortest_path_length(G, center)
            except Exception:
                continue
            for target, dist in lengths.items():
                if target in node_potentials and dist > 0:
                    node_potentials[target] += 1.0 / (dist ** beta)
        total_wcp = sum(node_potentials.values())

    denominator = initial_num_switches if initial_num_switches is not None else num_switches
    if denominator == 0:
        return 0.0
    return total_wcp / denominator


def simulate_cce_trajectory(
    G: nx.Graph,
    attack_sequence: List[int],
    centers: List[int],
    stop_threshold: float = 0.1
) -> List[Tuple[int, float, float]]:
    """
    模拟攻击过程中 CCE 的变化轨迹。

    每一步记录：
        (已移除节点数量, 已移除节点比例, 当前 CCE 值)

    攻击在最大LCC降到初始节点数的 stop_threshold 比例时停止。

    Args:
        G: 初始网络拓扑
        attack_sequence: 节点移除顺序列表
        centers: 初始控制器节点列表
        stop_threshold: 停止阈值，默认 0.1

    Returns:
        List[Tuple[int, float, float]]: CCE 轨迹。
        首元素为初始状态 (0, 0.0, initial_cce)。
    """
    n_init = G.number_of_nodes()
    if n_init == 0 or not centers:
        return []

    H = G.copy()
    alive_centers = set(centers)
    trajectory = []

    # 记录初始状态
    trajectory.append((0, 0.0, calculate_cce(H, list(alive_centers))))

    stop_size = max(1, int(np.ceil(n_init * stop_threshold)))

    for step, node in enumerate(attack_sequence, start=1):
        if node in H:
            H.remove_node(node)
        if node in alive_centers:
            alive_centers.remove(node)

        if H.number_of_nodes() == 0:
            trajectory.append((step, step / n_init, 0.0))
            break

        # 若所有控制器都被摧毁，CCE 为 0
        cce_val = calculate_cce(H, list(alive_centers)) if alive_centers else 0.0
        trajectory.append((step, step / n_init, cce_val))

        # 检查 GCC 停止条件
        sizes = sorted([len(c) for c in nx.connected_components(H)], reverse=True)
        max_lcc = sizes[0] if sizes else 0
        if max_lcc <= stop_size:
            break

    return trajectory


def simulate_csa_trajectory(
    G: nx.Graph,
    attack_sequence: List[int],
    centers: List[int],
    stop_threshold: float = 0.1,
    alpha: float = 0.2
) -> dict:
    """
    模拟攻击过程中 CSA 的变化轨迹，并计算 CSA_AUC。

    Args:
        G: 初始网络拓扑
        attack_sequence: 节点移除顺序列表
        centers: 初始控制器节点列表
        stop_threshold: 停止阈值，默认 0.1
        alpha: CSA 衰减因子，默认 0.2

    Returns:
        dict: {
            'trajectory': List[Tuple[int, float, float]],  # (移除数量, 移除比例, CSA值)
            'auc': float,                                   # CSA 曲线下面积
            'initial_nodes': int,
            'final_step': int,
        }
    """
    n_init = G.number_of_nodes()
    if n_init == 0 or not centers:
        return {'trajectory': [], 'auc': 0.0, 'initial_nodes': 0, 'final_step': 0}

    H = G.copy()
    alive_centers = set(centers)
    trajectory = []

    # 计算初始固定分母
    initial_num_switches = len([n for n in G.nodes() if n not in alive_centers])

    # 记录初始状态
    trajectory.append((0, 0.0, calculate_csa(H, list(alive_centers), alpha=alpha, initial_num_switches=initial_num_switches)))

    stop_size = max(1, int(np.ceil(n_init * stop_threshold)))

    for step, node in enumerate(attack_sequence, start=1):
        if node in H:
            H.remove_node(node)
        if node in alive_centers:
            alive_centers.remove(node)

        if H.number_of_nodes() == 0:
            trajectory.append((step, step / n_init, 0.0))
            break

        csa_val = calculate_csa(H, list(alive_centers), alpha=alpha, initial_num_switches=initial_num_switches) if alive_centers else 0.0
        trajectory.append((step, step / n_init, csa_val))

        # 检查 GCC 停止条件
        sizes = sorted([len(c) for c in nx.connected_components(H)], reverse=True)
        max_lcc = sizes[0] if sizes else 0
        if max_lcc <= stop_size:
            break

    # 计算 CSA AUC
    x = [t[1] for t in trajectory]
    y = [t[2] for t in trajectory]
    if len(x) >= 2:
        if x[-1] < 1.0:
            x.append(1.0)
            y.append(0.0)
        auc_val = float(np.trapz(y, x))
    else:
        auc_val = 0.0

    return {
        'trajectory': trajectory,
        'auc': auc_val,
        'initial_nodes': n_init,
        'final_step': trajectory[-1][0] if trajectory else 0,
    }


def simulate_wcp_trajectory(
    G: nx.Graph,
    attack_sequence: List[int],
    centers: List[int],
    stop_threshold: float = 0.1,
    beta: float = 1.0
) -> dict:
    """
    模拟攻击过程中 WCP 的变化轨迹，并计算 WCP_AUC。

    Args:
        G: 初始网络拓扑
        attack_sequence: 节点移除顺序列表
        centers: 初始控制器节点列表
        stop_threshold: 停止阈值，默认 0.1
        beta: WCP 距离衰减因子，默认 1.0

    Returns:
        dict: {
            'trajectory': List[Tuple[int, float, float]],  # (移除数量, 移除比例, WCP值)
            'auc': float,                                   # WCP 曲线下面积
            'initial_nodes': int,
            'final_step': int,
        }
    """
    n_init = G.number_of_nodes()
    if n_init == 0 or not centers:
        return {'trajectory': [], 'auc': 0.0, 'initial_nodes': 0, 'final_step': 0}

    H = G.copy()
    alive_centers = set(centers)
    trajectory = []

    # 计算初始固定分母
    initial_num_switches = len([n for n in G.nodes() if n not in alive_centers])

    # 记录初始状态
    trajectory.append((0, 0.0, calculate_wcp(H, list(alive_centers), beta=beta, initial_num_switches=initial_num_switches)))

    stop_size = max(1, int(np.ceil(n_init * stop_threshold)))

    for step, node in enumerate(attack_sequence, start=1):
        if node in H:
            H.remove_node(node)
        if node in alive_centers:
            alive_centers.remove(node)

        if H.number_of_nodes() == 0:
            trajectory.append((step, step / n_init, 0.0))
            break

        wcp_val = calculate_wcp(H, list(alive_centers), beta=beta, initial_num_switches=initial_num_switches) if alive_centers else 0.0
        trajectory.append((step, step / n_init, wcp_val))

        # 检查 GCC 停止条件
        sizes = sorted([len(c) for c in nx.connected_components(H)], reverse=True)
        max_lcc = sizes[0] if sizes else 0
        if max_lcc <= stop_size:
            break

    # 计算 WCP AUC
    x = [t[1] for t in trajectory]
    y = [t[2] for t in trajectory]
    if len(x) >= 2:
        if x[-1] < 1.0:
            x.append(1.0)
            y.append(0.0)
        auc_val = float(np.trapz(y, x))
    else:
        auc_val = 0.0

    return {
        'trajectory': trajectory,
        'auc': auc_val,
        'initial_nodes': n_init,
        'final_step': trajectory[-1][0] if trajectory else 0,
    }


def simulate_cce_trajectory_with_auc(
    G: nx.Graph,
    attack_sequence: List[int],
    centers: List[int],
    stop_threshold: float = 0.1
) -> dict:
    """
    模拟攻击过程中 CCE 的变化轨迹，并计算 CCE_AUC。

    Args:
        G: 初始网络拓扑
        attack_sequence: 节点移除顺序列表
        centers: 初始控制器节点列表
        stop_threshold: 停止阈值，默认 0.1

    Returns:
        dict: {
            'trajectory': List[Tuple[int, float, float]],  # (移除数量, 移除比例, CCE值)
            'auc': float,                                   # CCE 曲线下面积
            'initial_nodes': int,
            'final_step': int,
        }
    """
    n_init = G.number_of_nodes()
    if n_init == 0 or not centers:
        return {'trajectory': [], 'auc': 0.0, 'initial_nodes': 0, 'final_step': 0}

    H = G.copy()
    alive_centers = set(centers)
    trajectory = []

    # 记录初始状态
    trajectory.append((0, 0.0, calculate_cce(H, list(alive_centers))))

    stop_size = max(1, int(np.ceil(n_init * stop_threshold)))

    for step, node in enumerate(attack_sequence, start=1):
        if node in H:
            H.remove_node(node)
        if node in alive_centers:
            alive_centers.remove(node)

        if H.number_of_nodes() == 0:
            trajectory.append((step, step / n_init, 0.0))
            break

        cce_val = calculate_cce(H, list(alive_centers)) if alive_centers else 0.0
        trajectory.append((step, step / n_init, cce_val))

        # 检查 GCC 停止条件
        sizes = sorted([len(c) for c in nx.connected_components(H)], reverse=True)
        max_lcc = sizes[0] if sizes else 0
        if max_lcc <= stop_size:
            break

    # 计算 CCE AUC
    x = [t[1] for t in trajectory]
    y = [t[2] for t in trajectory]
    if len(x) >= 2:
        if x[-1] < 1.0:
            x.append(1.0)
            y.append(0.0)
        auc_val = float(np.trapz(y, x))
    else:
        auc_val = 0.0

    return {
        'trajectory': trajectory,
        'auc': auc_val,
        'initial_nodes': n_init,
        'final_step': trajectory[-1][0] if trajectory else 0,
    }
