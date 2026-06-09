# -*- coding: utf-8 -*-
import networkx as nx
import random
import logging
import numpy as np
from itertools import combinations
from tqdm import tqdm
from network_metrics.functional.network import max_component_size_compute as calculate_gcc
from network_metrics.functional.network import coverage_compute2, weight_efficiency_compute, max_component_size_compute
from network_metrics.functional.controller import calculate_csa, calculate_cce as calculate_control_entropy, calculate_wcp
from src.utils.logger import get_logger
from src.utils.io import save_metric_log
import network_dismantling as nd  # 统一拆解接口

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

logger = get_logger(__name__)

# -------------------------------------------------------
# New Dismantling Strategies (Static)
# -------------------------------------------------------

def bruteforce_dismantler(G, stop_condition=None, max_k=None):
    """
    使用暴力枚举方法找到最优网络拆解序列。
    该方法计算复杂度极高，仅适用于非常小的网络。
    
    Args:
        G (nx.Graph): 网络图。
        stop_condition (float, optional): 停止条件（如GCC大小），默认为网络大小的10%。
        max_k (int, optional): 最大移除节点数，默认为网络大小。
    
    Returns:
        list: 拆解节点序列。
    """
    network_size = G.number_of_nodes()
    if stop_condition is None:
        stop_condition = max(1, int(0.1 * network_size))
    if max_k is None:
        max_k = network_size
    
    max_k = min(max_k, 10)
    nodes = list(G.nodes())
    best_combination = None
    
    for k in range(1, max_k + 1):
        best_score = network_size
        for combination in combinations(range(network_size), k):
            temp_G = G.copy()
            temp_G.remove_nodes_from([nodes[i] for i in combination])
            
            if temp_G.number_of_nodes() > 0:
                components = list(nx.connected_components(temp_G))
                largest_cc_size = len(max(components, key=len)) if components else 0
            else:
                largest_cc_size = 0
            
            if largest_cc_size < best_score:
                best_score = largest_cc_size
                best_combination = [nodes[i] for i in combination]
        
        if best_score <= stop_condition:
            break
    
    if best_combination is None:
        return []
    
    dismantling_sequence = list(best_combination)
    remaining_nodes = [node for node in nodes if node not in dismantling_sequence]
    remaining_nodes.sort(key=lambda x: G.degree(x), reverse=True)
    dismantling_sequence.extend(remaining_nodes)
    return dismantling_sequence

def degree_dismantling(G):
    """
    基于度中心性（静态）的拆解策略。
    一次性计算所有节点的度，并按降序排列。
    
    支持 GPU 加速计算
    
    Args:
        G (nx.Graph): 网络图。
        
    Returns:
        list: 拆解节点序列。
    """
    nodes = list(G.nodes())
    
    if is_gpu_available() and len(nodes) > 100:
        # GPU 加速版本：使用 GPU 进行排序
        degrees = np.array([G.degree(n) for n in nodes], dtype=np.float32)
        degree_tensor = torch.tensor(degrees, device=DEVICE)
        
        # 获取排序索引（降序）
        _, sorted_indices = torch.sort(degree_tensor, descending=True)
        sorted_indices = sorted_indices.cpu().numpy()
        
        return [nodes[i] for i in sorted_indices]
    else:
        # CPU 版本
        sorted_nodes = sorted(nodes, key=G.degree, reverse=True)
        return list(sorted_nodes)

def random_dismantling(G):
    """
    随机拆解策略。
    随机打乱节点顺序。
    
    Args:
        G (nx.Graph): 网络图。
        
    Returns:
        list: 拆解节点序列。
    """
    random_nodes = list(G.nodes())
    random.shuffle(random_nodes)
    return random_nodes

def pagerank_dismantling(G):
    """
    基于 PageRank 的拆解策略。
    
    Args:
        G (nx.Graph): 网络图。
        
    Returns:
        list: 拆解节点序列。
    """
    pagerank = nx.pagerank(G)
    sorted_nodes = sorted(G.nodes(), key=lambda x: pagerank[x], reverse=True)
    return list(sorted_nodes)

def betweenness_dismantling(G):
    """
    基于介数中心性（Betweenness Centrality）的拆解策略。
    
    Args:
        G (nx.Graph): 网络图。
        
    Returns:
        list: 拆解节点序列。
    """
    betweenness = nx.betweenness_centrality(G)
    sorted_nodes = sorted(G.nodes(), key=lambda x: betweenness[x], reverse=True)
    return list(sorted_nodes)

def eigenvector_dismantling(G):
    """
    基于特征向量中心性（Eigenvector Centrality）的拆解策略。
    
    Args:
        G (nx.Graph): 网络图。
        
    Returns:
        list: 拆解节点序列。
    """
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        sorted_nodes = sorted(G.nodes(), key=lambda x: eigenvector[x], reverse=True)
        return list(sorted_nodes)
    except:
        # 如果不收敛，回退到度中心性
        return degree_dismantling(G)

# -------------------------------------------------------
# Adaptive Dismantling (To preserve original behavior)
# -------------------------------------------------------

def adaptive_degree_dismantling(G):
    """
    自适应度拆解策略（Target Attack）。
    每移除一个节点后，重新计算剩余节点的度，选择度最大的节点进行下一轮移除。
    
    注意：此函数返回的是攻击序列，但为了计算序列，它实际上在临时图上模拟了拆解过程。
    
    Args:
        G (nx.Graph): 网络图。
        
    Returns:
        list: 拆解节点序列。
    """
    G_temp = G.copy()
    sequence = []
    while G_temp.number_of_nodes() > 0:
        # 计算当前图的度中心性
        node_centrality = nx.degree_centrality(G_temp)
        if not node_centrality:
             break
        # 选择度最大的节点
        node = max(node_centrality, key=node_centrality.get)
        sequence.append(node)
        
        # 移除节点及其边
        if G_temp.has_node(node):
            edges = list(G_temp.edges(node))
            G_temp.remove_edges_from(edges)
            G_temp.remove_node(node)
            
    return sequence

def adaptive_hybrid_dismantling(G):
    """
    自适应混合拆解策略（Hybrid Attack）。
    每一步有 20% 的概率选择度最大的节点，80% 的概率随机选择节点。
    
    Args:
        G (nx.Graph): 网络图。
        
    Returns:
        list: 拆解节点序列。
    """
    G_temp = G.copy()
    sequence = []
    while G_temp.number_of_nodes() > 0:
        random_number = random.random()
        if random_number < 0.2:
            node_centrality = nx.degree_centrality(G_temp)
            node = max(node_centrality, key=node_centrality.get)
        else:
            node = random.choice(list(G_temp.nodes()))
        sequence.append(node)
        
        # 移除节点及其边
        if G_temp.has_node(node):
            edges = list(G_temp.edges(node))
            G_temp.remove_edges_from(edges)
            G_temp.remove_node(node)
            
    return sequence

# -------------------------------------------------------
# Main Simulation Logic
# -------------------------------------------------------

def get_dismantling_sequence(G, method):
    """
    根据指定的方法名称获取拆解序列。
    
    Args:
        G (nx.Graph): 网络图。
        method (str): 拆解方法名称。
        
    Returns:
        list: 拆解节点序列。
    """
    logger.info(f"Generating dismantling sequence using method: {method}")
    # 统一通过 network_dismantling 接口获取拆解序列
    method_map = {
        "bruteforce": "brute_force",
        "degree": "degree",
        "target": "degree",  # target 映射为自适应度拆解
        "random": "random",
        "pagerank": "pagerank",
        "betweenness": "betweenness",
        "eigenvector": "eigenvector",
        "hybrid": "adaptive_hybrid",
    }
    nd_method = method_map.get(method, method)
    try:
        return nd.dismantle(G, method=nd_method)
    except Exception as e:
        logger.warning(f"network_dismantling method '{nd_method}' failed: {e}, falling back to random.")
        return nd.dismantle(G, method="random")

def simulate_dismantling(G, sequence, nodes_num, centers):
    """
    模拟网络拆解过程。
    
    过程：
    1. 按照 sequence 顺序移除节点。
    2. 检查并移除级联失效的节点（即所在的连通分量中没有控制器的节点）。
    3. 计算当前网络的性能指标（GCC, CSA, CCE, WCP）。
    4. 记录指标随移除比例变化的曲线。
    
    Args:
        G (nx.Graph): 网络图（会被修改）。
        sequence (list): 节点移除序列。
        nodes_num (int): 初始节点总数（用于归一化）。
        centers (set): 控制器节点集合。
        
    Returns:
        tuple: (metrics_dict, x_when_gcc_20_percent, attack_steps_dict)
            - metrics_dict (dict): 包含所有指标的曲线数据
                - 'gcc': [(gcc_value, remove_ratio), ...]
                - 'csa': [(csa_value, remove_ratio), ...]
                - 'cce': [(cce_value, remove_ratio), ...]
                - 'wcp': [(wcp_value, remove_ratio), ...]
            - x_when_gcc_20_percent (float): GCC降到20%时的移除比例。
            - attack_steps_dict (dict): 包含每个指标对应的攻击步数
                - 'gcc': [(gcc_value, remove_ratio, attack_step), ...]
                - 'csa': [(csa_value, remove_ratio, attack_step), ...]
                - 'cce': [(cce_value, remove_ratio, attack_step), ...]
                - 'wcp': [(wcp_value, remove_ratio, attack_step), ...]
    """
    # 计算初始指标（在任何攻击之前）
    centers_list = list(centers)
    
    # 计算初始 GCC（包含控制器的最大连通分量）
    initial_gcc = calculate_gcc(G, centers_list)
    initial_gcc_normalized = initial_gcc / nodes_num if nodes_num > 0 else 0
    
    # 计算初始状态的sum(A_m)，用于固定CCE计算的分母
    # 计算每个控制器的服务域A_j
    centers_set = set(centers_list)
    components = list(nx.connected_components(G))
    A_map_initial = {}
    for comp in components:
        switches_in_comp = [n for n in comp if n not in centers_set]
        size_sw = len(switches_in_comp)
        centers_in_comp = [n for n in comp if n in centers_set]
        for c in centers_in_comp:
            A_map_initial[c] = size_sw
    A_list_initial = [A_map_initial.get(c, 0) for c in centers_list]
    initial_sum_A = sum(A_list_initial)
    
    # 计算初始交换节点总数，用于固定CSA计算的分母
    initial_switch_nodes = [n for n in G.nodes() if n not in centers_set]
    initial_num_switches = len(initial_switch_nodes)
    
    # 计算其他初始指标
    initial_csa = calculate_csa(G, centers_list, initial_num_switches=initial_num_switches)
    initial_cce = calculate_control_entropy(G, centers_list, initial_sum_A=initial_sum_A)
    initial_wcp = calculate_wcp(G, centers_list, initial_num_switches=initial_num_switches)
    
    # 初始化指标记录（使用实际计算的初始值）
    # 注意：'gcc' 表示 Giant Connected Component (极大连通子图)
    metrics_dict = {
        'gcc': [(initial_gcc_normalized, 0.0)],
        'csa': [(initial_csa, 0.0)],
        'cce': [(initial_cce, 0.0)],
        'wcp': [(initial_wcp, 0.0)]
    }
    
    # 初始化攻击步数记录（每个指标点对应的攻击步数）
    attack_steps_dict = {
        'gcc': [(initial_gcc_normalized, 0.0, 0)],  # (value, remove_ratio, attack_step)
        'csa': [(initial_csa, 0.0, 0)],
        'cce': [(initial_cce, 0.0, 0)],
        'wcp': [(initial_wcp, 0.0, 0)]
    }
    
    # 记录初始指标值到日志
    initial_edges = G.number_of_edges()
    logger.info(f"Initial metrics - GCC: {initial_gcc_normalized:.4f}, WCP: {initial_wcp:.4f}, CSA: {initial_csa:.4f}, CCE: {initial_cce:.4f}")
    
    x_when_gcc_20_percent = None
    attack_step = 0  # 攻击步数（从0开始，每次攻击+1，不管节点是否存在）
    
    # 记录上一次的指标值（用于节点已被移除时保持不变）
    last_metrics = {
        'gcc': initial_gcc_normalized,
        'csa': initial_csa,
        'cce': initial_cce,
        'wcp': initial_wcp
    }
    last_x_val = 0.0
    
    logger.debug(f"Starting simulation. Initial nodes: {G.number_of_nodes()}, Initial edges: {initial_edges}, Centers: {len(centers)}, Initial GCC: {initial_gcc_normalized:.4f}")
    
    # 遍历攻击序列
    for node in sequence:
        attack_step += 1  # 每次遍历都增加攻击步数
        
        # 如果节点已经不在图中（可能在之前的级联失效中被移除），记录上一次的值
        if not G.has_node(node):
            # 节点已被移除，但仍然记录这一步（y值保持不变）
            current_nodes = G.number_of_nodes()
            x_val = (nodes_num - current_nodes) / nodes_num
            
            metrics_dict['gcc'].append((last_metrics['gcc'], x_val))
            metrics_dict['csa'].append((last_metrics['csa'], x_val))
            metrics_dict['cce'].append((last_metrics['cce'], x_val))
            metrics_dict['wcp'].append((last_metrics['wcp'], x_val))
            
            attack_steps_dict['gcc'].append((last_metrics['gcc'], x_val, attack_step))
            attack_steps_dict['csa'].append((last_metrics['csa'], x_val, attack_step))
            attack_steps_dict['cce'].append((last_metrics['cce'], x_val, attack_step))
            attack_steps_dict['wcp'].append((last_metrics['wcp'], x_val, attack_step))
            continue
        
        # 1. 攻击移除
        edges = list(G.edges(node))
        G.remove_edges_from(edges)
        G.remove_node(node)
        
        # 判断被移除的节点是否是控制器
        if node in centers:
            centers.remove(node)
            logger.debug(f"Controller {node} removed by attack.")
            
        # 2. 级联失效逻辑 (Cascade Failure)
        # 关键修复：任何节点移除后都可能导致网络分裂，
        # 需要检查所有连通分量，移除没有控制器的分量
        if G.number_of_nodes() > 0 and len(centers) > 0:
            components = list(nx.connected_components(G))
            nodes_to_cascade_remove = []
            
            for comp in components:
                # 检查该分量是否包含控制器
                if centers.isdisjoint(comp):
                    # 该分量没有控制器，级联失效
                    nodes_to_cascade_remove.extend(list(comp))
            
            if nodes_to_cascade_remove:
                logger.debug(f"Cascade failure triggered. Removing {len(nodes_to_cascade_remove)} nodes from uncontrolled components.")
                for inode in nodes_to_cascade_remove:
                    if G.has_node(inode):
                        edges = list(G.edges(inode))
                        G.remove_edges_from(edges)
                        G.remove_node(inode)
                        # 注意：级联失效的节点不单独记录攻击步数

        # 3. 计算指标并记录
        current_nodes = G.number_of_nodes()
        x_val = (nodes_num - current_nodes) / nodes_num
        
        if current_nodes == 0:
            # 网络完全崩溃，记录零值
            metrics_dict['gcc'].append((0.0, x_val))
            metrics_dict['csa'].append((0.0, x_val))
            metrics_dict['cce'].append((0.0, x_val))
            metrics_dict['wcp'].append((0.0, x_val))
            # 记录攻击步数
            attack_steps_dict['gcc'].append((0.0, x_val, attack_step))
            attack_steps_dict['csa'].append((0.0, x_val, attack_step))
            attack_steps_dict['cce'].append((0.0, x_val, attack_step))
            attack_steps_dict['wcp'].append((0.0, x_val, attack_step))
            # 更新 last_metrics 为零
            last_metrics = {'gcc': 0.0, 'csa': 0.0, 'cce': 0.0, 'wcp': 0.0}
            last_x_val = x_val
            if x_when_gcc_20_percent is None:
                x_when_gcc_20_percent = x_val
            logger.debug("Network fully collapsed.")
            # 不再 break，继续记录后续步骤（y值保持为0）
        else:
            # 预先计算共同变量以避免重复计算
            centers_list = list(centers)
            components = list(nx.connected_components(G))
            centers_set = set(centers_list)
            
            # 计算GCC（极大连通子图占比）
            gcc_value = calculate_gcc(G, centers_list)
            gcc_normalized = gcc_value / nodes_num
            
            # 计算CSA（控制供给可用性），使用初始交换节点总数作为固定分母
            csa_value = calculate_csa(G, centers_list, initial_num_switches=initial_num_switches)
            
            # 计算CCE（控制熵），使用初始状态的固定分母
            cce_value = calculate_control_entropy(G, centers_list, initial_sum_A=initial_sum_A)
            
            # 计算WCP（加权控制势能），使用初始交换节点总数作为固定分母
            wcp_value = calculate_wcp(G, centers_list, initial_num_switches=initial_num_switches)
            
            # 直接记录指标（不再进行异常值检测）
            metrics_dict['gcc'].append((gcc_normalized, x_val))
            metrics_dict['csa'].append((csa_value, x_val))
            metrics_dict['cce'].append((cce_value, x_val))
            metrics_dict['wcp'].append((wcp_value, x_val))
            
            # 记录攻击步数（与指标值一起记录）
            attack_steps_dict['gcc'].append((gcc_normalized, x_val, attack_step))
            attack_steps_dict['csa'].append((csa_value, x_val, attack_step))
            attack_steps_dict['cce'].append((cce_value, x_val, attack_step))
            attack_steps_dict['wcp'].append((wcp_value, x_val, attack_step))
            
            # 更新 last_metrics
            last_metrics = {
                'gcc': gcc_normalized,
                'csa': csa_value,
                'cce': cce_value,
                'wcp': wcp_value
            }
            last_x_val = x_val
            
            # 记录关键指标变化到日志（每10%移除比例记录一次，或接近关键阈值时）
            if attack_step % max(1, nodes_num // 10) == 0 or abs(gcc_normalized - 0.2) < 0.05:
                current_edges = G.number_of_edges()
                logger.debug(f"Step={attack_step}, x={x_val:.4f}: Nodes={current_nodes}, Edges={current_edges}, GCC={gcc_normalized:.4f}, WCP={wcp_value:.4f}, CSA={csa_value:.4f}, CCE={cce_value:.4f}")
            
            # 记录GCC降到20%时的移除比例（但不停止攻击）
            if gcc_normalized <= 0.2 and x_when_gcc_20_percent is None:
                x_when_gcc_20_percent = x_val
                logger.debug(f"GCC dropped below 20% at x={x_val:.4f} (continuing attack)")
            
            # 继续攻击直到网络完全崩溃或序列结束
            # 不再提前break，让攻击继续进行
                
    return metrics_dict, x_when_gcc_20_percent, attack_steps_dict

def node_attack(G, nodes_num, centers, flag):
    """
    为了向后兼容的包装函数。
    1. 根据 flag 生成拆解序列。
    2. 运行仿真。
    """
    #1. 获取序列
    # 我们传递副本给 get_sequence，因为自适应方法会修改图
    sequence = get_dismantling_sequence(G.copy(), flag)
    
    logger.info(f"Starting node attack simulation. Mode: {flag}")
    
    #2. 运行仿真
    # 使用原始 G（仿真会修改它）和原始控制器集合（仿真会修改集合）
    centers_set = set(centers)
    metrics_dict, x_when_gcc_20_percent, attack_steps_dict = simulate_dismantling(G, sequence, nodes_num, centers_set)
    
    logger.info(f"Attack completed. Collapse point (GCC 20%): {x_when_gcc_20_percent}")
    
    return metrics_dict, x_when_gcc_20_percent, attack_steps_dict
