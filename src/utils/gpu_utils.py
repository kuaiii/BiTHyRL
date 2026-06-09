# -*- coding: utf-8 -*-
"""
GPU 加速工具模块
使用 PyTorch 进行 GPU 加速计算
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional

# 尝试导入 PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
    # 检测 CUDA 是否可用
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        # 设置默认设备为 GPU
        DEVICE = torch.device('cuda')
        # 获取 GPU 信息
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"[GPU] PyTorch CUDA 可用: {GPU_NAME} ({GPU_MEMORY:.1f} GB)")
    else:
        DEVICE = torch.device('cpu')
        print("[GPU] CUDA 不可用，使用 CPU")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    DEVICE = None
    print("[GPU] PyTorch 未安装，使用 NumPy")


def get_device():
    """获取当前设备"""
    return DEVICE


def is_gpu_available():
    """检查 GPU 是否可用"""
    return TORCH_AVAILABLE and CUDA_AVAILABLE


def to_tensor(data, dtype=None):
    """将数据转换为 PyTorch 张量并移动到 GPU"""
    if not TORCH_AVAILABLE:
        return np.array(data)
    
    if isinstance(data, torch.Tensor):
        return data.to(DEVICE)
    
    if dtype is None:
        dtype = torch.float32
    
    return torch.tensor(data, dtype=dtype, device=DEVICE)


def to_numpy(tensor):
    """将 PyTorch 张量转换回 NumPy 数组"""
    if not TORCH_AVAILABLE:
        return np.array(tensor)
    
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return np.array(tensor)


def graph_to_adjacency_tensor(G: nx.Graph) -> 'torch.Tensor':
    """
    将 NetworkX 图转换为 GPU 上的邻接矩阵张量
    
    Args:
        G: NetworkX 图对象
        
    Returns:
        torch.Tensor: 邻接矩阵张量 (在 GPU 上)
    """
    if not TORCH_AVAILABLE:
        return nx.to_numpy_array(G)
    
    # 获取邻接矩阵
    adj_matrix = nx.to_numpy_array(G)
    return torch.tensor(adj_matrix, dtype=torch.float32, device=DEVICE)


def compute_shortest_paths_gpu(adj_matrix: 'torch.Tensor', max_iter: int = 100) -> 'torch.Tensor':
    """
    使用 Floyd-Warshall 算法在 GPU 上计算所有节点对的最短路径
    
    Args:
        adj_matrix: 邻接矩阵张量 (权重表示边的存在, 0 表示无边)
        max_iter: 最大迭代次数
        
    Returns:
        torch.Tensor: 最短路径距离矩阵
    """
    if not TORCH_AVAILABLE:
        return adj_matrix
    
    n = adj_matrix.shape[0]
    
    # 初始化距离矩阵
    # 有边的地方使用权重，无边的地方使用 inf
    dist = torch.where(adj_matrix > 0, adj_matrix, torch.tensor(float('inf'), device=DEVICE))
    
    # 对角线设为 0
    dist.fill_diagonal_(0)
    
    # Floyd-Warshall 算法
    for k in range(min(n, max_iter)):
        # 使用广播进行并行计算
        dist_k = dist[:, k:k+1] + dist[k:k+1, :]
        dist = torch.minimum(dist, dist_k)
    
    return dist


def compute_shortest_paths_from_sources_gpu(G: nx.Graph, sources: List[int]) -> np.ndarray:
    """
    使用 GPU 加速计算从多个源节点到所有节点的最短路径
    
    这个函数结合了 NetworkX 的路径计算和 GPU 的并行处理能力
    
    Args:
        G: NetworkX 图对象
        sources: 源节点列表
        
    Returns:
        np.ndarray: 距离矩阵 (sources x all_nodes)
    """
    if not TORCH_AVAILABLE or not CUDA_AVAILABLE:
        # 回退到 NumPy 实现
        nodes_list = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes_list)}
        num_nodes = len(nodes_list)
        num_sources = len(sources)
        
        dist_matrix = np.full((num_sources, num_nodes), np.inf)
        
        for i, source in enumerate(sources):
            try:
                lengths = nx.single_source_shortest_path_length(G, source)
                for target, d in lengths.items():
                    if target in node_to_idx:
                        dist_matrix[i, node_to_idx[target]] = d
            except nx.NetworkXError:
                pass
        
        return dist_matrix
    
    # GPU 实现
    nodes_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    num_nodes = len(nodes_list)
    num_sources = len(sources)
    
    # 先在 CPU 上计算距离 (NetworkX 不支持 GPU)
    dist_list = []
    for source in sources:
        row = np.full(num_nodes, np.inf)
        try:
            lengths = nx.single_source_shortest_path_length(G, source)
            for target, d in lengths.items():
                if target in node_to_idx:
                    row[node_to_idx[target]] = d
        except nx.NetworkXError:
            pass
        dist_list.append(row)
    
    # 转移到 GPU 进行后续处理
    dist_matrix = torch.tensor(np.array(dist_list), dtype=torch.float32, device=DEVICE)
    
    return to_numpy(dist_matrix)


def parallel_metric_computation(G: nx.Graph, centers: List[int], 
                                 metric_func: callable, 
                                 batch_size: int = 100) -> float:
    """
    并行计算网络指标
    
    Args:
        G: NetworkX 图对象
        centers: 控制器节点列表
        metric_func: 指标计算函数
        batch_size: 批处理大小
        
    Returns:
        float: 计算的指标值
    """
    return metric_func(G, centers)


def gpu_distance_sum(dist_matrix: np.ndarray) -> float:
    """
    使用 GPU 计算距离矩阵的总和
    
    Args:
        dist_matrix: 距离矩阵
        
    Returns:
        float: 距离总和
    """
    if not TORCH_AVAILABLE or not CUDA_AVAILABLE:
        return np.sum(dist_matrix[np.isfinite(dist_matrix)])
    
    tensor = torch.tensor(dist_matrix, dtype=torch.float32, device=DEVICE)
    # 将 inf 替换为 0 以避免计算问题
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor.sum().item()


def gpu_min_distances(dist_matrix: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    使用 GPU 计算距离矩阵沿某个轴的最小值
    
    Args:
        dist_matrix: 距离矩阵
        axis: 计算轴
        
    Returns:
        np.ndarray: 最小距离数组
    """
    if not TORCH_AVAILABLE or not CUDA_AVAILABLE:
        return np.min(dist_matrix, axis=axis)
    
    tensor = torch.tensor(dist_matrix, dtype=torch.float32, device=DEVICE)
    min_vals, _ = torch.min(tensor, dim=axis)
    return to_numpy(min_vals)


def gpu_greedy_selection(dist_matrix: np.ndarray, num_select: int) -> List[int]:
    """
    使用 GPU 加速的贪婪选择算法
    选择能最小化总距离的节点子集
    
    Args:
        dist_matrix: 距离矩阵 (candidates x all_nodes)
        num_select: 要选择的节点数量
        
    Returns:
        List[int]: 选中节点的索引
    """
    if not TORCH_AVAILABLE or not CUDA_AVAILABLE:
        # NumPy 实现
        num_candidates, num_nodes = dist_matrix.shape
        current_min_dists = np.full(num_nodes, np.inf)
        selected = []
        remaining = set(range(num_candidates))
        
        for _ in range(num_select):
            if not remaining:
                break
            
            best_idx = -1
            best_total = np.inf
            
            for idx in remaining:
                new_dists = np.minimum(dist_matrix[idx], current_min_dists)
                total = np.sum(new_dists)
                if total < best_total:
                    best_total = total
                    best_idx = idx
            
            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)
                current_min_dists = np.minimum(dist_matrix[best_idx], current_min_dists)
        
        return selected
    
    # GPU 实现
    num_candidates, num_nodes = dist_matrix.shape
    dist_tensor = torch.tensor(dist_matrix, dtype=torch.float32, device=DEVICE)
    
    current_min_dists = torch.full((num_nodes,), float('inf'), device=DEVICE)
    selected = []
    remaining_mask = torch.ones(num_candidates, dtype=torch.bool, device=DEVICE)
    
    for _ in range(num_select):
        if not remaining_mask.any():
            break
        
        # 计算选择每个候选后的新最小距离
        # 广播计算: new_dists[i, j] = min(dist_matrix[i, j], current_min_dists[j])
        new_dists = torch.minimum(dist_tensor, current_min_dists.unsqueeze(0))
        
        # 计算每个候选的总距离
        totals = torch.sum(new_dists, dim=1)
        
        # 将已选择的设为 inf
        totals[~remaining_mask] = float('inf')
        
        # 选择最小的
        best_idx = torch.argmin(totals).item()
        
        selected.append(best_idx)
        remaining_mask[best_idx] = False
        current_min_dists = torch.minimum(dist_tensor[best_idx], current_min_dists)
    
    return selected


def cleanup_gpu():
    """清理 GPU 内存"""
    if TORCH_AVAILABLE and CUDA_AVAILABLE:
        torch.cuda.empty_cache()


# 批量计算工具函数
def batch_compute_distances(G: nx.Graph, node_pairs: List[Tuple[int, int]], 
                            batch_size: int = 1000) -> Dict[Tuple[int, int], float]:
    """
    批量计算节点对之间的距离
    
    Args:
        G: NetworkX 图对象
        node_pairs: 节点对列表
        batch_size: 批处理大小
        
    Returns:
        Dict: 节点对到距离的映射
    """
    result = {}
    
    # 预计算所有需要的源节点的最短路径
    sources = set(pair[0] for pair in node_pairs)
    source_paths = {}
    
    for source in sources:
        try:
            source_paths[source] = dict(nx.single_source_shortest_path_length(G, source))
        except nx.NetworkXError:
            source_paths[source] = {}
    
    # 查询距离
    for src, dst in node_pairs:
        if src in source_paths and dst in source_paths[src]:
            result[(src, dst)] = source_paths[src][dst]
        else:
            result[(src, dst)] = float('inf')
    
    return result
