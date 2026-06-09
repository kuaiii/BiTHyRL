# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import networkx as nx
import numpy as np
import random
import os
from math import ceil
from tqdm import tqdm
import logging
from src.utils.logger import get_logger
from network_metrics.functional.controller import calculate_csa, calculate_cce as calculate_control_entropy, calculate_wcp

logger = get_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import K-Median helper from strategies (we will need to move it or import carefully to avoid circular deps)
# To avoid circular imports, I will re-implement a localized k_median_vectorized or similar here.

def k_median_vectorized_subset(G, points, num_to_select, pre_selected, is_weight=0):
    """
    在已选定 pre_selected 节点作为中心的情况下，从 points 中再选择 num_to_select 个节点，
    以最小化距离和。
    
    支持 GPU 加速计算
    """
    if num_to_select <= 0:
        return []
        
    nodes_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    num_nodes = len(nodes_list)
    
    points_list = list(points)
    # 从候选集中移除已选节点
    points_list = [p for p in points_list if p not in pre_selected]
    num_candidates = len(points_list)
    
    if num_candidates == 0:
        return []
    
    # 检查是否使用 GPU
    use_gpu = torch.cuda.is_available() and num_candidates > 50
        
    dist_matrix = np.full((num_candidates, num_nodes), np.inf, dtype=np.float32)
    
    # 预计算候选节点的距离
    for i, candidate in enumerate(points_list):
        if is_weight:
            dists = nx.single_source_dijkstra_path_length(G, candidate, weight='weight')
        else:
            dists = nx.single_source_shortest_path_length(G, candidate)
        
        for target_node, d in dists.items():
            if target_node in node_to_idx:
                dist_matrix[i, node_to_idx[target_node]] = d
                
    # 基于已选节点初始化当前最小距离
    current_min_dists = np.full(num_nodes, np.inf, dtype=np.float32)
    
    if pre_selected:
        for center in pre_selected:
            if is_weight:
                dists = nx.single_source_dijkstra_path_length(G, center, weight='weight')
            else:
                dists = nx.single_source_shortest_path_length(G, center)
            
            for target_node, d in dists.items():
                if target_node in node_to_idx:
                    idx = node_to_idx[target_node]
                    if d < current_min_dists[idx]:
                        current_min_dists[idx] = d
    
    if use_gpu:
        # GPU 加速版本
        dist_tensor = torch.tensor(dist_matrix, dtype=torch.float32, device=DEVICE)
        current_min_tensor = torch.tensor(current_min_dists, dtype=torch.float32, device=DEVICE)
        
        selected_indices = []
        remaining_mask = torch.ones(num_candidates, dtype=torch.bool, device=DEVICE)
        
        for _ in range(num_to_select):
            if not remaining_mask.any():
                break
            
            # 计算选择每个候选后的新最小距离
            new_dists = torch.minimum(dist_tensor, current_min_tensor.unsqueeze(0))
            totals = torch.sum(new_dists, dim=1)
            totals[~remaining_mask] = float('inf')
            
            best_idx = torch.argmin(totals).item()
            selected_indices.append(best_idx)
            remaining_mask[best_idx] = False
            current_min_tensor = torch.minimum(dist_tensor[best_idx], current_min_tensor)
    else:
        # CPU 版本
        selected_indices = []
        remaining_indices = set(range(num_candidates))
        
        for _ in range(num_to_select):
            if not remaining_indices:
                break

            rem_indices_list = list(remaining_indices)
            candidate_dists = dist_matrix[rem_indices_list] 
            new_dists = np.minimum(candidate_dists, current_min_dists)
            total_dists = np.sum(new_dists, axis=1)
            
            min_idx_in_rem = np.argmin(total_dists)
            best_candidate_idx = rem_indices_list[min_idx_in_rem]
            
            selected_indices.append(best_candidate_idx)
            remaining_indices.remove(best_candidate_idx)
            current_min_dists = np.minimum(current_min_dists, dist_matrix[best_candidate_idx])
        
    return [points_list[i] for i in selected_indices]


# Context-Free RL Policy (MLP)
class MLPPolicy(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super(MLPPolicy, self).__init__()
        # 输入：节点特征 (度, 聚类系数, 介数)
        # 更深的网络：4个隐藏层
        self.fc1 = torch.nn.Linear(num_features, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.actor = torch.nn.Linear(hidden_dim, 1) # 输出每个节点的得分

    def forward(self, x, mask=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        # 获取每个节点的得分
        scores = self.actor(x).squeeze(-1)
        
        # 屏蔽已选节点或无效节点
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
            
        # Softmax 获取选择概率分布
        probs = F.softmax(scores, dim=0)
        return probs

def get_node_features(G):
    """
    Extract node features using PyTorch for GPU acceleration where possible.
    Replaces slow NetworkX centrality measures with spectral approximations.
    
    Features:
    1. Degree Centrality (Normalized)
    2. Clustering Coefficient (Approximated via Matrix Multiplication)
    3. PageRank (Power Iteration)
    4. Eigenvector Centrality (Power Iteration)
    5. Katz Centrality (Approximated)
    """
    num_nodes = G.number_of_nodes()
    if num_nodes == 0:
        return torch.zeros((0, 5)), []
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get Adjacency Matrix
    adj = nx.to_scipy_sparse_array(G, format='coo', dtype=np.float32)
    indices = torch.LongTensor(np.vstack((adj.row, adj.col))).to(device)
    values = torch.FloatTensor(adj.data).to(device)
    adj_tensor = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).to(device)
    
    # Convert to dense for some operations (if graph is small enough)
    # For large graphs, we should stick to sparse, but here max nodes ~1000 usually.
    if num_nodes < 5000:
        adj_dense = adj_tensor.to_dense()
    else:
        # Fallback to sparse or CPU if too large
        adj_dense = None

    # 1. Degree Centrality
    degrees = torch.sparse.sum(adj_tensor, dim=1).to_dense()
    deg_norm = degrees / (num_nodes - 1 + 1e-6)
    
    # 2. Clustering Coefficient (Approximation: triangles / potential_triangles)
    # C_i = (A^3)_ii / (d_i * (d_i - 1))
    # This requires dense matrix multiplication A @ A @ A
    if adj_dense is not None:
        adj2 = torch.mm(adj_dense, adj_dense)
        adj3 = torch.mm(adj2, adj_dense)
        triangles = torch.diagonal(adj3)
        potential = degrees * (degrees - 1)
        clust = triangles / (potential + 1e-6)
    else:
        # Fallback to NetworkX for large graphs (though slower) or just use 0
        # For compatibility with small graphs in this project, we use NX if dense is impossible
        clust_nx = nx.clustering(G)
        clust = torch.tensor([clust_nx[n] for n in G.nodes()], dtype=torch.float32).to(device)

    # 3. PageRank (Power Iteration)
    # M = (A * D^-1) -> but typically PR uses transition matrix
    # PR formula: x = alpha * A_norm * x + (1-alpha) * e/N
    # A_norm = A * D_inv
    if num_nodes > 0:
        deg_inv = 1.0 / (degrees + 1e-6)
        # Create sparse transition matrix A * D_inv
        # We can just multiply values by D_inv[col]
        # COO format: values[k] corresponds to edge (i, j). We want A_ij / d_j.
        # indices[1] is column index j.
        norm_values = values * deg_inv[indices[1]]
        transition_matrix = torch.sparse_coo_tensor(indices, norm_values, (num_nodes, num_nodes)).to(device)
        
        pr = torch.ones(num_nodes, 1).to(device) / num_nodes
        alpha = 0.85
        for _ in range(10): # 10 iterations usually enough for feature approximation
            pr = alpha * torch.sparse.mm(transition_matrix, pr) + (1 - alpha) / num_nodes
        pr = pr.squeeze()
    else:
        pr = torch.zeros(num_nodes).to(device)

    # 4. Eigenvector Centrality (Power Iteration)
    # x = A * x / |Ax|
    eig = torch.ones(num_nodes, 1).to(device)
    for _ in range(10):
        eig = torch.sparse.mm(adj_tensor, eig)
        norm = torch.norm(eig)
        if norm > 0:
            eig = eig / norm
    eig = eig.squeeze()

    # 5. Katz Centrality (Approximation)
    # x = (I - alpha * A)^-1 * beta
    # Can use power series: x = beta * (I + alpha A + alpha^2 A^2 + ...)
    # Let's just use a simple 2-step walk count as a proxy for Closeness/Katz
    # Katz ~ alpha * A * 1 + alpha^2 * A^2 * 1 ...
    # Let's use A^2 sum (2-hop reachability)
    if adj_dense is not None:
        katz = torch.sum(torch.mm(adj_dense, adj_dense), dim=1)
        katz = katz / (torch.max(katz) + 1e-6)
    else:
        # Fallback
        katz = degrees / (torch.max(degrees) + 1e-6)

    # Stack features
    # Ensure all are 1D tensors of shape (N,)
    # If any feature is a scalar (e.g. from 0-dim tensor), expand it
    
    # Debug shapes if needed
    # print(f"Shapes: deg={deg_norm.shape}, clust={clust.shape}, pr={pr.shape}, eig={eig.shape}, katz={katz.shape}")
    
    # Ensure consistent shapes
    if pr.dim() == 0: pr = pr.unsqueeze(0)
    if eig.dim() == 0: eig = eig.unsqueeze(0)
    
    # If graph has 1 node, squeeze() might have reduced them to 0-dim scalars
    if num_nodes == 1:
        if deg_norm.dim() == 0: deg_norm = deg_norm.unsqueeze(0)
        if clust.dim() == 0: clust = clust.unsqueeze(0)
        if pr.dim() == 0: pr = pr.unsqueeze(0)
        if eig.dim() == 0: eig = eig.unsqueeze(0)
        if katz.dim() == 0: katz = katz.unsqueeze(0)

    features = torch.stack([deg_norm, clust, pr, eig, katz], dim=1).cpu()
    
    return features, list(G.nodes())

def calculate_reward(G, centers, mode='combined'):
    """
    计算综合奖励（增强版 - 级联失效感知）
    
    改进：
    1. 模拟攻击后考虑级联失效（不含控制器的分量会失效）
    2. 增加对控制器分布的评估（覆盖多个连通分量）
    3. 惩罚控制器过度集中在单一区域
    
    Args:
        G (nx.Graph): 网络拓扑
        centers (list): 控制器节点列表
        mode (str): 优化目标模式 ('combined', 'robustness', 'csa', 'entropy', 'wcp')
    """
    if not centers:
        return 0.0
    
    centers_set = set(centers)
    total_nodes = G.number_of_nodes()
    
    # 1. Robustness with Cascade Failure Awareness
    if mode in ['combined', 'robustness']:
        G_copy = G.copy()
        num_remove = max(1, int(total_nodes * 0.1))
        degrees = dict(G_copy.degree())
        targets = sorted(degrees, key=degrees.get, reverse=True)[:num_remove]
        
        # 移除高度节点（模拟攻击）
        remaining_centers = centers_set.copy()
        for node in targets:
            if node in G_copy:
                G_copy.remove_node(node)
                remaining_centers.discard(node)
        
        # 级联失效：计算含控制器的节点数
        if G_copy.number_of_nodes() > 0 and remaining_centers:
            components = list(nx.connected_components(G_copy))
            controlled_nodes = 0
            for comp in components:
                # 只有包含控制器的分量是"存活"的
                if not remaining_centers.isdisjoint(comp):
                    controlled_nodes += len(comp)
            r_val = controlled_nodes / total_nodes
        else:
            r_val = 0.0
    else:
        r_val = 0.0
    
    # 2. CSA (Control Supply Availability)
    if mode in ['combined', 'csa']:
        csa_val = calculate_csa(G, centers)
    else:
        csa_val = 0.0
    
    # 3. H_C (Control Entropy) - 衡量控制器负载均衡
    if mode in ['combined', 'entropy']:
        hc_val = calculate_control_entropy(G, centers)
    else:
        hc_val = 0.0
    
    # 4. WCP (Weighted Control Potential)
    if mode in ['combined', 'wcp']:
        wcp_val = calculate_wcp(G, centers)
    else:
        wcp_val = 0.0
    
    # 5. 新增：控制器分布奖励（覆盖尽可能多的连通分量）
    if mode in ['combined', 'robustness']:
        components = list(nx.connected_components(G))
        num_controlled_components = sum(1 for comp in components 
                                        if not centers_set.isdisjoint(comp))
        coverage_ratio = num_controlled_components / max(1, len(components))
    else:
        coverage_ratio = 1.0
    
    # 6. 新增：控制器间距奖励（避免控制器过度集中）
    if mode in ['combined', 'robustness'] and len(centers) > 1:
        try:
            min_dist = float('inf')
            for i, c1 in enumerate(centers):
                for c2 in centers[i+1:]:
                    if G.has_node(c1) and G.has_node(c2) and nx.has_path(G, c1, c2):
                        dist = nx.shortest_path_length(G, c1, c2)
                        min_dist = min(min_dist, dist)
            # 归一化：最小距离至少为1，理想情况下应该分散
            if min_dist != float('inf'):
                dispersion_bonus = min(min_dist / 3.0, 1.0)  # 距离>=3时满分
            else:
                dispersion_bonus = 0.5  # 不连通时给中等分数
        except:
            dispersion_bonus = 0.5
    else:
        dispersion_bonus = 1.0
    
    # 综合奖励计算
    if mode == 'robustness':
        # 增强版：考虑覆盖率和分散度
        return r_val * 0.6 + coverage_ratio * 0.25 + dispersion_bonus * 0.15
    elif mode == 'csa':
        return csa_val
    elif mode == 'entropy':
        return hc_val
    elif mode == 'wcp':
        return wcp_val
    else:  # combined
        # 权重设置（考虑级联失效感知）
        # R: 主要指标 -> w=2.0
        # Coverage: 连通分量覆盖 -> w=0.5
        # Dispersion: 控制器分散度 -> w=0.3
        # CSA: 控制供给可用性 -> w=1.0
        # H_C: 控制熵 -> w=0.2
        # WCP: 加权控制势能 -> w=1.0
        base_reward = 2.0 * r_val + 1.0 * csa_val + 0.2 * hc_val + 1.0 * wcp_val
        cascade_bonus = 0.5 * coverage_ratio + 0.3 * dispersion_bonus
        return base_reward + cascade_bonus

_global_model = None

def load_or_train_model(model_path='src/train/v1/checkpoints/rl_agent.pth'):
    # 注意：这个函数现在主要用于加载 combined 模型作为默认
    # 如果需要加载特定模型，应该在外部指定 model_path
    
    # 这里我们不做全局单例缓存了，因为可能会加载不同的模型
    # 或者我们使用一个 dict 来缓存不同路径的模型
    
    model = MLPPolicy(num_features=5, hidden_dim=64, output_dim=1)
    if os.path.exists(model_path):
        try:
            ck = torch.load(model_path, map_location=DEVICE, weights_only=True)
            state = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
            model.load_state_dict(state, strict=False)
        except Exception:
            pass
            
    return model

def predict(G, k, model=None, model_path=None):
    if model is None:
        if model_path is None:
            model_path = 'src/train/v1/checkpoints/rl_agent.pth'
        model = load_or_train_model(model_path)
    
    # 将模型移动到 GPU
    model = model.to(DEVICE)
    model.eval()
    
    x, node_list = get_node_features(G)
    if len(node_list) == 0:
        return []
    
    # 将输入数据移动到 GPU
    x = x.to(DEVICE)
        
    selected_indices = []
    mask = torch.zeros(len(node_list), dtype=torch.bool, device=DEVICE)
    
    with torch.no_grad():
        for _ in range(k):
            probs = model(x, mask)
            action = torch.argmax(probs)
            
            selected_indices.append(action.item())
            mask[action.item()] = True
            
    best_centers = [node_list[i] for i in selected_indices]
    return best_centers

def train_and_select(G, k, episodes=50, metric_type='combined'):
    """
    增强版控制器选择：结合预训练模型和在线微调
    
    改进：
    1. 更积极的在线微调（episodes >= 60 时启用）
    2. 多轮采样取最优解
    3. 结合贪婪和随机探索
    
    Args:
        G: 网络拓扑
        k: 控制器数量
        episodes: 训练轮数
        metric_type: 优化目标类型
    """
    # 根据 metric_type 选择模型
    model_name = 'rl_agent.pth'  # default combined
    if metric_type != 'combined':
        model_name = f'rl_agent_{metric_type}.pth'
        
    model_path = os.path.join('models', model_name)
    
    # 检查是否应使用混合方法
    if os.path.exists(model_path):
        # 混合方法实现
        # 降低在线微调阈值：episodes >= 60 时启用
        if episodes >= 60:
            
            # 多轮采样策略
            best_centers = None
            best_reward = -float('inf')
            
            # 第1轮：使用预训练模型
            centers_pretrained = hybrid_rl_select(G, k, model_path=model_path)
            reward_pretrained = calculate_reward(G, centers_pretrained, mode=metric_type)
            if reward_pretrained > best_reward:
                best_reward = reward_pretrained
                best_centers = centers_pretrained
            
            # 第2轮：在线微调
            centers_finetuned = train_offline_single(
                G, k, episodes, 
                metric_type=metric_type, 
                initial_centers=centers_pretrained
            )
            if centers_finetuned:
                reward_finetuned = calculate_reward(G, centers_finetuned, mode=metric_type)
                if reward_finetuned > best_reward:
                    best_reward = reward_finetuned
                    best_centers = centers_finetuned
            
            # 第3轮：如果episodes足够多，尝试纯在线训练（更多探索）
            if episodes >= 100:
                centers_fresh = train_offline_single(G, k, episodes // 2, metric_type=metric_type)
                if centers_fresh:
                    reward_fresh = calculate_reward(G, centers_fresh, mode=metric_type)
                    if reward_fresh > best_reward:
                        best_reward = reward_fresh
                        best_centers = centers_fresh
            
            return best_centers if best_centers else centers_pretrained
        
        return hybrid_rl_select(G, k, model_path=model_path)
    else:
        # 在线训练的回退方案
        if metric_type != 'combined' and os.path.exists('src/train/v1/checkpoints/rl_agent.pth'):
            if episodes >= 60:
                centers = hybrid_rl_select(G, k, model_path='src/train/v1/checkpoints/rl_agent.pth')
                fine_tuned_centers = train_offline_single(
                    G, k, episodes, 
                    metric_type=metric_type, 
                    initial_centers=centers
                )
                return fine_tuned_centers if fine_tuned_centers else centers
            return hybrid_rl_select(G, k, model_path='src/train/v1/checkpoints/rl_agent.pth')
        
        # 纯在线训练
        return train_offline_single(G, k, max(episodes, 80), metric_type=metric_type)

def hybrid_rl_select(G, total_controllers_budget, model_path='src/train/v1/checkpoints/rl_agent.pth'):
    """
    实现混合 RL-RCP 选择逻辑：
    1. 划分为社区。
    2. 分配预算。
    3. 对每个社区：
       - 使用 RL 模型选择第 1 个节点。
       - 使用 K-Median 选择剩余节点（给定第 1 个节点）。
    """
    communities = list(nx.connected_components(G))
    
    # 过滤空社区以防万一
    communities = [list(c) for c in communities if len(c) > 0]
    if not communities:
        return []

    # 按大小排序社区以优先考虑大社区
    communities.sort(key=len, reverse=True)
    
    allocated_count = 0
    allocations = []
    
    # 1. 最小分配：如果预算允许，每个分量至少分配 1 个
    for i in range(len(communities)):
        if allocated_count < total_controllers_budget:
            allocations.append(1)
            allocated_count += 1
        else:
            allocations.append(0)
            
    # 2. 剩余预算的按比例分配
    remaining_budget = total_controllers_budget - allocated_count
    
    if remaining_budget > 0:
        total_nodes = G.number_of_nodes()
        for i, nodes in enumerate(communities):
            if remaining_budget <= 0: break
            
            # 简单的比例份额
            share = int((len(nodes) / total_nodes) * total_controllers_budget)
            # 减去已经分配的 (1)
            extra = max(0, share - allocations[i])
            
            take = min(remaining_budget, extra)
            allocations[i] += take
            remaining_budget -= take
            
    # 3. 如果仍有预算剩余（由于取整），分配给最大的社区
    if remaining_budget > 0:
        for i in range(len(communities)):
            if remaining_budget <= 0: break
            # 分配数不能超过社区大小
            if allocations[i] < len(communities[i]):
                allocations[i] += 1
                remaining_budget -= 1

    final_controllers = []
    
    model = load_or_train_model(model_path)
    
    # 对每个社区执行选择
    for i, nodes in enumerate(communities):
        num_controllers = allocations[i]
        if num_controllers > 0:
            subgraph = G.subgraph(nodes).copy()
            
            # 1. 使用 RL 选择第一个控制器
            rl_controllers = predict(subgraph, 1, model)
            final_controllers.extend(rl_controllers)
            
            # 2. 使用 K-Median 选择剩余控制器（如果需要）
            remaining_needed = num_controllers - 1
            if remaining_needed > 0:
                # 使用贪婪/K-Median 方法选择最佳的额外节点
                # 考虑到 RL 节点已被选中
                additional = k_median_vectorized_subset(subgraph, nodes, remaining_needed, rl_controllers, is_weight=0)
                final_controllers.extend(additional)
                
    return final_controllers

def evaluate(model, graphs, mode='combined'):
    model = model.to(DEVICE)
    model.eval()
    total_reward = 0
    with torch.no_grad():
        for G in graphs:
            if G.number_of_nodes() < 2: continue
            x, node_list = get_node_features(G)
            x = x.to(DEVICE)
            # 用于评估的贪婪选择
            k = 1
            mask = torch.zeros(len(node_list), dtype=torch.bool, device=DEVICE)
            selected_indices = []
            for _ in range(k):
                probs = model(x, mask)
                action = torch.argmax(probs)
                selected_indices.append(action.item())
                mask[action.item()] = True
            centers = [node_list[i] for i in selected_indices]
            reward = calculate_reward(G, centers, mode=mode)
            total_reward += reward
    return total_reward / len(graphs) if graphs else 0

def train_offline(graphs_list, epochs=100, save_path='src/train/v1/checkpoints/rl_agent.pth', mode='combined'):
    # 分割数据集：80% 训练，20% 测试
    random.shuffle(graphs_list)
    split_idx = int(len(graphs_list) * 0.8)
    train_graphs = graphs_list[:split_idx]
    test_graphs = graphs_list[split_idx:]
    
    
    # 将 hidden_dim 增加到 64
    # num_features 更新为 5
    model = MLPPolicy(num_features=5, hidden_dim=64, output_dim=1)
    model = model.to(DEVICE)  # 移动模型到 GPU
    
    # 降低学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    
    history = []
    
    # 梯度累积步数 (模拟 Batch Size = 32)
    accumulation_steps = 32
    
    # Baseline (Moving Average Reward) 用于降低方差
    moving_avg_reward = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_reward = 0
        random.shuffle(train_graphs)
        
        optimizer.zero_grad()
        
        for i, G in enumerate(train_graphs):
            if G.number_of_nodes() < 2: continue
            
            x, node_list = get_node_features(G)
            x = x.to(DEVICE)  # 移动特征到 GPU
            k = 1 
            
            log_probs = []
            selected_indices = []
            mask = torch.zeros(len(node_list), dtype=torch.bool, device=DEVICE)
            
            for _ in range(k):
                probs = model(x, mask)
                m = Categorical(probs)
                action = m.sample()
                
                selected_indices.append(action.item())
                log_probs.append(m.log_prob(action))
                
                mask = mask.clone()
                mask[action.item()] = True
                
            centers = [node_list[i] for i in selected_indices]
            # 传递 mode 参数
            reward = calculate_reward(G, centers, mode=mode)
            train_reward += reward
            
            # Update moving average
            if moving_avg_reward == 0.0:
                moving_avg_reward = reward
            else:
                moving_avg_reward = 0.95 * moving_avg_reward + 0.05 * reward
                
            # Advantage = Reward - Baseline
            advantage = reward - moving_avg_reward
            
            loss = []
            for log_prob in log_probs:
                loss.append(-log_prob * advantage) # 使用 Advantage 代替 Raw Reward
            
            if loss:
                batch_loss = sum(loss) / accumulation_steps # Normalize loss by accumulation steps
                batch_loss.backward()
                
            # Step optimizer every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
                optimizer.step()
                optimizer.zero_grad()
        
        # Handle remaining gradients
        if len(train_graphs) % accumulation_steps != 0:
             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
             optimizer.step()
             optimizer.zero_grad()
                
        avg_train_reward = train_reward / len(train_graphs) if train_graphs else 0
        avg_test_reward = evaluate(model, test_graphs, mode=mode)
        
        history.append([epoch+1, avg_train_reward, avg_test_reward])
        
    # 保存训练历史
    import csv
    history_path = os.path.join(os.path.dirname(save_path), f"training_history_{mode}.csv")
    if not os.path.exists(os.path.dirname(history_path)):
        os.makedirs(os.path.dirname(history_path))
        
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Reward', 'Test_Reward'])
        writer.writerows(history)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)

def train_offline_single(G, k, episodes, metric_type='combined', initial_centers=None):
    x, node_list = get_node_features(G)
    x = x.to(DEVICE)  # 移动特征到 GPU
    
    # num_features 更新为 5
    model = MLPPolicy(num_features=5, hidden_dim=64, output_dim=1)
    model = model.to(DEVICE)  # 移动模型到 GPU
    
    # 如果提供了初始控制器，使用它们作为起点
    if initial_centers:
        # 计算初始奖励
        initial_reward = calculate_reward(G, initial_centers, mode=metric_type)
        best_reward = initial_reward
        best_centers = initial_centers.copy()
    else:
        best_reward = -1.0
        best_centers = []
    
    # 根据episodes数量调整学习率：更多episodes时使用更小的学习率
    lr = 0.005 if episodes > 50 else 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for episode in range(episodes):
        selected_indices = []
        log_probs = []
        mask = torch.zeros(len(node_list), dtype=torch.bool, device=DEVICE)
        for _ in range(k):
            probs = model(x, mask)
            m = Categorical(probs)
            action = m.sample()
            selected_indices.append(action.item())
            log_probs.append(m.log_prob(action))
            mask = mask.clone()
            mask[action.item()] = True
        centers = [node_list[i] for i in selected_indices]
        reward = calculate_reward(G, centers, mode=metric_type)
        if reward > best_reward:
            best_reward = reward
            best_centers = centers
        loss = []
        for log_prob in log_probs:
            loss.append(-log_prob * reward)
        optimizer.zero_grad()
        if loss:
            sum(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
    if not best_centers and k > 0:
        best_centers = random.sample(list(G.nodes()), k)
    return best_centers
