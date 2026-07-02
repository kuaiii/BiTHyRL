# -*- coding: utf-8 -*-
"""BiT-HyRL 节点特征（Node2Vec + 统计特征 + 自适应增强）。"""
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx

from src.utils.logger import get_logger
from . import config

logger = get_logger(__name__)
_node2vec_model_cache = {}

# ============================================================
# 网络规模阈值配置
# ============================================================
SMALL_NETWORK_THRESHOLD = 50      # 小于此值为小规模网络
MEDIUM_NETWORK_THRESHOLD = 200    # 小于此值为中等规模网络


def get_node2vec_embeddings(G, dimensions=64, walk_length=10, num_walks=50, p=1, q=1, workers=1, seed=42):
    """
    使用 Node2Vec 生成节点嵌入。
    
    Args:
        G: 原始图
        dimensions: 嵌入维度
        walk_length: 随机游走长度
        num_walks: 随机游走次数
        p, q: Node2Vec 参数
        workers: 并行工作进程数
        
    Returns:
        np.ndarray: 节点嵌入矩阵，形状 (num_nodes, dimensions)
        如果失败返回 None
    """
    if not config.NODE2VEC_AVAILABLE or config.Node2Vec is None:
        logger.warning("Node2Vec not available, using fallback features")
        return None
    
    # 保存原始节点列表（用于最终返回）
    original_nodes = list(G.nodes())
    num_original_nodes = len(original_nodes)
    
    # 预检查：图太小时跳过 Node2Vec
    if num_original_nodes < 3:
        logger.debug(f"Graph too small ({num_original_nodes} nodes), skipping Node2Vec")
        return None
    
    if G.number_of_edges() == 0:
        logger.debug("Graph has no edges, skipping Node2Vec")
        return None
    
    # 创建工作副本进行预处理
    G_work = G.copy()
    
    # 检查并清理边权重（必须在移除节点前做，否则可能丢失边信息）
    for u, v, data in G_work.edges(data=True):
        if 'weight' in data:
            w = data['weight']
            if not np.isfinite(w) or w <= 0:
                data['weight'] = 1.0
    
    # 检查是否有孤立节点
    isolates = list(nx.isolates(G_work))
    if isolates:
        logger.debug(f"Graph has {len(isolates)} isolated nodes, removing for Node2Vec training")
        G_work.remove_nodes_from(isolates)
    
    # 检查处理后的图是否足够大
    if G_work.number_of_nodes() < 3 or G_work.number_of_edges() == 0:
        logger.debug("Graph too small after removing isolates, skipping Node2Vec")
        return None
    
    # 如果图不连通，只使用最大连通分量训练
    trained_nodes = set(G_work.nodes())
    if not nx.is_connected(G_work):
        logger.debug("Graph not connected, using largest connected component for Node2Vec training")
        largest_cc = max(nx.connected_components(G_work), key=len)
        G_work = G_work.subgraph(largest_cc).copy()
        trained_nodes = set(G_work.nodes())
        if len(trained_nodes) < 3:
            logger.debug("Largest component too small, skipping Node2Vec")
            return None
    
    # 构建缓存 key（基于工作图）
    node_set = frozenset(G_work.nodes())
    edge_set = frozenset(G_work.edges())
    graph_key = (len(node_set), len(edge_set), node_set, edge_set)
    
    if graph_key in _node2vec_model_cache:
        model = _node2vec_model_cache[graph_key]
    else:
        try:
            node2vec = config.Node2Vec(
                G_work, dimensions=dimensions, walk_length=walk_length,
                num_walks=num_walks, p=p, q=q, workers=workers, quiet=True,
                seed=seed
            )
            model = node2vec.fit(window=10, min_count=1, batch_words=4, seed=seed)
            _node2vec_model_cache[graph_key] = model
            logger.debug(f"Node2Vec model trained on {G_work.number_of_nodes()} nodes (original: {num_original_nodes})")
        except Exception as e:
            logger.warning(f"Node2Vec training failed: {e}, using fallback")
            return None
    
    # 为原始图的所有节点生成嵌入（不在模型中的用零向量）
    embeddings = []
    missing = []
    for node in original_nodes:
        s = str(node)
        if s in model.wv:
            embeddings.append(model.wv[s])
        else:
            missing.append(node)
            embeddings.append(np.zeros(dimensions))
    
    if missing:
        logger.debug(f"Node2Vec: {len(missing)}/{num_original_nodes} nodes using zero vectors (isolated or not in largest CC)")
    
    embeddings_array = np.array(embeddings)
    
    # #region agent log - 假设B: 记录Node2Vec嵌入质量（用于分析稀疏网络嵌入问题）
    import json, os, time as time_module
    debug_log_path = r"e:\项目\02-论文\03-论文计划\17-BIG\code\v2\BiT-HyRL\.cursor\debug.log"
    try:
        os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
        # 计算嵌入质量指标
        zero_vector_ratio = len(missing) / num_original_nodes if num_original_nodes > 0 else 0
        # 计算嵌入的平均范数（排除零向量）
        non_zero_mask = np.any(embeddings_array != 0, axis=1)
        non_zero_embeddings = embeddings_array[non_zero_mask]
        avg_embedding_norm = np.mean(np.linalg.norm(non_zero_embeddings, axis=1)) if len(non_zero_embeddings) > 0 else 0
        # 计算嵌入的方差（衡量区分度）
        embedding_variance = np.var(non_zero_embeddings) if len(non_zero_embeddings) > 0 else 0
        # 计算嵌入的余弦相似度分布（随机采样）
        cos_sim_mean = 0
        if len(non_zero_embeddings) > 1:
            from sklearn.metrics.pairwise import cosine_similarity
            sample_size = min(50, len(non_zero_embeddings))
            sample_idx = np.random.choice(len(non_zero_embeddings), sample_size, replace=False)
            sample_embeddings = non_zero_embeddings[sample_idx]
            cos_sim_matrix = cosine_similarity(sample_embeddings)
            # 取上三角（排除对角线）
            upper_tri = cos_sim_matrix[np.triu_indices_from(cos_sim_matrix, k=1)]
            cos_sim_mean = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0
        log_entry = {
            "timestamp": int(time_module.time() * 1000),
            "location": "features.py:get_node2vec_embeddings",
            "message": "Node2Vec embedding quality for hypothesis B",
            "hypothesisId": "B",
            "sessionId": "debug-session",
            "data": {
                "num_nodes": num_original_nodes,
                "num_edges": G.number_of_edges(),
                "avg_degree": round(2 * G.number_of_edges() / num_original_nodes, 4) if num_original_nodes > 0 else 0,
                "zero_vector_count": len(missing),
                "zero_vector_ratio": round(zero_vector_ratio, 4),
                "avg_embedding_norm": round(float(avg_embedding_norm), 4),
                "embedding_variance": round(float(embedding_variance), 6),
                "embedding_dim": dimensions,
                "avg_cosine_similarity": round(cos_sim_mean, 4),
                "is_connected": nx.is_connected(G)
            }
        }
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        pass
    # #endregion
    
    return embeddings_array


def get_node_features(G, use_node2vec=True, node2vec_dim=64, node2vec_seed=42):
    """提取节点特征（Node2Vec 嵌入 + 5 维统计特征）。"""
    num_nodes = G.number_of_nodes()
    if num_nodes == 0:
        return torch.zeros((0, 5)), []
    device = config.DEVICE
    adj = nx.to_scipy_sparse_array(G, format='coo', dtype=np.float32)
    indices = torch.LongTensor(np.vstack((adj.row, adj.col))).to(device)
    values = torch.FloatTensor(adj.data).to(device)
    adj_tensor = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).to(device)
    adj_dense = adj_tensor.to_dense() if num_nodes < 5000 else None

    degrees = torch.sparse.sum(adj_tensor, dim=1).to_dense()
    deg_norm = degrees / (num_nodes - 1 + 1e-6)
    if adj_dense is not None:
        adj2 = torch.mm(adj_dense, adj_dense)
        adj3 = torch.mm(adj2, adj_dense)
        triangles = torch.diagonal(adj3)
        potential = degrees * (degrees - 1)
        clust = triangles / (potential + 1e-6)
    else:
        cc = nx.clustering(G)
        clust = torch.tensor([cc[n] for n in G.nodes()], dtype=torch.float32).to(device)

    if num_nodes > 0:
        deg_inv = 1.0 / (degrees + 1e-6)
        norm_values = values * deg_inv[indices[1]]
        trans = torch.sparse_coo_tensor(indices, norm_values, (num_nodes, num_nodes)).to(device)
        pr = torch.ones(num_nodes, 1).to(device) / num_nodes
        for _ in range(10):
            pr = 0.85 * torch.sparse.mm(trans, pr) + 0.15 / num_nodes
        pr = pr.squeeze()
    else:
        pr = torch.zeros(num_nodes).to(device)

    eig = torch.ones(num_nodes, 1).to(device)
    for _ in range(10):
        eig = torch.sparse.mm(adj_tensor, eig)
        n = torch.norm(eig)
        if n > 0:
            eig = eig / n
    eig = eig.squeeze()

    if adj_dense is not None:
        katz = torch.sum(torch.mm(adj_dense, adj_dense), dim=1)
        katz = katz / (torch.max(katz) + 1e-6)
    else:
        katz = degrees / (torch.max(degrees) + 1e-6)

    if pr.dim() == 0:
        pr = pr.unsqueeze(0)
    if eig.dim() == 0:
        eig = eig.unsqueeze(0)
    if num_nodes == 1:
        if deg_norm.dim() == 0:
            deg_norm = deg_norm.unsqueeze(0)
        if clust.dim() == 0:
            clust = clust.unsqueeze(0)
        if pr.dim() == 0:
            pr = pr.unsqueeze(0)
        if eig.dim() == 0:
            eig = eig.unsqueeze(0)
        if katz.dim() == 0:
            katz = katz.unsqueeze(0)
    stat = torch.stack([deg_norm, clust, pr, eig, katz], dim=1)

    if use_node2vec:
        emb = get_node2vec_embeddings(G, dimensions=node2vec_dim, seed=node2vec_seed)
        if emb is not None:
            node2vec_t = torch.tensor(emb, dtype=torch.float32, device=device)
            node2vec_norm = F.normalize(node2vec_t, p=2, dim=1)
            features = torch.cat([node2vec_norm, stat], dim=1)
            logger.debug(f"Node2Vec features: {node2vec_dim}D + 5D = {features.shape[1]}D")
        else:
            features = stat
    else:
        features = stat
    
    # #region agent log - 假设C: 记录特征区分度（用于分析稀疏网络特征问题）
    import json, os, time as time_module
    debug_log_path = r"e:\项目\02-论文\03-论文计划\17-BIG\code\v2\BiT-HyRL\.cursor\debug.log"
    try:
        os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
        features_np = features.cpu().numpy()
        # 计算每个特征维度的方差
        feature_variances = np.var(features_np, axis=0)
        avg_variance = float(np.mean(feature_variances))
        # 计算特征值的范围（max - min）
        feature_ranges = np.ptp(features_np, axis=0)
        avg_range = float(np.mean(feature_ranges))
        # 计算特征向量之间的平均距离（随机采样）
        sample_size = min(50, len(features_np))
        if sample_size > 1:
            sample_idx = np.random.choice(len(features_np), sample_size, replace=False)
            sample_features = features_np[sample_idx]
            from sklearn.metrics.pairwise import euclidean_distances
            dist_matrix = euclidean_distances(sample_features)
            avg_pairwise_dist = float(np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)]))
        else:
            avg_pairwise_dist = 0
        log_entry = {
            "timestamp": int(time_module.time() * 1000),
            "location": "features.py:get_node_features",
            "message": "Feature discriminability for hypothesis C",
            "hypothesisId": "C",
            "sessionId": "debug-session",
            "data": {
                "num_nodes": num_nodes,
                "feature_dim": features.shape[1],
                "use_node2vec": use_node2vec and emb is not None,
                "avg_feature_variance": round(avg_variance, 6),
                "avg_feature_range": round(avg_range, 4),
                "avg_pairwise_distance": round(avg_pairwise_dist, 4),
                "stat_feature_variances": {
                    "deg_norm": round(float(feature_variances[-5]), 6) if len(feature_variances) >= 5 else 0,
                    "clust": round(float(feature_variances[-4]), 6) if len(feature_variances) >= 4 else 0,
                    "pagerank": round(float(feature_variances[-3]), 6) if len(feature_variances) >= 3 else 0,
                    "eigenvector": round(float(feature_variances[-2]), 6) if len(feature_variances) >= 2 else 0,
                    "katz": round(float(feature_variances[-1]), 6) if len(feature_variances) >= 1 else 0
                }
            }
        }
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        pass
    # #endregion
    
    return features.cpu(), list(G.nodes())


# ============================================================
# 自适应增强特征（针对小规模网络）
# ============================================================

def get_enhanced_features_for_small_graph(G):
    """
    为小规模网络计算增强特征（不使用Node2Vec）
    
    返回额外的7维特征，专门针对小图优化：
    1. 介数中心性（小图可精确计算）
    2. 接近中心性
    3. K-core数
    4. 平均邻居度
    5. 本地桥接分数
    6. 三角形参与度
    7. 结构洞分数
    """
    num_nodes = G.number_of_nodes()
    node_list = list(G.nodes())
    
    if num_nodes < 3:
        return torch.zeros((num_nodes, 7)), node_list
    
    degrees = dict(G.degree())
    
    # 介数中心性（小图精确计算，大图近似）
    if num_nodes <= 500:
        betweenness = nx.betweenness_centrality(G)
    else:
        betweenness = nx.betweenness_centrality(G, k=min(100, num_nodes))
    
    # 接近中心性
    if nx.is_connected(G):
        closeness = nx.closeness_centrality(G)
    else:
        closeness = {n: 0.0 for n in G.nodes()}
        for comp in nx.connected_components(G):
            subG = G.subgraph(comp)
            for node, val in nx.closeness_centrality(subG).items():
                closeness[node] = val
    
    # K-core数
    core_number = nx.core_number(G)
    max_core = max(core_number.values()) if core_number else 1
    
    # 平均邻居度
    avg_neighbor_degree = nx.average_neighbor_degree(G)
    
    # 三角形数
    triangles = nx.triangles(G)
    
    features = []
    for node in node_list:
        deg = degrees[node]
        
        # 本地桥接分数
        neighbors = list(G.neighbors(node))
        if len(neighbors) >= 2:
            neighbor_edges = sum(1 for i, n1 in enumerate(neighbors) 
                               for n2 in neighbors[i+1:] if G.has_edge(n1, n2))
            max_possible = len(neighbors) * (len(neighbors) - 1) / 2
            bridge_score = 1.0 - neighbor_edges / max_possible if max_possible > 0 else 0.0
        else:
            bridge_score = 0.0
        
        # 三角形参与度
        if deg >= 2:
            triangle_ratio = triangles[node] / (deg * (deg - 1) / 2)
        else:
            triangle_ratio = 0.0
        
        # 结构洞分数（与桥接分数类似但考虑更多）
        structural_hole = bridge_score * (1.0 - triangle_ratio)
        
        feat = [
            betweenness[node],
            closeness[node],
            core_number[node] / max_core,
            avg_neighbor_degree[node] / max(num_nodes - 1, 1) if deg > 0 else 0,
            bridge_score,
            triangle_ratio,
            structural_hole,
        ]
        features.append(feat)
    
    features = torch.tensor(features, dtype=torch.float32)
    
    # 归一化
    if num_nodes > 1:
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True) + 1e-6
        features = (features - mean) / std
        # Min-max缩放到[0,1]
        min_val = features.min(dim=0, keepdim=True)[0]
        max_val = features.max(dim=0, keepdim=True)[0]
        features = (features - min_val) / (max_val - min_val + 1e-6)
    
    return features, node_list


def get_adaptive_node_features(G, use_node2vec=True, node2vec_dim=64, node2vec_seed=42):
    """
    自适应节点特征提取
    
    根据网络规模自动选择最佳特征提取策略：
    - 小规模网络（<50节点）：使用增强统计特征（不依赖Node2Vec）
    - 中等规模网络（50-200节点）：混合策略
    - 大规模网络（>200节点）：标准Node2Vec + 统计特征
    
    Args:
        G: NetworkX图
        use_node2vec: 是否使用Node2Vec（大图时）
        node2vec_dim: Node2Vec嵌入维度
        
    Returns:
        features: 节点特征张量
        node_list: 节点列表
        feature_type: 特征类型标识 ('enhanced', 'mixed', 'standard')
    """
    num_nodes = G.number_of_nodes()
    
    if num_nodes == 0:
        return torch.zeros((0, 5)), [], 'empty'
    
    # 小规模网络：使用增强统计特征
    if num_nodes < SMALL_NETWORK_THRESHOLD:
        # 基础5维特征
        base_features, node_list = get_node_features(G, use_node2vec=False)
        
        # 增强7维特征
        enhanced_features, _ = get_enhanced_features_for_small_graph(G)
        
        # 合并：5D + 7D = 12D
        features = torch.cat([base_features, enhanced_features], dim=1)
        
        logger.debug(f"小规模网络 ({num_nodes} nodes): 使用增强特征 {features.shape[1]}D")
        return features, node_list, 'enhanced'
    
    # 中等规模网络：混合策略
    elif num_nodes < MEDIUM_NETWORK_THRESHOLD:
        # 尝试Node2Vec，失败则使用增强特征
        if use_node2vec and config.NODE2VEC_AVAILABLE:
            features, node_list = get_node_features(G, use_node2vec=True, node2vec_dim=node2vec_dim, node2vec_seed=node2vec_seed)
            if features.shape[1] > 5:  # Node2Vec成功
                logger.debug(f"中等规模网络 ({num_nodes} nodes): 使用Node2Vec特征 {features.shape[1]}D")
                return features, node_list, 'standard'
        
        # Node2Vec失败，使用增强特征
        base_features, node_list = get_node_features(G, use_node2vec=False)
        enhanced_features, _ = get_enhanced_features_for_small_graph(G)
        features = torch.cat([base_features, enhanced_features], dim=1)
        
        logger.debug(f"中等规模网络 ({num_nodes} nodes): Node2Vec失败，使用增强特征 {features.shape[1]}D")
        return features, node_list, 'enhanced'
    
    # 大规模网络：标准流程
    else:
        features, node_list = get_node_features(G, use_node2vec=use_node2vec, node2vec_dim=node2vec_dim, node2vec_seed=node2vec_seed)
        feature_type = 'standard' if features.shape[1] > 5 else 'basic'
        logger.debug(f"大规模网络 ({num_nodes} nodes): 使用{feature_type}特征 {features.shape[1]}D")
        return features, node_list, feature_type
