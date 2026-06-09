# -*- coding: utf-8 -*-
"""
BiT-HyRL GNN 策略网络 (GAT + Actor-Critic)

使用 Graph Attention Network 捕捉图结构信息，
支持 PPO 训练的 Actor-Critic 架构。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from torch_geometric.utils import from_networkx, degree
import networkx as nx
import numpy as np

from . import config

DEVICE = config.DEVICE


class GATPolicy(nn.Module):
    """
    深层 GAT 策略网络 + PPO Value Head
    
    架构:
    - 输入投影层（将 Node2Vec 嵌入投影到隐藏维度）
    - 5层 GAT 提取高阶图结构特征（带残差连接和跳跃连接）
    - 全局注意力池化获取图级表示
    - Actor Head: 节点选择得分
    - Critic Head: 状态价值估计
    
    Args:
        in_channels: 输入节点特征维度 (默认128，Node2Vec嵌入维度)
        hidden_channels: 隐藏层维度 (默认128)
        heads: GAT 注意力头数 (默认8)
        dropout: Dropout 比例
        num_layers: GAT 层数 (默认5，捕捉高阶信息)
    """
    
    def __init__(self, in_channels=128, hidden_channels=128, heads=8, dropout=0.1, num_layers=5):
        super(GATPolicy, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers
        
        # 输入投影层：将 Node2Vec 嵌入投影到 hidden_channels
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU()
        )
        
        # GAT Layers with residual connections
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                # 第一层：hidden_channels -> hidden_channels * heads
                self.gat_layers.append(GATConv(
                    hidden_channels, 
                    hidden_channels // heads,  # 每个头的维度
                    heads=heads, 
                    dropout=dropout,
                    concat=True  # 输出维度 = (hidden_channels // heads) * heads = hidden_channels
                ))
                self.layer_norms.append(nn.LayerNorm(hidden_channels))
                # 残差：维度匹配，直接用 Identity
                self.residual_projs.append(nn.Identity())
            elif i == num_layers - 1:
                # 最后一层：hidden_channels -> hidden_channels (单头)
                self.gat_layers.append(GATConv(
                    hidden_channels, 
                    hidden_channels, 
                    heads=1, 
                    dropout=dropout,
                    concat=False
                ))
                self.layer_norms.append(nn.LayerNorm(hidden_channels))
                self.residual_projs.append(nn.Identity())
            else:
                # 中间层：保持 hidden_channels 维度
                self.gat_layers.append(GATConv(
                    hidden_channels, 
                    hidden_channels // heads,
                    heads=heads, 
                    dropout=dropout,
                    concat=True
                ))
                self.layer_norms.append(nn.LayerNorm(hidden_channels))
                self.residual_projs.append(nn.Identity())
        
        # 跳跃连接聚合（收集所有层的输出）
        self.jump_proj = nn.Linear(hidden_channels * num_layers, hidden_channels)
        self.jump_norm = nn.LayerNorm(hidden_channels)
        
        # 已选控制器的位置编码（可学习）
        self.selected_embedding = nn.Parameter(torch.randn(1, hidden_channels))
        
        # 全局上下文注意力（多头）
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )
        
        # Actor Head: 节点选择得分（深层 MLP）
        self.actor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),  # 节点特征 + 全局上下文
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ELU(),
            nn.Linear(hidden_channels // 4, 1)
        )
        
        # Critic Head: 状态价值估计（深层 MLP）
        self.critic = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ELU(),
            nn.Linear(hidden_channels // 4, 1)
        )
        
        # 温度参数（可学习）
        self.temperature = nn.Parameter(torch.ones(1))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch=None, selected_mask=None):
        """
        前向传播（深层 GAT，带残差连接和跳跃连接）
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            batch: 批次索引（多图时使用）
            selected_mask: 已选节点 mask [num_nodes] (True = 已选)
            
        Returns:
            probs: 节点选择概率 [num_nodes]
            value: 状态价值估计 [1] 或 [batch_size]
        """
        # 输入投影
        h = self.input_proj(x)
        
        # 收集所有层的输出（用于跳跃连接）
        layer_outputs = []
        
        # GAT Layers with residual connections
        for i, (gat, ln, res_proj) in enumerate(zip(self.gat_layers, self.layer_norms, self.residual_projs)):
            # 保存残差
            residual = res_proj(h)
            
            # GAT 前向传播
            h = gat(h, edge_index)
            h = ln(h)
            h = F.elu(h)
            
            # 添加残差连接
            h = h + residual
            
            # 收集层输出
            layer_outputs.append(h)
            
            # Dropout（除了最后一层）
            if i < self.num_layers - 1:
                h = self.dropout(h)
        
        # 跳跃连接：聚合所有层的输出
        jump_concat = torch.cat(layer_outputs, dim=-1)  # [num_nodes, hidden * num_layers]
        h = self.jump_proj(jump_concat)  # [num_nodes, hidden]
        h = self.jump_norm(h)
        h = F.elu(h)
        
        # 添加已选控制器的位置编码
        if selected_mask is not None and selected_mask.any():
            # 为已选节点添加位置嵌入
            selected_embedding = self.selected_embedding.expand(h.size(0), -1)
            h = h + selected_mask.float().unsqueeze(-1) * selected_embedding
        
        # 全局注意力池化
        attention_weights = self.global_attention(h)  # [num_nodes, 1]
        attention_weights = F.softmax(attention_weights, dim=0)
        global_context = (attention_weights * h).sum(dim=0, keepdim=True)  # [1, hidden]
        
        # 如果是批处理，使用 batch 信息
        if batch is not None:
            # 批处理情况下的全局池化
            global_context = global_mean_pool(h, batch)  # [batch_size, hidden]
            # 扩展到每个节点
            global_context_expanded = global_context[batch]  # [num_nodes, hidden]
        else:
            # 单图情况
            global_context_expanded = global_context.expand(h.size(0), -1)
        
        # Actor: 节点选择得分
        actor_input = torch.cat([h, global_context_expanded], dim=-1)
        scores = self.actor(actor_input).squeeze(-1)  # [num_nodes]
        
        # 应用温度缩放
        temp = torch.clamp(self.temperature, min=0.1, max=2.0)
        scores = scores / temp
        
        # 应用 mask（已选节点不能再选）
        if selected_mask is not None:
            scores = scores.masked_fill(selected_mask, -1e9)
        
        # Softmax 得到概率
        probs = F.softmax(scores, dim=0)
        
        # Critic: 状态价值估计
        if batch is not None:
            pooled = global_mean_pool(h, batch)
        else:
            pooled = h.mean(dim=0, keepdim=True)
        value = self.critic(pooled).squeeze(-1)  # [batch_size] 或 [1]
        
        return probs, value
    
    def get_action(self, x, edge_index, selected_mask=None, deterministic=False):
        """
        获取动作（选择一个节点）
        
        Args:
            x: 节点特征
            edge_index: 边索引
            selected_mask: 已选节点 mask
            deterministic: 是否使用确定性策略（贪婪选择）
            
        Returns:
            action: 选择的节点索引
            log_prob: 动作的对数概率
            value: 状态价值估计
        """
        probs, value = self.forward(x, edge_index, selected_mask=selected_mask)
        
        if deterministic:
            action = torch.argmax(probs)
            log_prob = torch.log(probs[action] + 1e-10)
        else:
            # 采样动作
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value, probs
    
    def evaluate_actions(self, x, edge_index, actions, selected_masks=None):
        """
        评估动作（PPO 更新时使用）
        
        Args:
            x: 节点特征
            edge_index: 边索引
            actions: 已执行的动作
            selected_masks: 当时的已选节点 mask
            
        Returns:
            log_probs: 动作的对数概率
            values: 状态价值估计
            entropy: 策略熵
        """
        probs, values = self.forward(x, edge_index, selected_mask=selected_masks)
        
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy


class GATPolicyLegacy(nn.Module):
    """
    旧版 2 层 GAT 策略网络，用于加载 deep_gat_gcc_L5_H8_D128.pth 等
    使用 gat1/gat2、ln1/ln2 命名的 checkpoint。
    接口与 GATPolicy 一致：forward(x, edge_index, batch, selected_mask)、get_action()。
    """
    def __init__(self, in_channels=128, hidden_channels=128, heads=8, dropout=0.1, num_layers=2):
        super(GATPolicyLegacy, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers
        # 2 层 GAT，命名与旧 checkpoint 一致
        self.gat1 = GATConv(
            in_channels,
            hidden_channels // heads,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.gat2 = GATConv(
            hidden_channels,
            hidden_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )
        self.ln2 = nn.LayerNorm(hidden_channels)
        # global_attention: 3 个子模块，索引 2 为 Linear(hidden, 1)
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, 1),
        )
        # actor: 4 个子模块，索引 3 为最后一层 Linear
        self.actor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, 1),
        )
        # critic: 4 个子模块
        self.critic = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None, selected_mask=None):
        h = self.gat1(x, edge_index)
        h = self.ln1(h)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.gat2(h, edge_index)
        h = self.ln2(h)
        h = F.elu(h)
        # 全局注意力
        attention_weights = self.global_attention(h)
        attention_weights = F.softmax(attention_weights, dim=0)
        global_context = (attention_weights * h).sum(dim=0, keepdim=True)
        if batch is not None:
            global_context = global_mean_pool(h, batch)
            global_context_expanded = global_context[batch]
        else:
            global_context_expanded = global_context.expand(h.size(0), -1)
        # Actor
        actor_input = torch.cat([h, global_context_expanded], dim=-1)
        scores = self.actor(actor_input).squeeze(-1)
        if selected_mask is not None:
            scores = scores.masked_fill(selected_mask, -1e9)
        probs = F.softmax(scores, dim=0)
        # Critic
        if batch is not None:
            pooled = global_mean_pool(h, batch)
        else:
            pooled = h.mean(dim=0, keepdim=True)
        value = self.critic(pooled).squeeze(-1)
        return probs, value

    def get_action(self, x, edge_index, selected_mask=None, deterministic=False):
        probs, value = self.forward(x, edge_index, selected_mask=selected_mask)
        if deterministic:
            action = torch.argmax(probs)
            log_prob = torch.log(probs[action] + 1e-10)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob, value, probs


def graph_to_pyg_data(G, node_features=None, device=None):
    """
    将 NetworkX 图转换为 PyTorch Geometric Data 对象
    
    Args:
        G: NetworkX 图
        node_features: 节点特征张量 [num_nodes, feat_dim]，如果为 None 则使用度数特征
        device: 目标设备
        
    Returns:
        data: PyG Data 对象
        node_list: 节点列表（保持顺序）
    """
    if device is None:
        device = DEVICE
    
    # 获取节点列表（固定顺序）
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    num_nodes = len(node_list)
    
    # 构建边索引
    edges = list(G.edges())
    if len(edges) > 0:
        # 无向图需要双向边
        edge_index = []
        for u, v in edges:
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_index.append([node_to_idx[v], node_to_idx[u]])
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    
    # 节点特征
    if node_features is not None:
        x = node_features.to(device)
    else:
        # 默认使用度数作为特征
        degrees = torch.tensor([G.degree(n) for n in node_list], dtype=torch.float32, device=device)
        x = degrees.unsqueeze(-1)  # [num_nodes, 1]
    
    return x, edge_index, node_list


def get_node2vec_features(G, device=None, dimensions=128, walk_length=20, num_walks=100, p=1, q=1):
    """
    获取 Node2Vec 节点嵌入特征
    
    使用 Node2Vec 算法生成高维节点嵌入，捕捉图的结构信息。
    
    Args:
        G: NetworkX 图
        device: 目标设备
        dimensions: 嵌入维度 (默认128)
        walk_length: 随机游走长度 (默认20)
        num_walks: 每个节点的随机游走次数 (默认100)
        p: 返回参数 (默认1)
        q: 进出参数 (默认1)
        
    Returns:
        features: 节点特征张量 [num_nodes, dimensions]
        node_list: 节点列表
    """
    if device is None:
        device = DEVICE
    
    node_list = list(G.nodes())
    num_nodes = len(node_list)
    
    if num_nodes == 0:
        return torch.zeros((0, dimensions), device=device), []
    
    if num_nodes < 3 or G.number_of_edges() == 0:
        # 图太小，返回随机初始化的嵌入
        features = torch.randn(num_nodes, dimensions, device=device) * 0.1
        return features, node_list
    
    # 尝试使用 Node2Vec
    try:
        from node2vec import Node2Vec
        
        # 创建工作副本
        G_work = G.copy()
        
        # 清理边权重
        for u, v, data in G_work.edges(data=True):
            if 'weight' in data:
                w = data['weight']
                if not np.isfinite(w) or w <= 0:
                    data['weight'] = 1.0
        
        # 移除孤立节点
        isolates = list(nx.isolates(G_work))
        if isolates:
            G_work.remove_nodes_from(isolates)
        
        # 检查处理后的图
        if G_work.number_of_nodes() < 3 or G_work.number_of_edges() == 0:
            features = torch.randn(num_nodes, dimensions, device=device) * 0.1
            return features, node_list
        
        # 训练 Node2Vec
        node2vec = Node2Vec(
            G_work, 
            dimensions=dimensions, 
            walk_length=walk_length,
            num_walks=num_walks, 
            p=p, 
            q=q, 
            workers=1, 
            quiet=True
        )
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        # 提取嵌入
        embeddings = []
        for node in node_list:
            s = str(node)
            if s in model.wv:
                embeddings.append(model.wv[s])
            else:
                # 孤立节点使用零向量
                embeddings.append(np.zeros(dimensions))
        
        features = np.array(embeddings, dtype=np.float32)
        
        # L2 归一化
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        features = features / norms
        
        features = torch.tensor(features, dtype=torch.float32, device=device)
        return features, node_list
        
    except ImportError:
        print("Warning: node2vec not installed, using random features")
        features = torch.randn(num_nodes, dimensions, device=device) * 0.1
        return features, node_list
    except Exception as e:
        print(f"Warning: Node2Vec failed: {e}, using random features")
        features = torch.randn(num_nodes, dimensions, device=device) * 0.1
        return features, node_list


def get_gnn_node_features(G, device=None, embed_dim=128):
    """
    获取 GNN 训练/推理使用的节点特征（Node2Vec 嵌入）
    
    这是 GNN 模型的主要特征提取函数。
    
    Args:
        G: NetworkX 图
        device: 目标设备
        embed_dim: 嵌入维度 (默认128)
        
    Returns:
        features: 节点特征张量 [num_nodes, embed_dim]
        node_list: 节点列表
    """
    return get_node2vec_features(G, device=device, dimensions=embed_dim)


# ============================================================
# 规模自适应特征提取和统一模型架构
# ============================================================

# 网络规模阈值常量
SCALE_TINY = 30        # 极小规模
SCALE_SMALL = 50       # 小规模
SCALE_MEDIUM_LOW = 100 # 中小规模
SCALE_MEDIUM = 200     # 中等规模
SCALE_LARGE = 500      # 大规模


def get_scale_adaptive_node2vec(G, device=None, target_dim=64):
    """
    规模自适应的Node2Vec特征提取
    
    根据网络规模自动调整Node2Vec参数，确保不同规模网络都能获得高质量的特征表示。
    
    参数设计原理:
    - 小网络：更短游走避免路径重复、更高q值增加BFS特性探索局部结构
    - 大网络：标准游走长度、更低q值允许DFS探索全局结构
    - 统一输出target_dim维度
    
    Args:
        G: NetworkX 图
        device: 目标设备
        target_dim: 目标输出维度 (默认64)
        
    Returns:
        features: 节点特征张量 [num_nodes, raw_dim]
        node_list: 节点列表
        raw_dim: 原始Node2Vec嵌入维度（用于后续投影）
    """
    if device is None:
        device = DEVICE
    
    node_list = list(G.nodes())
    num_nodes = len(node_list)
    
    if num_nodes == 0:
        return torch.zeros((0, target_dim), device=device), [], target_dim
    
    # 根据规模自适应选择参数
    if num_nodes < SCALE_TINY:
        # 极小网络：最短游走，高BFS偏好
        params = dict(dimensions=32, walk_length=6, num_walks=100, p=1.0, q=2.5)
    elif num_nodes < SCALE_SMALL:
        # 小网络：短游走，较高BFS偏好
        params = dict(dimensions=32, walk_length=8, num_walks=80, p=1.0, q=2.0)
    elif num_nodes < SCALE_MEDIUM_LOW:
        # 中小网络：适中游走
        params = dict(dimensions=48, walk_length=12, num_walks=60, p=1.0, q=1.5)
    elif num_nodes < SCALE_MEDIUM:
        # 中等网络：标准参数
        params = dict(dimensions=64, walk_length=15, num_walks=50, p=1.0, q=1.0)
    else:
        # 大网络：长游走，DFS偏好
        params = dict(dimensions=64, walk_length=20, num_walks=40, p=1.0, q=0.8)
    
    raw_dim = params['dimensions']
    
    # 检查图的有效性
    if num_nodes < 3 or G.number_of_edges() == 0:
        features = torch.randn(num_nodes, raw_dim, device=device) * 0.1
        return features, node_list, raw_dim
    
    # 尝试使用 Node2Vec
    try:
        from node2vec import Node2Vec
        
        # 创建工作副本
        G_work = G.copy()
        
        # 清理边权重
        for u, v, data in G_work.edges(data=True):
            if 'weight' in data:
                w = data['weight']
                if not np.isfinite(w) or w <= 0:
                    data['weight'] = 1.0
        
        # 移除孤立节点
        isolates = list(nx.isolates(G_work))
        if isolates:
            G_work.remove_nodes_from(isolates)
        
        # 检查处理后的图
        if G_work.number_of_nodes() < 3 or G_work.number_of_edges() == 0:
            features = torch.randn(num_nodes, raw_dim, device=device) * 0.1
            return features, node_list, raw_dim
        
        # 如果图不连通，使用最大连通分量
        if not nx.is_connected(G_work):
            largest_cc = max(nx.connected_components(G_work), key=len)
            G_work = G_work.subgraph(largest_cc).copy()
            if G_work.number_of_nodes() < 3:
                features = torch.randn(num_nodes, raw_dim, device=device) * 0.1
                return features, node_list, raw_dim
        
        # 训练 Node2Vec
        node2vec = Node2Vec(
            G_work, 
            dimensions=params['dimensions'], 
            walk_length=params['walk_length'],
            num_walks=params['num_walks'], 
            p=params['p'], 
            q=params['q'], 
            workers=1, 
            quiet=True
        )
        
        # 调整window大小以适应小图
        window_size = min(10, max(3, num_nodes // 5))
        model = node2vec.fit(window=window_size, min_count=1, batch_words=4)
        
        # 提取嵌入
        embeddings = []
        for node in node_list:
            s = str(node)
            if s in model.wv:
                embeddings.append(model.wv[s])
            else:
                # 孤立节点使用小随机值
                embeddings.append(np.random.randn(raw_dim) * 0.1)
        
        features = np.array(embeddings, dtype=np.float32)
        
        # L2 归一化
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        features = features / norms
        
        features = torch.tensor(features, dtype=torch.float32, device=device)
        return features, node_list, raw_dim
        
    except ImportError:
        print("Warning: node2vec not installed, using random features")
        features = torch.randn(num_nodes, raw_dim, device=device) * 0.1
        return features, node_list, raw_dim
    except Exception as e:
        print(f"Warning: Node2Vec failed: {e}, using random features")
        features = torch.randn(num_nodes, raw_dim, device=device) * 0.1
        return features, node_list, raw_dim


def get_scale_encoding(G, dim=8, device=None):
    """
    计算网络规模编码，作为全局特征
    
    编码网络的规模和结构特性，帮助模型感知当前处理的网络类型。
    
    特征包括:
    1. log(节点数) 归一化
    2. 网络密度
    3. 平均度归一化
    4. 最大度归一化
    5. 度数标准差归一化
    6. 聚类系数
    7. 连通性指标
    8. 规模类别编码
    
    Args:
        G: NetworkX 图
        dim: 编码维度 (默认8)
        device: 目标设备
        
    Returns:
        encoding: 规模编码张量 [dim]
    """
    if device is None:
        device = DEVICE
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    if n == 0:
        return torch.zeros(dim, device=device)
    
    # 1. log(节点数) 归一化到 [0, 1]，假设最大1000节点
    log_n = np.log(n + 1) / np.log(1000)
    log_n = min(1.0, log_n)
    
    # 2. 网络密度
    max_edges = n * (n - 1) / 2
    density = m / max_edges if max_edges > 0 else 0
    
    # 3. 平均度归一化
    avg_degree = 2 * m / n if n > 0 else 0
    avg_degree_norm = min(1.0, avg_degree / 50)  # 假设最大平均度50
    
    # 4. 最大度归一化
    degrees = [d for _, d in G.degree()]
    max_degree = max(degrees) if degrees else 0
    max_degree_norm = min(1.0, max_degree / (n - 1)) if n > 1 else 0
    
    # 5. 度数标准差归一化
    if len(degrees) > 1:
        degree_std = np.std(degrees)
        degree_std_norm = min(1.0, degree_std / (avg_degree + 1e-6))
    else:
        degree_std_norm = 0
    
    # 6. 平均聚类系数
    try:
        avg_clustering = nx.average_clustering(G)
    except:
        avg_clustering = 0
    
    # 7. 连通性指标（连通分量数量的倒数）
    try:
        num_components = nx.number_connected_components(G)
        connectivity = 1.0 / num_components
    except:
        connectivity = 0
    
    # 8. 规模类别编码（one-hot style，但连续化）
    # 根据规模返回不同的值
    if n < SCALE_SMALL:
        scale_category = 0.2
    elif n < SCALE_MEDIUM_LOW:
        scale_category = 0.4
    elif n < SCALE_MEDIUM:
        scale_category = 0.6
    elif n < SCALE_LARGE:
        scale_category = 0.8
    else:
        scale_category = 1.0
    
    # 组合编码
    encoding = [
        log_n,
        density,
        avg_degree_norm,
        max_degree_norm,
        degree_std_norm,
        avg_clustering,
        connectivity,
        scale_category,
    ]
    
    # 确保维度匹配
    if len(encoding) < dim:
        encoding.extend([0.0] * (dim - len(encoding)))
    elif len(encoding) > dim:
        encoding = encoding[:dim]
    
    return torch.tensor(encoding, dtype=torch.float32, device=device)


class FeatureProjector(nn.Module):
    """
    特征投影层：将不同维度的Node2Vec特征投影到统一维度
    
    支持动态输入维度，自动选择对应的投影层。
    """
    
    def __init__(self, output_dim=64):
        """
        Args:
            output_dim: 统一的输出维度 (默认64)
        """
        super().__init__()
        self.output_dim = output_dim
        
        # 为不同的输入维度创建投影层
        self.projectors = nn.ModuleDict({
            '32': nn.Linear(32, output_dim),
            '48': nn.Linear(48, output_dim),
            '64': nn.Linear(64, output_dim) if output_dim != 64 else nn.Identity(),
            '128': nn.Linear(128, output_dim),
        })
        
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.ELU()
    
    def forward(self, x, input_dim=None):
        """
        前向传播
        
        Args:
            x: 输入特征 [num_nodes, input_dim]
            input_dim: 输入维度（如果为None，自动检测）
            
        Returns:
            projected: 投影后的特征 [num_nodes, output_dim]
        """
        if input_dim is None:
            input_dim = x.size(-1)
        
        key = str(input_dim)
        
        if key in self.projectors:
            projected = self.projectors[key](x)
        else:
            # 动态创建投影层（如果需要）
            proj = nn.Linear(input_dim, self.output_dim).to(x.device)
            projected = proj(x)
        
        return self.activation(self.norm(projected))


class UnifiedGATPolicy(nn.Module):
    """
    统一的GAT策略网络，支持所有规模的网络
    
    架构特点:
    - 规模自适应Node2Vec特征作为输入
    - 规模编码作为全局条件
    - 3层GAT（减少层数避免小网络过拟合）
    - 4个注意力头（减少头数）
    - 残差连接和跳跃连接
    - Actor-Critic架构
    
    Args:
        in_channels: Node2Vec投影后的统一维度 (默认64)
        hidden_channels: 隐藏层维度 (默认96)
        scale_encoding_dim: 规模编码维度 (默认8)
        heads: 注意力头数 (默认4)
        num_layers: GAT层数 (默认3)
        dropout: Dropout比例 (默认0.15)
    """
    
    def __init__(
        self,
        in_channels=64,
        hidden_channels=96,
        scale_encoding_dim=8,
        heads=4,
        num_layers=3,
        dropout=0.15,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.scale_encoding_dim = scale_encoding_dim
        self.heads = heads
        self.num_layers = num_layers
        
        # 特征投影器
        self.feature_projector = FeatureProjector(output_dim=in_channels)
        
        # 输入投影层：Node2Vec特征 + 规模编码
        total_input_dim = in_channels + scale_encoding_dim
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout * 0.5)  # 输入层使用较小的dropout
        )
        
        # GAT Layers with residual connections
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # 最后一层：单头
                self.gat_layers.append(GATConv(
                    hidden_channels, 
                    hidden_channels, 
                    heads=1, 
                    dropout=dropout,
                    concat=False
                ))
            else:
                # 其他层：多头
                self.gat_layers.append(GATConv(
                    hidden_channels, 
                    hidden_channels // heads,
                    heads=heads, 
                    dropout=dropout,
                    concat=True
                ))
            self.layer_norms.append(nn.LayerNorm(hidden_channels))
        
        # 跳跃连接聚合
        self.jump_proj = nn.Linear(hidden_channels * num_layers, hidden_channels)
        self.jump_norm = nn.LayerNorm(hidden_channels)
        
        # 已选控制器的位置编码
        self.selected_embedding = nn.Parameter(torch.randn(1, hidden_channels) * 0.1)
        
        # 全局上下文注意力
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )
        
        # Actor Head: 节点选择得分
        self.actor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # Critic Head: 状态价值估计
        self.critic = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # 温度参数
        self.temperature = nn.Parameter(torch.ones(1))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, scale_encoding=None, batch=None, selected_mask=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            scale_encoding: 规模编码 [scale_encoding_dim] 或 None
            batch: 批次索引（多图时使用）
            selected_mask: 已选节点 mask [num_nodes]
            
        Returns:
            probs: 节点选择概率 [num_nodes]
            value: 状态价值估计 [1] 或 [batch_size]
        """
        num_nodes = x.size(0)
        
        # 如果没有规模编码，创建默认的
        if scale_encoding is None:
            scale_encoding = torch.zeros(self.scale_encoding_dim, device=x.device)
        
        # 扩展规模编码到所有节点
        scale_expanded = scale_encoding.unsqueeze(0).expand(num_nodes, -1)
        
        # 拼接节点特征和规模编码
        x_combined = torch.cat([x, scale_expanded], dim=-1)
        
        # 输入投影
        h = self.input_proj(x_combined)
        
        # 收集所有层的输出
        layer_outputs = []
        
        # GAT Layers with residual connections
        for i, (gat, ln) in enumerate(zip(self.gat_layers, self.layer_norms)):
            residual = h
            
            h = gat(h, edge_index)
            h = ln(h)
            h = F.elu(h)
            h = h + residual  # 残差连接
            
            layer_outputs.append(h)
            
            if i < self.num_layers - 1:
                h = self.dropout(h)
        
        # 跳跃连接
        jump_concat = torch.cat(layer_outputs, dim=-1)
        h = self.jump_proj(jump_concat)
        h = self.jump_norm(h)
        h = F.elu(h)
        
        # 添加已选控制器的位置编码
        if selected_mask is not None and selected_mask.any():
            selected_emb = self.selected_embedding.expand(num_nodes, -1)
            h = h + selected_mask.float().unsqueeze(-1) * selected_emb
        
        # 全局注意力池化
        attention_weights = self.global_attention(h)
        attention_weights = F.softmax(attention_weights, dim=0)
        global_context = (attention_weights * h).sum(dim=0, keepdim=True)
        
        if batch is not None:
            global_context = global_mean_pool(h, batch)
            global_context_expanded = global_context[batch]
        else:
            global_context_expanded = global_context.expand(num_nodes, -1)
        
        # Actor: 节点选择得分
        actor_input = torch.cat([h, global_context_expanded], dim=-1)
        scores = self.actor(actor_input).squeeze(-1)
        
        # 温度缩放
        temp = torch.clamp(self.temperature, min=0.1, max=2.0)
        scores = scores / temp
        
        # 应用 mask
        if selected_mask is not None:
            scores = scores.masked_fill(selected_mask, -1e9)
        
        probs = F.softmax(scores, dim=0)
        
        # Critic: 状态价值估计
        if batch is not None:
            pooled = global_mean_pool(h, batch)
        else:
            pooled = h.mean(dim=0, keepdim=True)
        value = self.critic(pooled).squeeze(-1)
        
        return probs, value
    
    def get_action(self, x, edge_index, scale_encoding=None, selected_mask=None, deterministic=False):
        """
        获取动作（选择一个节点）
        
        Args:
            x: 节点特征
            edge_index: 边索引
            scale_encoding: 规模编码
            selected_mask: 已选节点 mask
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 选择的节点索引
            log_prob: 动作的对数概率
            value: 状态价值估计
            probs: 所有节点的概率
        """
        probs, value = self.forward(x, edge_index, scale_encoding, selected_mask=selected_mask)
        
        if deterministic:
            action = torch.argmax(probs)
            log_prob = torch.log(probs[action] + 1e-10)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value, probs
    
    def evaluate_actions(self, x, edge_index, actions, scale_encoding=None, selected_masks=None):
        """
        评估动作（PPO更新时使用）
        
        Args:
            x: 节点特征
            edge_index: 边索引
            actions: 已执行的动作
            scale_encoding: 规模编码
            selected_masks: 当时的已选节点 mask
            
        Returns:
            log_probs: 动作的对数概率
            values: 状态价值估计
            entropy: 策略熵
        """
        probs, values = self.forward(x, edge_index, scale_encoding, selected_mask=selected_masks)
        
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy


def get_unified_features(G, device=None, target_dim=64):
    """
    获取统一模型使用的特征
    
    包括规模自适应的Node2Vec特征和规模编码。
    
    Args:
        G: NetworkX 图
        device: 目标设备
        target_dim: 目标特征维度
        
    Returns:
        node_features: 节点特征张量 [num_nodes, raw_dim]
        scale_encoding: 规模编码张量 [8]
        node_list: 节点列表
        raw_dim: 原始Node2Vec维度
    """
    if device is None:
        device = DEVICE
    
    # 获取规模自适应的Node2Vec特征
    node_features, node_list, raw_dim = get_scale_adaptive_node2vec(G, device=device, target_dim=target_dim)
    
    # 获取规模编码
    scale_encoding = get_scale_encoding(G, dim=8, device=device)
    
    return node_features, scale_encoding, node_list, raw_dim


# ============================================================
# 动态覆盖特征 (Dynamic State Representation)
# ============================================================

def compute_coverage_features(G, centers, node_list, device=None):
    """
    计算动态覆盖特征
    
    为每个节点计算到最近控制器的距离，以及是否被覆盖的信息。
    这让模型直接"看到"哪些区域还是盲区，显著提升覆盖率指标的收敛速度。
    
    特征包括:
    1. is_covered: 是否被覆盖（节点到最近控制器的距离 <= 2）
    2. distance_to_nearest: 到最近控制器的距离（归一化）
    3. coverage_density: 局部覆盖密度（邻域内有多少控制器）
    4. is_controller: 该节点是否已被选为控制器
    
    Args:
        G: NetworkX 图
        centers: 已选择的控制器列表
        node_list: 节点列表（保持顺序）
        device: 目标设备
        
    Returns:
        coverage_features: 覆盖特征张量 [num_nodes, 4]
    """
    if device is None:
        device = DEVICE
    
    num_nodes = len(node_list)
    
    if num_nodes == 0:
        return torch.zeros((0, 4), device=device)
    
    centers_set = set(centers) if centers else set()
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    
    # 初始化特征
    is_covered = np.zeros(num_nodes)
    distance_to_nearest = np.ones(num_nodes) * float('inf')
    coverage_density = np.zeros(num_nodes)
    is_controller = np.zeros(num_nodes)
    
    # 标记控制器节点
    for c in centers_set:
        if c in node_to_idx:
            is_controller[node_to_idx[c]] = 1.0
    
    if centers_set:
        # 计算每个节点到最近控制器的距离
        # 使用 BFS 从所有控制器同时开始，计算最短距离
        visited = {}
        queue = []
        
        for c in centers_set:
            if c in node_to_idx:
                visited[c] = 0
                queue.append(c)
        
        # 多源 BFS
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            current_dist = visited[node]
            
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited[neighbor] = current_dist + 1
                    queue.append(neighbor)
        
        # 填充距离特征
        for node, dist in visited.items():
            if node in node_to_idx:
                idx = node_to_idx[node]
                distance_to_nearest[idx] = dist
                # 覆盖范围：距离 <= 2 的节点被认为是"被覆盖的"
                if dist <= 2:
                    is_covered[idx] = 1.0
        
        # 计算局部覆盖密度（每个节点2跳邻域内的控制器数量）
        for i, node in enumerate(node_list):
            # 获取2跳邻域
            neighbors_1hop = set(G.neighbors(node))
            neighbors_2hop = set()
            for n in neighbors_1hop:
                neighbors_2hop.update(G.neighbors(n))
            neighborhood = neighbors_1hop | neighbors_2hop | {node}
            
            # 计算邻域内的控制器数量
            controllers_in_neighborhood = len(centers_set & neighborhood)
            max_possible = min(len(neighborhood), len(centers_set) + 1)
            coverage_density[i] = controllers_in_neighborhood / max_possible if max_possible > 0 else 0.0
    
    # 归一化距离特征（使用网络直径或最大距离）
    max_dist = max(distance_to_nearest[distance_to_nearest != float('inf')]) if np.any(distance_to_nearest != float('inf')) else 1.0
    max_dist = max(max_dist, 1.0)
    distance_to_nearest = np.where(
        distance_to_nearest == float('inf'),
        1.0,  # 无法到达的节点设为1
        distance_to_nearest / max_dist
    )
    # 反转距离（越近越大），使其成为"接近度"特征
    distance_to_nearest = 1.0 - distance_to_nearest
    
    # 组合特征
    features = np.stack([
        is_covered,
        distance_to_nearest,
        coverage_density,
        is_controller,
    ], axis=1)
    
    return torch.tensor(features, dtype=torch.float32, device=device)


def compute_dispersion_features(G, centers, node_list, device=None):
    """
    计算分散度相关特征
    
    帮助模型理解当前控制器的分布情况，促进分散部署。
    
    特征包括:
    1. distance_to_farthest_controller: 到最远控制器的距离
    2. potential_dispersion_gain: 如果选择该节点，分散度的潜在增益
    
    Args:
        G: NetworkX 图
        centers: 已选择的控制器列表
        node_list: 节点列表
        device: 目标设备
        
    Returns:
        dispersion_features: 分散度特征张量 [num_nodes, 2]
    """
    if device is None:
        device = DEVICE
    
    num_nodes = len(node_list)
    
    if num_nodes == 0:
        return torch.zeros((0, 2), device=device)
    
    centers_set = set(centers) if centers else set()
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    
    # 初始化特征
    distance_to_farthest = np.zeros(num_nodes)
    potential_dispersion_gain = np.zeros(num_nodes)
    
    if centers_set:
        # 计算每个节点到每个控制器的距离
        node_controller_dists = {}
        
        for c in centers_set:
            if c not in node_to_idx:
                continue
            # BFS 计算从控制器 c 到所有节点的距离
            visited = {c: 0}
            queue = [c]
            head = 0
            
            while head < len(queue):
                node = queue[head]
                head += 1
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        visited[neighbor] = visited[node] + 1
                        queue.append(neighbor)
            
            for node, dist in visited.items():
                if node not in node_controller_dists:
                    node_controller_dists[node] = []
                node_controller_dists[node].append(dist)
        
        # 计算每个节点到最远控制器的距离
        max_dist = 0
        for node in node_list:
            if node in node_controller_dists:
                dists = node_controller_dists[node]
                max_dist = max(max_dist, max(dists))
        max_dist = max(max_dist, 1)
        
        for i, node in enumerate(node_list):
            if node in node_controller_dists:
                dists = node_controller_dists[node]
                distance_to_farthest[i] = max(dists) / max_dist
                
                # 潜在分散度增益：选择该节点后，最小控制器间距的变化
                # 如果该节点离所有控制器都很远，则增益高
                min_dist_to_controllers = min(dists)
                potential_dispersion_gain[i] = min_dist_to_controllers / max_dist
    else:
        # 没有控制器时，所有节点的分散度增益相同
        potential_dispersion_gain = np.ones(num_nodes) * 0.5
    
    features = np.stack([
        distance_to_farthest,
        potential_dispersion_gain,
    ], axis=1)
    
    return torch.tensor(features, dtype=torch.float32, device=device)


class DynamicGATPolicy(nn.Module):
    """
    带动态覆盖特征的 GAT 策略网络
    
    改进点:
    1. 动态覆盖特征：每一步都更新节点的覆盖状态特征
    2. 分散度特征：帮助模型理解控制器分布情况
    3. 特征融合：将静态特征和动态特征有效融合
    
    架构:
    - 静态特征: Node2Vec 嵌入 + 规模编码
    - 动态特征: 覆盖特征 (4D) + 分散度特征 (2D) = 6D
    - GAT 层提取高阶结构特征
    - Actor-Critic 架构
    
    Args:
        in_channels: 静态输入特征维度 (默认64)
        hidden_channels: 隐藏层维度 (默认96)
        scale_encoding_dim: 规模编码维度 (默认8)
        dynamic_feature_dim: 动态特征维度 (默认6)
        heads: GAT注意力头数 (默认4)
        num_layers: GAT层数 (默认3)
        dropout: Dropout比例 (默认0.15)
    """
    
    def __init__(
        self,
        in_channels=64,
        hidden_channels=96,
        scale_encoding_dim=8,
        dynamic_feature_dim=6,  # 4 (coverage) + 2 (dispersion)
        heads=4,
        num_layers=3,
        dropout=0.15,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.scale_encoding_dim = scale_encoding_dim
        self.dynamic_feature_dim = dynamic_feature_dim
        self.heads = heads
        self.num_layers = num_layers
        
        # 特征投影器（用于不同维度的Node2Vec特征）
        self.feature_projector = FeatureProjector(output_dim=in_channels)
        
        # 动态特征编码器（将动态特征投影到隐藏维度）
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(dynamic_feature_dim, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels // 2, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ELU(),
        )
        
        # 输入投影层：静态特征 + 规模编码 + 动态特征
        total_input_dim = in_channels + scale_encoding_dim + hidden_channels // 2
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # GAT Layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == num_layers - 1:
                self.gat_layers.append(GATConv(
                    hidden_channels, 
                    hidden_channels, 
                    heads=1, 
                    dropout=dropout,
                    concat=False
                ))
            else:
                self.gat_layers.append(GATConv(
                    hidden_channels, 
                    hidden_channels // heads,
                    heads=heads, 
                    dropout=dropout,
                    concat=True
                ))
            self.layer_norms.append(nn.LayerNorm(hidden_channels))
        
        # 跳跃连接聚合
        self.jump_proj = nn.Linear(hidden_channels * num_layers, hidden_channels)
        self.jump_norm = nn.LayerNorm(hidden_channels)
        
        # 全局上下文注意力
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )
        
        # Actor Head
        self.actor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # Critic Head
        self.critic = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # 温度参数
        self.temperature = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, scale_encoding=None, dynamic_features=None, batch=None, selected_mask=None):
        """
        前向传播
        
        Args:
            x: 静态节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            scale_encoding: 规模编码 [scale_encoding_dim]
            dynamic_features: 动态特征 [num_nodes, dynamic_feature_dim]
            batch: 批次索引
            selected_mask: 已选节点 mask [num_nodes]
            
        Returns:
            probs: 节点选择概率 [num_nodes]
            value: 状态价值估计 [1] 或 [batch_size]
        """
        num_nodes = x.size(0)
        
        # 处理规模编码
        if scale_encoding is None:
            scale_encoding = torch.zeros(self.scale_encoding_dim, device=x.device)
        scale_expanded = scale_encoding.unsqueeze(0).expand(num_nodes, -1)
        
        # 处理动态特征
        if dynamic_features is None:
            dynamic_features = torch.zeros((num_nodes, self.dynamic_feature_dim), device=x.device)
        
        # 编码动态特征
        dynamic_encoded = self.dynamic_encoder(dynamic_features)
        
        # 拼接所有特征
        x_combined = torch.cat([x, scale_expanded, dynamic_encoded], dim=-1)
        
        # 输入投影
        h = self.input_proj(x_combined)
        
        # 收集所有层的输出
        layer_outputs = []
        
        # GAT Layers with residual connections
        for i, (gat, ln) in enumerate(zip(self.gat_layers, self.layer_norms)):
            residual = h
            
            h = gat(h, edge_index)
            h = ln(h)
            h = F.elu(h)
            h = h + residual
            
            layer_outputs.append(h)
            
            if i < self.num_layers - 1:
                h = self.dropout(h)
        
        # 跳跃连接
        jump_concat = torch.cat(layer_outputs, dim=-1)
        h = self.jump_proj(jump_concat)
        h = self.jump_norm(h)
        h = F.elu(h)
        
        # 全局注意力池化
        attention_weights = self.global_attention(h)
        attention_weights = F.softmax(attention_weights, dim=0)
        global_context = (attention_weights * h).sum(dim=0, keepdim=True)
        
        if batch is not None:
            global_context = global_mean_pool(h, batch)
            global_context_expanded = global_context[batch]
        else:
            global_context_expanded = global_context.expand(num_nodes, -1)
        
        # Actor: 节点选择得分
        actor_input = torch.cat([h, global_context_expanded], dim=-1)
        scores = self.actor(actor_input).squeeze(-1)
        
        # 温度缩放
        temp = torch.clamp(self.temperature, min=0.1, max=2.0)
        scores = scores / temp
        
        # 应用 mask
        if selected_mask is not None:
            scores = scores.masked_fill(selected_mask, -1e9)
        
        probs = F.softmax(scores, dim=0)
        
        # Critic
        if batch is not None:
            pooled = global_mean_pool(h, batch)
        else:
            pooled = h.mean(dim=0, keepdim=True)
        value = self.critic(pooled).squeeze(-1)
        
        return probs, value
    
    def get_action(self, x, edge_index, scale_encoding=None, dynamic_features=None, selected_mask=None, deterministic=False):
        """
        获取动作
        
        Args:
            x: 静态节点特征
            edge_index: 边索引
            scale_encoding: 规模编码
            dynamic_features: 动态特征
            selected_mask: 已选节点 mask
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 选择的节点索引
            log_prob: 动作的对数概率
            value: 状态价值估计
            probs: 所有节点的概率
        """
        probs, value = self.forward(
            x, edge_index, scale_encoding, dynamic_features, selected_mask=selected_mask
        )
        
        if deterministic:
            action = torch.argmax(probs)
            log_prob = torch.log(probs[action] + 1e-10)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value, probs
    
    def evaluate_actions(self, x, edge_index, actions, scale_encoding=None, dynamic_features=None, selected_masks=None):
        """
        评估动作（PPO更新时使用）
        """
        probs, values = self.forward(
            x, edge_index, scale_encoding, dynamic_features, selected_mask=selected_masks
        )
        
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy


# ============================================================
# Graph Transformer 编码器
# ============================================================

class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer 层
    
    实现全图注意力机制，能够捕捉长距离依赖关系。
    相比 GAT，Graph Transformer 使用全图注意力而非仅邻居注意力，
    这对控制器分散度指标非常重要。
    
    架构特点:
    - 全图自注意力（可选：带边特征的注意力）
    - 位置编码（可选：拉普拉斯位置编码）
    - 前馈网络 (FFN)
    - 残差连接和层归一化
    
    Args:
        hidden_dim: 隐藏层维度
        num_heads: 注意力头数
        dropout: Dropout 比例
        use_edge_features: 是否使用边特征
        attention_dropout: 注意力 Dropout 比例
    """
    
    def __init__(
        self,
        hidden_dim=96,
        num_heads=4,
        dropout=0.1,
        use_edge_features=False,
        attention_dropout=0.1,
        ffn_multiplier=4,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        # 多头自注意力
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 可选：边特征投影
        self.use_edge_features = use_edge_features
        if use_edge_features:
            self.edge_proj = nn.Linear(hidden_dim, num_heads)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 前馈网络
        ffn_hidden = hidden_dim * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, edge_index=None, edge_attr=None, attention_mask=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, hidden_dim]
            edge_index: 边索引（可选，用于稀疏注意力）
            edge_attr: 边特征（可选）
            attention_mask: 注意力掩码（可选）
            
        Returns:
            x: 更新后的节点特征 [num_nodes, hidden_dim]
        """
        num_nodes = x.size(0)
        
        # Pre-norm
        x_normed = self.norm1(x)
        
        # 计算 Q, K, V
        Q = self.q_proj(x_normed).view(num_nodes, self.num_heads, self.head_dim)
        K = self.k_proj(x_normed).view(num_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(x_normed).view(num_nodes, self.num_heads, self.head_dim)
        
        # 计算注意力分数 [num_nodes, num_nodes, num_heads]
        # Q: [N, H, D], K: [N, H, D] -> attn: [N, N, H]
        attn = torch.einsum('nhd,mhd->nmh', Q, K) * self.scale
        
        # 可选：添加边特征偏置
        if self.use_edge_features and edge_attr is not None and edge_index is not None:
            # 为有边连接的节点对添加偏置
            edge_bias = self.edge_proj(edge_attr)  # [num_edges, num_heads]
            # 创建稀疏偏置矩阵
            for idx in range(edge_index.size(1)):
                i, j = edge_index[0, idx], edge_index[1, idx]
                attn[i, j] = attn[i, j] + edge_bias[idx]
        
        # 可选：应用注意力掩码
        if attention_mask is not None:
            attn = attn.masked_fill(~attention_mask.unsqueeze(-1), -1e9)
        
        # Softmax
        attn = F.softmax(attn, dim=1)  # 对每个 query 的所有 key 归一化
        attn = self.attention_dropout(attn)
        
        # 加权求和
        # attn: [N, N, H], V: [N, H, D] -> out: [N, H, D]
        out = torch.einsum('nmh,mhd->nhd', attn, V)
        out = out.reshape(num_nodes, self.hidden_dim)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        # 残差连接
        x = x + out
        
        # FFN with Pre-norm
        x = x + self.ffn(self.norm2(x))
        
        return x


class LaplacianPositionalEncoding(nn.Module):
    """
    拉普拉斯位置编码
    
    使用图拉普拉斯矩阵的特征向量作为位置编码，
    捕捉图的全局结构信息。
    
    Args:
        hidden_dim: 隐藏维度
        max_k: 使用的特征向量数量
    """
    
    def __init__(self, hidden_dim=96, max_k=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_k = max_k
        
        # 将位置编码投影到隐藏维度
        self.pos_proj = nn.Linear(max_k, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, G, device=None):
        """
        计算拉普拉斯位置编码
        
        Args:
            G: NetworkX 图
            device: 目标设备
            
        Returns:
            pos_encoding: 位置编码 [num_nodes, hidden_dim]
        """
        if device is None:
            device = DEVICE
        
        num_nodes = G.number_of_nodes()
        
        if num_nodes < 2:
            return torch.zeros((num_nodes, self.hidden_dim), device=device)
        
        try:
            # 计算归一化拉普拉斯矩阵
            L = nx.normalized_laplacian_matrix(G).todense()
            L = np.array(L)
            
            # 计算特征值和特征向量
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            
            # 取前 k 个特征向量（跳过第一个，因为它对应特征值0）
            k = min(self.max_k, num_nodes - 1)
            pos_enc = eigenvectors[:, 1:k+1]  # [num_nodes, k]
            
            # 如果特征向量不足，用零填充
            if pos_enc.shape[1] < self.max_k:
                padding = np.zeros((num_nodes, self.max_k - pos_enc.shape[1]))
                pos_enc = np.concatenate([pos_enc, padding], axis=1)
            
            pos_enc = torch.tensor(pos_enc, dtype=torch.float32, device=device)
            
        except Exception:
            # 如果计算失败，返回随机初始化的编码
            pos_enc = torch.randn((num_nodes, self.max_k), device=device) * 0.1
        
        # 投影到隐藏维度
        pos_encoding = self.pos_proj(pos_enc)
        pos_encoding = self.norm(pos_encoding)
        
        return pos_encoding


class GraphTransformerPolicy(nn.Module):
    """
    Graph Transformer 策略网络
    
    使用全图注意力机制代替 GAT 的局部注意力，
    能够捕捉长距离依赖关系，对控制器分散度优化特别有效。
    
    架构:
    - 静态特征: Node2Vec 嵌入 + 规模编码
    - 动态特征: 覆盖特征 + 分散度特征
    - 拉普拉斯位置编码
    - N 层 Graph Transformer
    - Actor-Critic 架构
    
    Args:
        in_channels: 静态输入特征维度 (默认64)
        hidden_channels: 隐藏层维度 (默认96)
        scale_encoding_dim: 规模编码维度 (默认8)
        dynamic_feature_dim: 动态特征维度 (默认6)
        num_heads: 注意力头数 (默认4)
        num_layers: Transformer层数 (默认3)
        dropout: Dropout比例 (默认0.15)
        use_laplacian_pe: 是否使用拉普拉斯位置编码
        max_pe_k: 位置编码使用的特征向量数量
    """
    
    def __init__(
        self,
        in_channels=64,
        hidden_channels=96,
        scale_encoding_dim=8,
        dynamic_feature_dim=6,
        num_heads=4,
        num_layers=3,
        dropout=0.15,
        use_laplacian_pe=True,
        max_pe_k=8,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.scale_encoding_dim = scale_encoding_dim
        self.dynamic_feature_dim = dynamic_feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_laplacian_pe = use_laplacian_pe
        
        # 特征投影器
        self.feature_projector = FeatureProjector(output_dim=in_channels)
        
        # 动态特征编码器
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(dynamic_feature_dim, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels // 2, hidden_channels // 2),
        )
        
        # 拉普拉斯位置编码
        if use_laplacian_pe:
            self.laplacian_pe = LaplacianPositionalEncoding(hidden_channels, max_pe_k)
        
        # 输入投影层
        total_input_dim = in_channels + scale_encoding_dim + hidden_channels // 2
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Graph Transformer 层
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim=hidden_channels,
                num_heads=num_heads,
                dropout=dropout,
                attention_dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # 跳跃连接
        self.jump_proj = nn.Linear(hidden_channels * num_layers, hidden_channels)
        self.jump_norm = nn.LayerNorm(hidden_channels)
        
        # 全局上下文
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )
        
        # Actor Head
        self.actor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # Critic Head
        self.critic = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # 温度参数
        self.temperature = nn.Parameter(torch.ones(1))
        self.dropout_layer = nn.Dropout(dropout)
        
        # 缓存位置编码
        self._cached_pe = None
        self._cached_pe_graph_hash = None
    
    def forward(
        self, 
        x, 
        edge_index=None,  # 保留兼容性，但 Transformer 不需要
        scale_encoding=None, 
        dynamic_features=None, 
        laplacian_pe=None,  # 外部传入的位置编码
        batch=None, 
        selected_mask=None
    ):
        """
        前向传播
        
        Args:
            x: 静态节点特征 [num_nodes, in_channels]
            edge_index: 边索引（保留兼容性）
            scale_encoding: 规模编码
            dynamic_features: 动态特征
            laplacian_pe: 拉普拉斯位置编码
            batch: 批次索引
            selected_mask: 已选节点 mask
            
        Returns:
            probs: 节点选择概率
            value: 状态价值估计
        """
        num_nodes = x.size(0)
        
        # 处理规模编码
        if scale_encoding is None:
            scale_encoding = torch.zeros(self.scale_encoding_dim, device=x.device)
        scale_expanded = scale_encoding.unsqueeze(0).expand(num_nodes, -1)
        
        # 处理动态特征
        if dynamic_features is None:
            dynamic_features = torch.zeros((num_nodes, self.dynamic_feature_dim), device=x.device)
        dynamic_encoded = self.dynamic_encoder(dynamic_features)
        
        # 拼接所有输入特征
        x_combined = torch.cat([x, scale_expanded, dynamic_encoded], dim=-1)
        
        # 输入投影
        h = self.input_proj(x_combined)
        
        # 添加位置编码
        if self.use_laplacian_pe and laplacian_pe is not None:
            h = h + laplacian_pe
        
        # Graph Transformer 层
        layer_outputs = []
        for transformer in self.transformer_layers:
            h = transformer(h, edge_index)
            layer_outputs.append(h)
        
        # 跳跃连接
        jump_concat = torch.cat(layer_outputs, dim=-1)
        h = self.jump_proj(jump_concat)
        h = self.jump_norm(h)
        h = F.gelu(h)
        
        # 全局上下文
        attention_weights = self.global_attention(h)
        attention_weights = F.softmax(attention_weights, dim=0)
        global_context = (attention_weights * h).sum(dim=0, keepdim=True)
        
        if batch is not None:
            global_context = global_mean_pool(h, batch)
            global_context_expanded = global_context[batch]
        else:
            global_context_expanded = global_context.expand(num_nodes, -1)
        
        # Actor
        actor_input = torch.cat([h, global_context_expanded], dim=-1)
        scores = self.actor(actor_input).squeeze(-1)
        
        # 温度缩放
        temp = torch.clamp(self.temperature, min=0.1, max=2.0)
        scores = scores / temp
        
        # 应用 mask
        if selected_mask is not None:
            scores = scores.masked_fill(selected_mask, -1e9)
        
        probs = F.softmax(scores, dim=0)
        
        # Critic
        if batch is not None:
            pooled = global_mean_pool(h, batch)
        else:
            pooled = h.mean(dim=0, keepdim=True)
        value = self.critic(pooled).squeeze(-1)
        
        return probs, value
    
    def get_action(
        self, 
        x, 
        edge_index=None, 
        scale_encoding=None, 
        dynamic_features=None,
        laplacian_pe=None,
        selected_mask=None, 
        deterministic=False
    ):
        """获取动作"""
        probs, value = self.forward(
            x, edge_index, scale_encoding, dynamic_features, laplacian_pe, selected_mask=selected_mask
        )
        
        if deterministic:
            action = torch.argmax(probs)
            log_prob = torch.log(probs[action] + 1e-10)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value, probs
    
    def evaluate_actions(
        self, 
        x, 
        edge_index, 
        actions, 
        scale_encoding=None, 
        dynamic_features=None,
        laplacian_pe=None,
        selected_masks=None
    ):
        """评估动作"""
        probs, values = self.forward(
            x, edge_index, scale_encoding, dynamic_features, laplacian_pe, selected_mask=selected_masks
        )
        
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy


def get_laplacian_pe(G, hidden_dim=96, max_k=8, device=None):
    """
    计算图的拉普拉斯位置编码（独立函数，用于预计算）
    
    Args:
        G: NetworkX 图
        hidden_dim: 目标维度
        max_k: 使用的特征向量数量
        device: 目标设备
        
    Returns:
        pos_encoding: 位置编码 [num_nodes, hidden_dim]
    """
    if device is None:
        device = DEVICE
    
    num_nodes = G.number_of_nodes()
    
    if num_nodes < 2:
        return torch.zeros((num_nodes, hidden_dim), device=device)
    
    try:
        # 计算归一化拉普拉斯矩阵
        L = nx.normalized_laplacian_matrix(G).todense()
        L = np.array(L)
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # 取前 k 个非零特征向量
        k = min(max_k, num_nodes - 1)
        pos_enc = eigenvectors[:, 1:k+1]
        
        # 填充
        if pos_enc.shape[1] < max_k:
            padding = np.zeros((num_nodes, max_k - pos_enc.shape[1]))
            pos_enc = np.concatenate([pos_enc, padding], axis=1)
        
        # 简单线性投影到目标维度
        proj = np.random.randn(max_k, hidden_dim) * 0.02
        pos_enc = pos_enc @ proj
        
        pos_enc = torch.tensor(pos_enc, dtype=torch.float32, device=device)
        
    except Exception:
        pos_enc = torch.randn((num_nodes, hidden_dim), device=device) * 0.1
    
    return pos_enc
