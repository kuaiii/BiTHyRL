# 导入必要的库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数库
import torch.optim as optim  # 优化器
import numpy as np  # 数值计算库
import igraph as ig  # 图论库
import random  # 随机数生成
import math  # 数学函数
from typing import List, Tuple, Dict, Optional, Set  # 类型注解
from collections import defaultdict  # 默认字典
import matplotlib.pyplot as plt  # 绘图库
import matplotlib  # 绘图配置
from tqdm import tqdm  # 进度条
import networkx as nx  # 网络分析库

# 设置matplotlib中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
matplotlib.use('TkAgg')  # 使用TkAgg后端

class SmartTROEnvironment:
    """SmartTRO环境类，用于处理IoT网络拓扑和奖励计算
    
    属性:
        num_nodes: 网络节点数量
        edge_density: 边密度参数
        comm_radius: 通信半径
        area_size: 区域大小
        max_degree: 最大节点度数
        max_actions: 最大动作数量
        node_coords: 节点坐标字典
        current_graph: 当前网络图
        initial_robustness: 初始鲁棒性
        actions_taken: 已执行的动作数
    """
    def __init__(self, num_nodes: int = 100, edge_density: int = 2, 
             comm_radius: float = 200.0, area_size: Tuple[int, int] = (500, 500),
             max_actions: int = None):
        self.num_nodes = num_nodes
        self.edge_density = edge_density
        self.comm_radius = comm_radius
        self.area_size = area_size
        self.max_degree = min(20 + edge_density * 5, num_nodes - 1)
        
        # 设置最大动作数(论文中的K参数)
        if max_actions is None:
            # K = p * |E|, 论文公式(6)中p设为0.2
            self.max_actions = int(0.2 * (self.num_nodes * self.edge_density))
        else:
            self.max_actions = max_actions
            
        # 节点坐标和当前网络状态
        self.node_coords = None
        self.current_graph = None
        self.initial_robustness = None
        self.actions_taken = 0

    def generate_scale_free_iot_topology(self) -> nx.Graph:
        """生成符合论文改进BA模型的尺度无关IoT拓扑"""
        # 随机部署节点
        self.node_coords = {}
        for i in range(self.num_nodes):
            x = random.uniform(0, self.area_size[0])
            y = random.uniform(0, self.area_size[1])
            self.node_coords[i] = (x, y)
        
        # 创建图
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_nodes))
        
        # 按距离中心的顺序添加节点（论文步骤1）
        center = (self.area_size[0] / 2, self.area_size[1] / 2)
        nodes_by_distance = sorted(range(self.num_nodes), 
                                key=lambda i: self._euclidean_distance(self.node_coords[i], center))
        
        # 初始连接前几个节点
        for i in range(min(self.edge_density + 1, self.num_nodes)):
            for j in range(i + 1, min(self.edge_density + 1, self.num_nodes)):
                if self._can_connect(nodes_by_distance[i], nodes_by_distance[j]):
                    graph.add_edge(nodes_by_distance[i], nodes_by_distance[j])
        
        # 优先连接机制添加剩余节点（论文步骤2-3）
        for i in range(self.edge_density + 1, self.num_nodes):
            node = nodes_by_distance[i]
            candidates = []
            
            # 找到通信范围内的候选节点
            for existing_node in graph.nodes():
                if (existing_node != node and 
                    self._can_connect(node, existing_node) and
                    graph.degree(existing_node) < self.max_degree):
                    candidates.append(existing_node)
            
            if not candidates:
                continue
                
            # 按度数加权选择连接（轮盘赌法）
            degrees = [graph.degree(n) + 1 for n in candidates]  # +1避免度数为0
            total_weight = sum(degrees)
            
            connections_made = 0
            while connections_made < self.edge_density and candidates:
                # 轮盘赌选择
                rand_val = random.uniform(0, total_weight)
                cumsum = 0
                selected_node = None
                
                for j, candidate in enumerate(candidates):
                    cumsum += degrees[j]
                    if cumsum >= rand_val:
                        selected_node = candidate
                        break
                
                if selected_node is not None:
                    graph.add_edge(node, selected_node)
                    connections_made += 1
                    
                    # 移除已连接的候选节点
                    idx = candidates.index(selected_node)
                    candidates.pop(idx)
                    total_weight -= degrees[idx]
                    degrees.pop(idx)
                    
                    # 如果目标节点达到最大度数，移除它
                    if graph.degree(selected_node) >= self.max_degree:
                        if selected_node in candidates:
                            idx = candidates.index(selected_node)
                            candidates.pop(idx)
                            total_weight -= degrees[idx]
                            degrees.pop(idx)
                else:
                    break
        
        return graph

    def _euclidean_distance(self, coord1: Tuple[float, float], 
                        coord2: Tuple[float, float]) -> float:
        """计算欧几里得距离"""
        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])** 2)

    def _can_connect(self, node1: int, node2: int) -> bool:
        """检查两个节点是否能连接（距离约束）"""
        if node1 == node2:
            return False
        dist = self._euclidean_distance(self.node_coords[node1], self.node_coords[node2])
        return dist <= self.comm_radius

    def reset(self) -> Dict:
        """重置环境"""
        self.current_graph = self.generate_scale_free_iot_topology()
        # 确保图是连通的
        while not nx.is_connected(self.current_graph) or self.current_graph.number_of_edges() < 2:
            self.current_graph = self.generate_scale_free_iot_topology()
        
        self.initial_robustness = self.calculate_robustness(self.current_graph)
        self.actions_taken = 0
        return self.get_state()

    def get_state(self) -> Dict:
        """获取当前状态（包含邻接矩阵和度差矩阵）"""
        adj_matrix = nx.adjacency_matrix(self.current_graph).todense()
        
        # 计算度差矩阵（论文3.2节）
        degrees = dict(self.current_graph.degree())
        degree_diff_matrix = np.zeros((self.num_nodes, self.num_nodes))
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.current_graph.has_edge(i, j):  # 只计算存在边的度差
                    degree_diff_matrix[i][j] = abs(degrees[i] - degrees[j])
        
        return {
            'adjacency_matrix': np.array(adj_matrix),
            'degree_diff_matrix': degree_diff_matrix,
            'edge_list': list(self.current_graph.edges()),
            'node_coords': self.node_coords,
            'degrees': degrees
        }

    def calculate_robustness(self, graph: nx.Graph) -> float:
        """计算网络鲁棒性（HDA攻击策略，论文3.1节）"""
        if graph.number_of_nodes() == 0:
            return 0.0
        
        g_copy = graph.copy()
        total_nodes = graph.number_of_nodes()
        robustness_sum = 0.0
        
        # 初始最大连通子图
        if g_copy.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(g_copy), key=len)
            robustness_sum += len(largest_cc) / total_nodes
        
        # HDA攻击：每次移除度数最大的节点
        for _ in range(total_nodes):
            if g_copy.number_of_nodes() == 0:
                break
                
            # 找到度数最大的节点
            max_degree_node = max(g_copy.degree(), key=lambda x: x[1])[0]
            g_copy.remove_node(max_degree_node)
            
            # 计算最大连通子图大小
            if g_copy.number_of_nodes() > 0:
                try:
                    largest_cc = max(nx.connected_components(g_copy), key=len)
                    robustness_sum += len(largest_cc) / total_nodes
                except:
                    # 没有连通分量
                    pass
        
        return robustness_sum / (total_nodes + 1)  # 归一化因子（论文公式1）

    def step(self, action: Dict) -> Tuple[Dict, float, bool]:
        """执行重连操作（论文定义1和Algorithm 2）"""
        self.actions_taken += 1
        edge1 = action['edge1']
        edge2 = action['edge2']
        operation_type = action['operation_type']
        
        # 检查两条边是否有公共节点（论文重连操作前提）
        if len(set(edge1) & set(edge2)) > 0:
            return self.get_state(), -0.1, False  # 无效操作惩罚
        
        # 执行重连操作
        new_graph = self.current_graph.copy()
        reward = 0.0
        
        if operation_type == 1:  # 第一种重连方式：(i,j)和(m,n) → (i,m)和(j,n)
            i, j = edge1
            m, n = edge2
            if (self._can_connect(i, m) and self._can_connect(j, n)):
                new_graph.remove_edge(*edge1)
                new_graph.remove_edge(*edge2)
                new_graph.add_edge(i, m)
                new_graph.add_edge(j, n)
        elif operation_type == 2:  # 第二种重连方式：(i,j)和(m,n) → (i,n)和(j,m)
            i, j = edge1
            m, n = edge2
            if (self._can_connect(i, n) and self._can_connect(j, m)):
                new_graph.remove_edge(*edge1)
                new_graph.remove_edge(*edge2)
                new_graph.add_edge(i, n)
                new_graph.add_edge(j, m)
        else:  # 不改变边
            return self.get_state(), 0.0, False
        
        # 检查网络连通性并计算奖励
        if nx.is_connected(new_graph):
            old_robustness = self.calculate_robustness(self.current_graph)
            new_robustness = self.calculate_robustness(new_graph)
            reward = new_robustness - old_robustness
            
            if reward > 0:
                self.current_graph = new_graph
        else:
            reward = -0.1  # 惩罚导致网络不连通的动作
        
        # 终止条件：达到最大操作次数或鲁棒性不再提升
        done = (abs(reward) < 1e-6) or (self.actions_taken >= self.max_actions)
        return self.get_state(), reward, done

class GraphConvLayer(nn.Module):
    """图卷积层（论文4.3.1节）"""
    def __init__(self, in_features: int, out_features: int):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 节点特征变换矩阵（论文公式10）
        self.W3 = nn.Linear(in_features, out_features, bias=False)  # W3: 自身特征
        self.W4 = nn.Linear(in_features, out_features, bias=False)  # W4: 邻居特征
        
        # 边特征变换矩阵（论文公式11）
        self.W5 = nn.Linear(in_features, out_features, bias=False)  # W5: 边自身特征
        self.W6 = nn.Linear(in_features, out_features, bias=False)  # W6: 节点i特征
        self.W7 = nn.Linear(in_features, out_features, bias=False)  # W7: 节点j特征
        
        # 边门控机制（论文公式10中的η_ij^l）
        self.edge_gate = nn.Linear(in_features, 1, bias=False)
        
        # 批归一化
        self.bn_node = nn.BatchNorm1d(out_features)
        self.bn_edge = nn.BatchNorm1d(out_features)
        
        # 小值ε避免除零
        self.epsilon = 1e-8
    
    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, num_nodes, _ = node_features.shape
        
        # 节点特征更新（论文公式10）
        node_self = self.W3(node_features)  # W3*v_i^l
        
        # 计算边门控值（论文公式10中的η_ij^l）
        edge_gates = torch.sigmoid(self.edge_gate(edge_features))  # σ(e_ij^l)
        
        # 按照论文公式10计算归一化的门控值
        normalized_gates = torch.zeros_like(edge_gates)
        for b in range(batch_size):
            for i in range(num_nodes):
                neighbors = torch.nonzero(adjacency[b, i]).squeeze(-1)
                if len(neighbors.shape) == 0 and neighbors.numel() > 0:  # 只有一个邻居
                    neighbors = neighbors.unsqueeze(0)
                
                if len(neighbors) > 0:
                    gate_sum = torch.sum(edge_gates[b, i, neighbors]) + self.epsilon
                    for j in neighbors:
                        normalized_gates[b, i, j] = edge_gates[b, i, j] / gate_sum
        
        # 邻居特征聚合（修正维度）
        neighbor_features = self.W4(node_features)  # 形状：[batch, num_nodes, feature_dim]
        aggregated = torch.zeros_like(node_self)  # 形状：[batch, num_nodes, feature_dim]
        
        for b in range(batch_size):
            for i in range(num_nodes):
                neighbors = torch.nonzero(adjacency[b, i] > 0, as_tuple=False).view(-1)
                
                if len(neighbors) > 0:
                    # 提取邻居特征并加权求和
                    gates = normalized_gates[b, i, neighbors].view(-1, 1)  # 明确设为 [n_neighbors, 1]
                    feats = neighbor_features[b, neighbors]  # [n_neighbors, feature_dim]
                    weighted_feats = gates * feats  # [n_neighbors, feature_dim]
                    weighted_sum = torch.sum(weighted_feats, dim=0)  # [feature_dim]
                    
                    # 强制压缩所有多余维度，确保为1维向量
                    if weighted_sum.dim() != 1:
                        weighted_sum = weighted_sum.flatten()
                    
                    # 若压缩后为空（极端情况），用零向量填充
                    if weighted_sum.numel() == 0:
                        weighted_sum = torch.zeros(self.out_features, device=weighted_sum.device)
        
                    aggregated[b, i] = weighted_sum  # 此时维度匹配 [feature_dim]
        
        # 残差连接和激活：v_i^{l+1} = v_i^l + ReLU(BN(W_3^l*v_i^l + sum_j(η_ij^l⊙W_4^l*v_j^l)))
        combined = node_self + aggregated
        normalized = self.bn_node(combined.view(-1, self.out_features)).view(batch_size, num_nodes, self.out_features)
        new_node_features = node_features + F.relu(normalized)
        
        # 边特征更新（论文公式11）
        edge_self = self.W5(edge_features)  # W5*e_ij^l
        
        # 为边特征计算节点i和j的特征影响
        new_edge_features = edge_features.clone()
        for b in range(batch_size):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if adjacency[b, i, j] > 0:  # 存在边
                        node_i_feat = self.W6(node_features[b, i])  # W6*v_i^l
                        node_j_feat = self.W7(node_features[b, j])  # W7*v_j^l
                        
                        # e_ij^{l+1} = e_ij^l + ReLU(BN(W_5^l*e_ij^l + W_6^l*v_i^l + W_7^l*v_j^l))
                        edge_update = edge_self[b, i, j] + node_i_feat + node_j_feat
                        edge_update = edge_update.unsqueeze(0)  # 添加批次维度以适应BN
                        edge_update = edge_update.view(-1)
                        if edge_update.shape[0] != self.out_features:
                            # 自动补齐或抛出警告
                            edge_update = F.pad(edge_update, (0, self.out_features - edge_update.shape[0]))
                        
                        new_edge_features[b, i, j] = edge_features[b, i, j] + edge_update
        
        return new_node_features, new_edge_features


class GraphConvNet(nn.Module):
    """图卷积网络（论文4.3.1节）"""
    def __init__(self, feature_dim: int = 128, num_layers: int = 10):
        super(GraphConvNet, self).__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # 输入层：节点特征和边特征初始化（论文公式8-9）
        self.node_input = nn.Linear(1, feature_dim)  # 邻接矩阵→节点特征
        self.edge_input = nn.Linear(1, feature_dim)  # 度差矩阵→边特征
        
        # 图卷积层（论文表1中L_conv=10）
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(feature_dim, feature_dim) for _ in range(num_layers)
        ])
    
    def forward(self, adjacency_matrix: torch.Tensor, degree_diff_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        batch_size, num_nodes, _ = adjacency_matrix.shape
        
        # 初始化节点特征：v_i^0 = W1*a_i + b1（论文公式8）
        node_features = torch.zeros(batch_size, num_nodes, self.feature_dim).to(adjacency_matrix.device)
        for b in range(batch_size):
            for i in range(num_nodes):
                # 提取邻接矩阵的第i列（节点i的连接特征）
                node_col = adjacency_matrix[b, :, i].unsqueeze(-1)  # 形状：[num_nodes, 1]
                # 通过线性层转换为节点特征（形状：[num_nodes, feature_dim]）
                col_feat = self.node_input(node_col)  # 论文中W1的作用
                # 对列特征聚合（例如取平均）得到节点i的特征向量（形状：[feature_dim]）
                if col_feat.numel() == 0:
                    node_features[b, i] = torch.zeros(self.feature_dim).to(adjacency_matrix.device)
                else:
                    node_features[b, i] = torch.mean(col_feat, dim=0)
        
        # 初始化边特征：e_ij^0 = (W2*d_ij + b2)*δ_ij（论文公式9）
        edge_features = torch.zeros(batch_size, num_nodes, num_nodes, self.feature_dim).to(adjacency_matrix.device)
        for b in range(batch_size):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    delta_ij = adjacency_matrix[b, i, j]
                    if delta_ij > 0:
                        edge_input = degree_diff_matrix[b, i, j].unsqueeze(-1)
                        edge_features[b, i, j] = self.edge_input(edge_input) * delta_ij
        
        # 应用图卷积层
        for conv_layer in self.conv_layers:
            node_features, edge_features = conv_layer(node_features, edge_features, adjacency_matrix)
        
        # 提取边嵌入字典
        edge_embedding_dict = {}
        for b in range(batch_size):
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if adjacency_matrix[b, i, j] > 0:
                        edge_embedding_dict[(i, j)] = edge_features[b, i, j].detach().cpu()
        
        # 计算图嵌入（全局平均池化）
        graph_embedding = torch.mean(edge_features, dim=[1, 2])
        
        return edge_features, graph_embedding, edge_embedding_dict


class PolicyNetwork(nn.Module):
    """策略网络（论文4.3.2节） - 修复版本"""
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 256):
        super(PolicyNetwork, self).__init__()
        self.feature_dim = feature_dim
        
        # 第一条边选择网络：p_fir(e_fir|s_t)（论文公式14）
        self.first_edge_mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),  # 边特征+图嵌入
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 第二条边选择网络：p_sec(e_sec|e_fir, s_t)（论文公式15）
        self.second_edge_mlp = nn.Sequential(
            nn.Linear(3 * feature_dim, hidden_dim),  # 边特征+图嵌入+第一条边特征
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward_first_edge(self, edge_features: Dict[Tuple[int, int], torch.Tensor], 
                    graph_embedding: torch.Tensor,
                    edge_list: List[Tuple[int, int]]) -> Tuple[torch.Tensor, List]:
        """选择第一条边 - 修复版本"""
        if not edge_list or not edge_features:
            return torch.tensor([]), []
            
        # 修复：正确构建边输入特征
        edge_inputs = []
        valid_edges = []
        
        for edge in edge_list:
            edge_feat = None
            # 尝试正序和反序查找边
            if edge in edge_features:
                edge_feat = edge_features[edge]
            elif (edge[1], edge[0]) in edge_features:
                edge_feat = edge_features[(edge[1], edge[0])]
            
            if edge_feat is not None:
                # 确保维度正确：边特征 + 图嵌入
                combined_feat = torch.cat([edge_feat, graph_embedding], dim=0)
                edge_inputs.append(combined_feat)
                valid_edges.append(edge)
        
        if not edge_inputs:
            return torch.tensor([]), []
            
        # 转换为批次张量
        edge_inputs = torch.stack(edge_inputs)
        
        # 通过MLP获取每条边的分数
        edge_scores = self.first_edge_mlp(edge_inputs).squeeze(-1)
        
        # 应用softmax获取概率分布
        edge_probs = F.softmax(edge_scores, dim=0)
        
        return edge_probs, valid_edges

    def forward_second_edge(self, first_edge: Tuple[int, int], 
                        first_edge_feat: torch.Tensor,
                        edge_features: Dict[Tuple[int, int], torch.Tensor],
                        graph_embedding: torch.Tensor,
                        edge_list: List[Tuple[int, int]]) -> Tuple[torch.Tensor, List]:
        """选择第二条边 - 修复版本"""
        # 过滤候选边（不能与第一条边有公共节点）
        first_edge_nodes = set(first_edge)
        candidate_edges = []
        
        for edge in edge_list:
            edge_nodes = set(edge)
            if len(first_edge_nodes & edge_nodes) == 0:  # 没有公共节点
                candidate_edges.append(edge)
        
        if not candidate_edges:
            return torch.tensor([]), []
            
        # 修复：正确构建边输入特征
        edge_inputs = []
        valid_candidates = []
        
        for edge in candidate_edges:
            edge_feat = None
            # 尝试正序和反序查找边
            if edge in edge_features:
                edge_feat = edge_features[edge]
            elif (edge[1], edge[0]) in edge_features:
                edge_feat = edge_features[(edge[1], edge[0])]
            
            if edge_feat is not None:
                # 确保维度正确：边特征 + 第一条边特征 + 图嵌入
                combined_feat = torch.cat([
                    edge_feat.view(-1),
                    first_edge_feat.view(-1),
                    graph_embedding.view(-1)
                ], dim=0)
                edge_inputs.append(combined_feat)
                valid_candidates.append(edge)
        
        if not edge_inputs:
            return torch.tensor([]), []
            
        # 转换为批次张量
        edge_inputs = torch.stack(edge_inputs)
        
        # 通过MLP获取每条边的分数
        edge_scores = self.second_edge_mlp(edge_inputs).squeeze(-1)
        
        # 应用softmax获取概率分布
        edge_probs = F.softmax(edge_scores, dim=0)
        
        return edge_probs, valid_candidates

    def sample_action(self, edge_features: Dict[Tuple[int, int], torch.Tensor],
                graph_embedding: torch.Tensor,
                edge_list: List[Tuple[int, int]]) -> Tuple[Dict, List[float]]:
        """采样动作 - 修复版本"""
        # 获取第一条边的概率分布
        first_edge_probs, valid_first_edges = self.forward_first_edge(edge_features, graph_embedding, edge_list)
        
        if len(first_edge_probs) == 0:
            return None, []
            
        # 根据概率分布采样第一条边
        first_edge_idx = torch.multinomial(first_edge_probs, 1).item()
        first_edge = valid_first_edges[first_edge_idx]
        first_edge_prob = first_edge_probs[first_edge_idx].item()
        
        # 获取第一条边的特征
        if first_edge in edge_features:
            first_edge_feat = edge_features[first_edge]
        elif (first_edge[1], first_edge[0]) in edge_features:
            first_edge_feat = edge_features[(first_edge[1], first_edge[0])]
        else:
            return None, []
        
        # 获取第二条边的概率分布
        second_edge_probs, candidate_edges = self.forward_second_edge(
            first_edge, first_edge_feat, edge_features, graph_embedding, edge_list)
        
        if len(second_edge_probs) == 0 or not candidate_edges:
            return None, []
            
        # 根据概率分布采样第二条边
        second_edge_idx = torch.multinomial(second_edge_probs, 1).item()
        second_edge = candidate_edges[second_edge_idx]
        second_edge_prob = second_edge_probs[second_edge_idx].item()
        
        # 构建动作
        action = {
            'edge1': first_edge,
            'edge2': second_edge,
            'operation_type': 0  # 将由ActionComparator决定
        }
        
        log_probs = [math.log(max(first_edge_prob, 1e-8)), math.log(max(second_edge_prob, 1e-8))]
        
        return action, log_probs


class ActionComparator:
    """动作比较器（论文Algorithm 2）"""
    def __init__(self, env: SmartTROEnvironment):
        self.env = env

    def compare_actions(self, edge1: Tuple[int, int], edge2: Tuple[int, int]) -> Dict:
        """比较不同重连操作的效果并选择最佳操作"""
        i, j = edge1
        m, n = edge2
        
        # 检查是否有共同节点
        if len(set(edge1) & set(edge2)) > 0:
            return {'edge1': edge1, 'edge2': edge2, 'operation_type': 3}
        
        # 候选动作集
        candidate_actions = []
        
        # 检查操作类型1：(i,j)和(m,n) → (i,m)和(j,n)
        if (self.env._can_connect(i, m) and self.env._can_connect(j, n)):
            candidate_actions.append({
                'edge1': edge1,
                'edge2': edge2,
                'operation_type': 1,
                'new_edges': [(i, m), (j, n)]
            })
        
        # 检查操作类型2：(i,j)和(m,n) → (i,n)和(j,m)
        if (self.env._can_connect(i, n) and self.env._can_connect(j, m)):
            candidate_actions.append({
                'edge1': edge1,
                'edge2': edge2,
                'operation_type': 2,
                'new_edges': [(i, n), (j, m)]
            })
        
        # 如果没有可行的操作，保持原始边
        if not candidate_actions:
            return {'edge1': edge1, 'edge2': edge2, 'operation_type': 3}
        
        # 评估每个候选动作
        best_action = None
        best_robustness = -float('inf')
        
        for action in candidate_actions:
            # 创建新图进行测试
            test_graph = self.env.current_graph.copy()
            
            # 应用重连操作
            test_graph.remove_edge(*edge1)
            test_graph.remove_edge(*edge2)
            test_graph.add_edge(*action['new_edges'][0])
            test_graph.add_edge(*action['new_edges'][1])
            
            # 检查是否保持连通性
            if nx.is_connected(test_graph):
                new_robustness = self.env.calculate_robustness(test_graph)
                current_robustness = self.env.calculate_robustness(self.env.current_graph)
                robustness_increase = new_robustness - current_robustness
                
                if robustness_increase > best_robustness:
                    best_robustness = robustness_increase
                    best_action = action
        
        # 如果没有找到能提高鲁棒性的操作，则保持原始边
        if best_action is None:
            return {'edge1': edge1, 'edge2': edge2, 'operation_type': 3}
        
        return best_action
    
    
class SmartTRO:
    """SmartTRO算法主类 - 修复版本"""
    def __init__(self, num_nodes: int = 100, edge_density: int = 2,
        feature_dim: int = 128, num_layers: int = 10,
        learning_rate: float = 0.001, gamma: float = 0.99,
        num_agents: int = 8):
        
        self.num_nodes = num_nodes
        self.edge_density = edge_density
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.num_agents = num_agents

       # 创建环境
        self.env = SmartTROEnvironment(num_nodes, edge_density)
        
        # 为每个智能体创建环境
        self.envs = [SmartTROEnvironment(num_nodes, edge_density) for _ in range(num_agents)]
        
        # 创建神经网络
        self.gcn = GraphConvNet(feature_dim, num_layers)
        self.policy_net = PolicyNetwork(feature_dim)
        self.action_comparators = [ActionComparator(env) for env in self.envs]
        
        # 创建优化器
        self.optimizer = optim.Adam(
            list(self.gcn.parameters()) + list(self.policy_net.parameters()), 
            lr=learning_rate
        )
        
        # 用于跟踪训练过程
        self.episode_rewards = []
        self.episode_lengths = []

    def preprocess_state(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """预处理状态为神经网络输入"""
        adjacency = torch.tensor(state['adjacency_matrix']).float().unsqueeze(0)
        degree_diff = torch.tensor(state['degree_diff_matrix']).float().unsqueeze(0)
        return adjacency, degree_diff

    def select_action(self, state: Dict, env_idx: int) -> Tuple[Dict, List[float]]:
        adjacency, degree_diff = self.preprocess_state(state)

        with torch.no_grad():
            edge_features, graph_embedding, edge_embedding_dict = self.gcn(adjacency, degree_diff)
            action, log_probs = self.policy_net.sample_action(
                edge_embedding_dict, graph_embedding[0], state['edge_list']
            )

        if action is None:
            return None, []

        # 使用当前环境对应的动作比较器
        final_action = self.action_comparators[env_idx].compare_actions(
            action['edge1'], action['edge2']
        )

        return final_action, log_probs

    def collect_trajectories(self) -> List[Dict]:
        """并行收集多个智能体的轨迹"""
        trajectories = []
        
        for i in range(self.num_agents):
            state = self.envs[i].reset()
            done = False
            trajectory = {
                'states': [],
                'actions': [],
                'log_probs': [],
                'rewards': [],
                'total_reward': 0,
                'steps': 0
            }
            
            while not done:
                action, log_probs = self.select_action(state, env_idx=i)
                
                if action is None:
                    break
                    
                next_state, reward, done = self.envs[i].step(action)
                
                trajectory['states'].append(state)
                trajectory['actions'].append(action)
                trajectory['log_probs'].extend(log_probs)
                trajectory['rewards'].append(reward)
                trajectory['total_reward'] += reward
                trajectory['steps'] += 1
                
                state = next_state
                
                if trajectory['steps'] >= self.envs[i].max_actions:
                    break
            
            trajectories.append(trajectory)
        
        return trajectories

    def train(self, num_iterations: int = 1000) -> None:
        """训练模型 - 修复版本"""
        for iteration in range(num_iterations):
            # 运行N个智能体并行收集轨迹
            trajectories = self.collect_trajectories()
            
            # 计算策略梯度更新
            self.optimizer.zero_grad()
            total_loss = 0
            num_updates = 0
            
            for traj in trajectories:
                if len(traj['rewards']) == 0:
                    continue
                    
                # 计算折扣回报
                returns = []
                R = 0
                for r in reversed(traj['rewards']):
                    R = r + self.gamma * R
                    returns.insert(0, R)
                
                # 计算基线（所有轨迹的平均回报）
                if returns:
                    baseline = sum(returns) / len(returns)
                    
                    # 计算策略梯度
                    for t in range(len(returns)):
                        if t < len(traj['log_probs']):
                            advantage = returns[t] - baseline
                            log_prob = traj['log_probs'][t]
                            
                            # 确保log_prob是张量
                            if not isinstance(log_prob, torch.Tensor):
                                log_prob = torch.tensor(log_prob, requires_grad=True)
                            
                            # 策略梯度损失（负号因为要最大化）
                            loss = -log_prob * advantage
                            total_loss += loss
                            num_updates += 1
            
            # 反向传播和参数更新
            if num_updates > 0 and isinstance(total_loss, torch.Tensor):
                avg_loss = total_loss / num_updates
                avg_loss.backward()
                self.optimizer.step()
            
            # 记录训练数据
            if trajectories:
                avg_reward = sum(traj['total_reward'] for traj in trajectories) / len(trajectories)
                avg_steps = sum(traj['steps'] for traj in trajectories) / len(trajectories)
                self.episode_rewards.append(avg_reward)
                self.episode_lengths.append(avg_steps)
                
                # 每训练一次迭代后，输出训练进度和性能指标
                if iteration % 1 == 0:
                    # 获取最近10次episode的奖励，如果不足10次则使用所有可用的奖励
                    recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
                    # 计算最近10次episode的平均奖励，用于监控训练效果
                    avg_reward_10 = sum(recent_rewards) / len(recent_rewards)
                    # 打印训练进度信息，包括当前迭代次数、平均奖励和平均步数
                    print(f"Iteration {iteration}, Avg Reward: {avg_reward_10:.4f}, Avg Steps: {avg_steps:.1f}")

    def optimize_topology(self, initial_graph: nx.Graph = None) -> nx.Graph:
        """优化给定拓扑
        Args:
            initial_graph: 初始网络拓扑图，如果为None则使用环境默认的图
        Returns:
            优化后的网络拓扑图
        """
        # 如果提供了初始图，则设置环境的当前图为初始图
        if initial_graph is not None:
            self.env.current_graph = initial_graph
            # 为图中的每个节点随机分配坐标，用于可视化
            self.env.node_coords = {i: (random.uniform(0, 500), random.uniform(0, 500)) 
                                for i in initial_graph.nodes()}
            # 初始化动作比较器，用于评估不同动作的效果
            self.action_comparator = ActionComparator(self.env)
            # 同步更新 envs[0] 以避免 select_action 中的索引问题
            self.envs[0].current_graph = initial_graph.copy()
            self.envs[0].node_coords = self.env.node_coords.copy()
            self.action_comparators[0] = ActionComparator(self.envs[0])
        else:
            # 如果没有提供初始图，则重置环境
            self.env.reset()
        
        # 获取当前环境状态
        state = self.env.get_state()
        done = False
        step_count = 0
        
        # 计算初始网络的鲁棒性，用于后续比较优化效果
        initial_robustness = self.env.calculate_robustness(self.env.current_graph)
        
        # 在未完成且未达到最大动作次数的情况下，持续优化
        while not done and step_count < self.env.max_actions:
            # 根据当前状态选择最优动作
            action, _ = self.select_action(state, 0)
            
            # 如果没有可用的动作，则结束优化
            if action is None:
                break
                
            # 执行选择的动作，获取新的状态、奖励和是否完成标志
            next_state, reward, done = self.env.step(action)
            state = next_state
            step_count += 1
        
        # 计算最终网络的鲁棒性，并打印优化结果
        final_robustness = self.env.calculate_robustness(self.env.current_graph)
        print(f"优化完成: 初始鲁棒性 {initial_robustness:.4f} -> 最终鲁棒性 {final_robustness:.4f}, 提升: {(final_robustness-initial_robustness)/initial_robustness*100:.2f}%")
        
        return self.env.current_graph

    def save_model(self, path: str) -> None:
        """保存模型
        Args:
            path: 模型保存路径
        """
        # 将GCN和策略网络的参数保存到指定路径
        torch.save({
            'gcn_state_dict': self.gcn.state_dict(),
            'policy_net_state_dict': self.policy_net.state_dict()
        }, path)

    def load_model(self, path: str) -> None:
        """加载模型
        Args:
            path: 模型加载路径
        """
        # 从指定路径加载模型参数，并分别加载到GCN和策略网络中
        checkpoint = torch.load(path)
        self.gcn.load_state_dict(checkpoint['gcn_state_dict'])
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])


def plot_robustness_comparison(initial_graph, optimized_graph, title="SmartTRO优化前后的鲁棒性比较"):
    """绘制优化前后的鲁棒性比较图"""
    env = SmartTROEnvironment()
    # 计算攻击后的网络连通性
    
    def calculate_connectivity_after_attacks(graph):
        g = graph.copy()
        n = g.vcount()
        connectivity = [1.0]  # 初始完全连通
        
        for _ in range(n):
            if g.vcount() == 0:
                connectivity.extend([0] * (n - len(connectivity) + 1))
                break
                
            # HDA攻击
            degrees = g.degree()
            max_degree_node = max(range(g.vcount()), key=lambda x: degrees[x])
            g.delete_vertices(max_degree_node)
            
            # 计算最大连通分量比例
            if g.vcount() > 0:
                components = g.components()
                largest_cc_size = max(len(comp) for comp in components)
                connectivity.append(largest_cc_size / n)
            else:
                connectivity.append(0)
            return connectivity[:n+1]  # 限制为n+1个点（包括初始状态）

    initial_connectivity = calculate_connectivity_after_attacks(initial_graph)
    optimized_connectivity = calculate_connectivity_after_attacks(optimized_graph)

    # 计算鲁棒性值
    initial_robustness = env.calculate_robustness(initial_graph)
    optimized_robustness = env.calculate_robustness(optimized_graph)

    # 绘图
    plt.figure(figsize=(10, 6))
    x = range(len(initial_connectivity))
    plt.plot(x, initial_connectivity, 'r-', label=f'初始拓扑 (R={initial_robustness:.4f})')
    plt.plot(x, optimized_connectivity, 'b-', label=f'优化拓扑 (R={optimized_robustness:.4f})')
    plt.fill_between(x, initial_connectivity, alpha=0.3, color='r')
    plt.fill_between(x, optimized_connectivity, alpha=0.3, color='b')

    plt.xlabel('攻击节点数')
    plt.ylabel('最大连通分量比例')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def visualize_network(graph, node_coords=None, title="网络拓扑可视化"):
    """可视化网络拓扑"""
    plt.figure(figsize=(10, 8))
    
    # 获取节点度数
    degrees = dict(graph.degree())
    
    if node_coords:
        # 使用固定的节点坐标
        pos = node_coords
    else:
        # 使用Fruchterman-Reingold布局
        pos = nx.spring_layout(graph, seed=42)
    
    # 绘制节点，大小与度数成正比
    node_sizes = [50 + 10 * degrees[node] for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
    
    # 绘制边
    nx.draw_networkx_edges(graph, pos, width=0.5, alpha=0.5)
    
    # 绘制节点标签
    nx.draw_networkx_labels(graph, pos, font_size=8)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 创建并训练SmartTRO模型
    model = SmartTRO(num_nodes=50, edge_density=2)
    model.train(num_iterations=2)  # 减少迭代次数以便测试
    
    # 优化网络拓扑
    num_nodes = 50
    p = 0.1  # 边概率
    initial_graph = nx.erdos_renyi_graph(num_nodes, p)
    
    optimized_graph = model.optimize_topology(initial_graph)
    
    # 可视化比较
    plot_robustness_comparison(initial_graph, optimized_graph)
    
    # 可视化网络拓扑
    visualize_network(initial_graph, model.env.node_coords, "初始网络拓扑")
    visualize_network(optimized_graph, model.env.node_coords, "优化后的网络拓扑")