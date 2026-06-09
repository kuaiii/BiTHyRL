# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import random
import math


class UNITY:
    def __init__(self, area_diameter: float = 500, comm_range: float = 200, epsilon: float = 1e-9):
        """
        初始化UNITY算法
        """
        self.area_diameter = area_diameter
        self.comm_range = comm_range
        self.epsilon = epsilon
        
    def _initialize_network(self, N: int) -> nx.Graph:
        """
        初始化网络拓扑
        """
        G = nx.Graph()
        
        # 在指定区域内随机生成节点位置
        for i in range(N):
            # 在圆形区域内随机生成坐标
            r = self.area_diameter/2 * math.sqrt(random.random())
            theta = random.random() * 2 * math.pi
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            
            # 添加节点及其位置
            G.add_node(i, pos=(x, y))
            
        return G
    
    def _get_potential_neighbors(self, G: nx.Graph, node_idx: int) -> tuple[list[int], list[int]]:
        """
        获取节点的潜在邻居列表及其度数
        """
        neighbor_list = []
        degree_list = []
        
        node_pos = G.nodes[node_idx]['pos']
        
        for j in G.nodes():
            # 跳过自身和已经连接的节点
            if j == node_idx or G.has_edge(node_idx, j):
                continue
                
            j_pos = G.nodes[j]['pos']
            
            # 计算两节点间距离
            distance = math.sqrt((node_pos[0] - j_pos[0])**2 + (node_pos[1] - j_pos[1])**2)
            
            # 如果在通信范围内，添加到潜在邻居列表
            if distance <= self.comm_range:
                neighbor_list.append(j)
                degree_list.append(G.degree(j))
                
        return neighbor_list, degree_list
    
    def _mapping_operator(self, degree_list: list[int]) -> list[float]:
        """
        映射操作符：将节点度数映射为连接概率
        实现度平衡连接机制
        """
        if not degree_list:
            return []
            
        # 为所有度加上epsilon，避免除零错误
        adjusted_degrees = [d + self.epsilon for d in degree_list]
        
        # 计算反比概率（度越小，概率越大）
        inverse_degrees = [1.0 / d for d in adjusted_degrees]
        sum_inverse = sum(inverse_degrees)
        
        # 归一化得到概率分布
        probability_list = [inv_d / sum_inverse for inv_d in inverse_degrees]
        
        return probability_list
    
    def _roulette_selection(self, probability_list: list[float]) -> int:
        """
        轮盘赌选择算法
        """
        if not probability_list:
            return None
            
        # 计算累积概率
        cum_prob = np.cumsum(probability_list)
        
        # 生成随机数
        r = random.random()
        
        # 轮盘赌选择
        for i, prob in enumerate(cum_prob):
            if r <= prob:
                return i
                
        return len(probability_list) - 1
    
    def generate_matched_topology(self, input_graph: nx.Graph) -> nx.Graph:
        """
        生成与输入网络节点数和边数完全一致的鲁棒拓扑
        """
        # 获取输入网络的核心参数
        N = input_graph.number_of_nodes()  # 节点数严格匹配
        target_edges = input_graph.number_of_edges()  # 目标边数严格匹配
        
        # 初始化网络
        G = self._initialize_network(N)
        
        # 复制节点位置 (如果有)
        for node in input_graph.nodes():
             if 'pos' in input_graph.nodes[node]:
                 G.nodes[node]['pos'] = input_graph.nodes[node]['pos']

        # 循环添加边直到达到目标数量
        attempts = 0
        max_attempts = target_edges * 20 # 防止死循环
        
        while G.number_of_edges() < target_edges and attempts < max_attempts:
            attempts += 1
            # 随机选择一个源节点
            i = random.choice(list(G.nodes()))
            
            # 获取潜在邻居
            neighbor_list, degree_list = self._get_potential_neighbors(G, i)
            if not neighbor_list:
                continue
                
            # 计算连接概率
            prob_list = self._mapping_operator(degree_list)
            
            # 轮盘赌选择目标节点
            sel_idx = self._roulette_selection(prob_list)
            
            if sel_idx is not None:
                j = neighbor_list[sel_idx]
                if not G.has_edge(i, j):
                    G.add_edge(i, j)
        
        return G