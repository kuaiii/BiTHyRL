import numpy as np
import random
import math
import networkx as nx
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import matplotlib
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy import stats as scipy_stats
from tqdm import tqdm

# 尝试导入numba，如果失败则定义空装饰器
try:
    from numba import jit, njit
except ImportError:
    def njit(func):
        return func
    def jit(func):
        return func

# 添加中文字体配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.use('TkAgg')

# 使用Numba JIT编译加速数值计算
@njit
def fast_distance_calculation(x1, y1, x2, y2):
    """快速距离计算"""
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

@njit
def fast_normalize_probabilities(alpha_chain, beta_chain):
    """快速概率振幅归一化"""
    for i in range(len(alpha_chain)):
        norm = math.sqrt(alpha_chain[i]**2 + beta_chain[i]**2)
        if norm > 0:
            alpha_chain[i] /= norm
            beta_chain[i] /= norm

class UnionFind:
    """并查集结构，用于快速连通性判断"""
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int32)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        fx, fy = self.find(x), self.find(y)
        if fx != fy:
            if self.rank[fx] < self.rank[fy]:
                self.parent[fx] = fy
            elif self.rank[fx] > self.rank[fy]:
                self.parent[fy] = fx
            else:
                self.parent[fy] = fx
                self.rank[fx] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def count_components(self):
        return len(set(self.find(i) for i in range(len(self.parent))))

@dataclass
class Node:
    """网络节点类"""
    id: int
    x: float
    y: float
    neighbors: List[int]
    degree: int

class QuantumChromosome:
    """量子染色体类（双链结构）- 兼容版本"""
    def __init__(self, length: int, use_numpy: bool = True):
        self.length = length
        self.use_numpy = use_numpy
        
        if use_numpy:
            # 使用NumPy数组提高性能
            self.alpha_chain = np.zeros(length, dtype=np.float64)
            self.beta_chain = np.zeros(length, dtype=np.float64)
            self.measured_binary = np.zeros(length, dtype=np.int32)
        else:
            # 使用Python列表保持兼容性
            self.alpha_chain = []
            self.beta_chain = []
            self.measured_binary = []
            
        self.node_mapping = []
        self.fitness = 0.0
    
    def copy(self):
        """兼容的深拷贝"""
        new_chromosome = QuantumChromosome(self.length, self.use_numpy)
        
        if self.use_numpy:
            new_chromosome.alpha_chain = self.alpha_chain.copy()
            new_chromosome.beta_chain = self.beta_chain.copy()
            new_chromosome.measured_binary = self.measured_binary.copy()
        else:
            new_chromosome.alpha_chain = self.alpha_chain.copy()
            new_chromosome.beta_chain = self.beta_chain.copy()
            new_chromosome.measured_binary = self.measured_binary.copy()
            
        new_chromosome.node_mapping = self.node_mapping.copy()
        new_chromosome.fitness = self.fitness
        return new_chromosome
    
    def normalize(self):
        """兼容的归一化方法"""
        if self.use_numpy:
            # 使用向量化操作
            norms = np.sqrt(self.alpha_chain**2 + self.beta_chain**2)
            valid_mask = norms > 0
            self.alpha_chain[valid_mask] /= norms[valid_mask]
            self.beta_chain[valid_mask] /= norms[valid_mask]
        else:
            # 使用循环操作
            for i in range(self.length):
                norm = math.sqrt(self.alpha_chain[i]**2 + self.beta_chain[i]**2)
                if norm > 0:
                    self.alpha_chain[i] /= norm
                    self.beta_chain[i] /= norm

class OptimizedScaleFreeNetwork:
    """优化的无标度网络类"""
    def __init__(self, N: int, M: int, W: float, L: float, R: float):
        self.N = N
        self.M = M
        self.W = W
        self.L = L
        self.R = R
        self.nodes = []
        self.adjacency_matrix = np.zeros((N, N), dtype=np.int8)
        self.valid_edges = []
        self._distance_matrix = None
        
    def deploy_nodes(self):
        """批量部署节点"""
        positions = np.random.uniform(0, [self.W, self.L], (self.N, 2))
        self.nodes = [Node(id=i, x=positions[i,0], y=positions[i,1], neighbors=[], degree=0) 
                      for i in range(self.N)]
    
    def _compute_distance_matrix(self):
        """预计算距离矩阵"""
        if self._distance_matrix is None:
            positions = np.array([[node.x, node.y] for node in self.nodes])
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            self._distance_matrix = np.sqrt(np.sum(diff**2, axis=2))
        return self._distance_matrix
    
    def find_neighbors(self):
        """使用预计算的距离矩阵找邻居"""
        dist_matrix = self._compute_distance_matrix()
        for i, node in enumerate(self.nodes):
            neighbors_mask = (dist_matrix[i] <= self.R) & (np.arange(self.N) != i)
            node.neighbors = np.where(neighbors_mask)[0].tolist()
    
    def build_scale_free_network(self):
        """构建无标度网络"""
        self.deploy_nodes()
        self.find_neighbors()
        self.adjacency_matrix = np.zeros((self.N, self.N), dtype=np.int8)
        
        # 初始节点连接
        if self.nodes[0].neighbors:
            initial_connections = min(self.M, len(self.nodes[0].neighbors))
            selected_neighbors = np.random.choice(self.nodes[0].neighbors, initial_connections, replace=False)
            for neighbor in selected_neighbors:
                self.adjacency_matrix[0, neighbor] = 1
                self.adjacency_matrix[neighbor, 0] = 1
                self.nodes[0].degree += 1
                self.nodes[neighbor].degree += 1
        
        # 优先连接机制
        for i in range(1, self.N):
            available_neighbors = [n for n in self.nodes[i].neighbors if n < i]
            if not available_neighbors:
                continue
            
            degrees = np.array([np.sum(self.adjacency_matrix[neighbor]) + 1 for neighbor in available_neighbors])
            total_degree = np.sum(degrees)
            probabilities = degrees / total_degree if total_degree > 0 else np.ones(len(available_neighbors)) / len(available_neighbors)
            
            connections_to_make = min(self.M, len(available_neighbors))
            selected_neighbors = np.random.choice(available_neighbors, connections_to_make, replace=False, p=probabilities)
            
            for neighbor in selected_neighbors:
                self.adjacency_matrix[i, neighbor] = 1
                self.adjacency_matrix[neighbor, i] = 1
                self.nodes[i].degree += 1
                self.nodes[neighbor].degree += 1
        
        # 批量更新节点度数
        degrees = np.sum(self.adjacency_matrix, axis=1)
        for i in range(self.N):
            self.nodes[i].degree = int(degrees[i])
        
        # 构建有效边列表
        self.valid_edges = [(i, j) for i in range(self.N) for j in range(i+1, self.N) 
                          if j in self.nodes[i].neighbors]
    
    def get_initial_chromosome_data(self) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """返回NumPy数组格式的染色体数据"""
        connection_states = np.array([int(self.adjacency_matrix[i, j]) for (i, j) in self.valid_edges], dtype=np.int32)
        return connection_states, self.valid_edges

class OptimizedQRobustAlgorithm:
    """优化的Q-Robust算法类"""
    def __init__(self, network: OptimizedScaleFreeNetwork, U: int = 5, I: int = 20, Pf: float = 0.2, max_gen: int = 500, T: int = 5):
        self.network = network
        self.U = U
        self.I = I
        self.Pf = Pf
        self.max_gen = max_gen
        self.T = T
        
        # 量子参数
        self.theta_max = 0.05 * math.pi
        self.theta_min = 0.001 * math.pi
        self.Pbm = 0.06
        self.delta_P = 0.05
        
        # 缓存鲁棒性计算结果
        self._robustness_cache = {}
        
        self.quantum_universes = self.initialize_quantum_universes()
    
    def initialize_quantum_universes(self):
        """使用向量化操作初始化量子宇宙"""
        connection_states, valid_edges = self.network.get_initial_chromosome_data()
        chromosome_length = len(connection_states)
        universes = []
        
        for _ in range(self.U):
            universe = []
            for _ in range(self.I):
                chromosome = QuantumChromosome(chromosome_length, use_numpy=True)
                chromosome.node_mapping = valid_edges.copy()
                chromosome.measured_binary = connection_states.copy()
                
                # 向量化初始化概率振幅
                alpha_vals = np.where(connection_states == 1, 
                                    np.random.uniform(0.1, 0.4, chromosome_length),
                                    np.random.uniform(0.6, 0.9, chromosome_length))
                chromosome.alpha_chain = alpha_vals
                chromosome.beta_chain = np.sqrt(1 - alpha_vals**2)
                
                universe.append(chromosome)
            universes.append(universe)
        return universes

    def fast_robustness_calculation(self, matrix: np.ndarray) -> float:
        """优化的鲁棒性计算"""
        matrix_key = hash(matrix.tobytes())
        if matrix_key in self._robustness_cache:
            return self._robustness_cache[matrix_key]
        
        N = matrix.shape[0]
        total = 0.0
        
        degrees = np.array(matrix.sum(axis=1)).flatten()
        remaining_nodes = list(range(N))
        current_matrix = matrix.copy()
        
        for step in range(N):
            if len(remaining_nodes) == 0:
                break
                
            # 使用scipy计算连通分量
            n_components, labels = connected_components(
                csr_matrix(current_matrix[np.ix_(remaining_nodes, remaining_nodes)]), 
                directed=False, return_labels=True
            )
            
            if len(remaining_nodes) > 0:
                component_sizes = np.bincount(labels)
                largest_cc = np.max(component_sizes) if len(component_sizes) > 0 else 0
                total += largest_cc / N
            
            if len(remaining_nodes) <= 1:
                break
                
            current_degrees = degrees[remaining_nodes]
            if len(current_degrees) == 0:
                break
                
            max_degree_idx = np.argmax(current_degrees)
            target_node = remaining_nodes[max_degree_idx]
            
            remaining_nodes.remove(target_node)
            current_matrix[target_node, :] = 0
            current_matrix[:, target_node] = 0
        
        robustness = total / N
        self._robustness_cache[matrix_key] = robustness
        return robustness

    def quantum_measurement(self, chromosome: QuantumChromosome, debug: bool = False) -> np.ndarray:
        """优化的量子测量"""
        original_matrix = self.network.adjacency_matrix.copy()
        N = self.network.N
        
        connected_edges = []
        unconnected_edges = []
        
        for idx, (i, j) in enumerate(chromosome.node_mapping):
            if original_matrix[i, j] > 0:
                connected_edges.append((i, j, idx))
            else:
                unconnected_edges.append((i, j, idx))
        
        # 向量化概率计算
        if connected_edges:
            connected_indices = np.array([idx for _, _, idx in connected_edges])
            edge_probs = chromosome.alpha_chain[connected_indices]**2
            
            edge_swap_probs = [(connected_edges[i][0], connected_edges[i][1], 
                              connected_edges[i][2], edge_probs[i]) 
                             for i in range(len(connected_edges))]
            edge_swap_probs.sort(key=lambda x: x[3], reverse=True)
        else:
            edge_swap_probs = []
        
        new_matrix = original_matrix.copy()
        swapped_edges = set()
        
        # 执行边交换
        for i, j, idx, prob_disconnect in edge_swap_probs:
            if (i, j) in swapped_edges or (j, i) in swapped_edges:
                continue
            
            best_candidate = None
            best_prob = -1
            
            for m, n, idx_to_connect in unconnected_edges:
                if (m, n) in swapped_edges or (n, m) in swapped_edges:
                    continue
                    
                if len({i, j, m, n}) != 4:
                    continue
                    
                prob_connect = chromosome.beta_chain[idx_to_connect]**2
                
                if prob_connect > best_prob:
                    best_prob = prob_connect
                    best_candidate = (m, n, idx_to_connect)
            
            if best_candidate and random.random() < prob_disconnect:
                m, n, idx_to_connect = best_candidate
                
                new_matrix[i, j] = new_matrix[j, i] = 0
                new_matrix[m, n] = new_matrix[n, m] = 1
                
                swapped_edges.add((i, j))
                swapped_edges.add((m, n))
        
        # 批量更新染色体的二进制测量结果
        chromosome.measured_binary = np.array([int(new_matrix[i, j]) for (i, j) in chromosome.node_mapping], dtype=np.int32)
        
        return new_matrix
    
    def quantum_rotation(self, chromosome: QuantumChromosome, best_chromosome: QuantumChromosome, fitness_values: List[float]) -> QuantumChromosome:
        """向量化的量子旋转"""
        new_chrom = chromosome.copy()
        f_max, f_min = max(fitness_values), min(fitness_values)
        if f_max == f_min:
            return new_chrom
        
        delta_theta = ((f_max - chromosome.fitness) / (f_max - f_min)) * (self.theta_max - self.theta_min) + self.theta_min
        
        # 向量化旋转计算
        x_i = chromosome.measured_binary
        b_i = best_chromosome.measured_binary
        alpha = chromosome.alpha_chain
        beta = chromosome.beta_chain
        
        f_X = chromosome.fitness
        f_B = best_chromosome.fitness
        e = alpha * beta
        
        # 批量计算旋转方向
        s = np.zeros(chromosome.length)
        
        # 根据查询表逻辑批量计算s值
        condition2 = (x_i == 0) & (b_i == 1)
        condition3 = (x_i == 1) & (b_i == 0)
        
        # 简化的旋转方向计算
        if f_X < f_B:
            s[condition2 & (e > 0)] = 1
            s[condition2 & (e <= 0)] = -1
            s[condition3 & (e > 0)] = -1
            s[condition3 & (e <= 0)] = 1
        
        # 执行向量化旋转
        theta = s * delta_theta
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        new_chrom.alpha_chain = alpha * cos_theta - beta * sin_theta
        new_chrom.beta_chain = alpha * sin_theta + beta * cos_theta
        
        new_chrom.normalize()
        return new_chrom
    
    def quantum_crossover(self, population: List[QuantumChromosome]) -> List[QuantumChromosome]:
        """优化的量子交叉"""
        n = len(population)
        if n < 2:
            return population
        
        new_pop = [ind.copy() for ind in population]
        gene_length = len(population[0].alpha_chain)
        
        # 向量化交叉操作
        for i in range(gene_length):
            if i < n:
                shifts = i - 1
                if shifts >= 0:
                    alpha_values = np.array([population[j].alpha_chain[i] for j in range(n)])
                    beta_values = np.array([population[j].beta_chain[i] for j in range(n)])
                    
                    for j in range(n):
                        target_j = (j + shifts + 1) % n
                        new_pop[target_j].alpha_chain[i] = alpha_values[j]
                        new_pop[target_j].beta_chain[i] = beta_values[j]
        
        for ind in new_pop:
            ind.normalize()
            
        return new_pop
    
    def quantum_mutation(self, chromosome: QuantumChromosome, f_max: float) -> QuantumChromosome:
        """向量化的量子变异"""
        new_chrom = chromosome.copy()
        
        mutation_prob = self.Pbm + self.delta_P * (1 - chromosome.fitness / f_max) if f_max > 0 else self.Pbm
        
        # 向量化变异操作
        mutation_mask = np.random.random(len(new_chrom.alpha_chain)) < mutation_prob
        
        # 交换alpha和beta
        temp_alpha = new_chrom.alpha_chain[mutation_mask].copy()
        new_chrom.alpha_chain[mutation_mask] = new_chrom.beta_chain[mutation_mask]
        new_chrom.beta_chain[mutation_mask] = temp_alpha
        
        return new_chrom
    
    def multiverse_communication(self):
        """简化的多宇宙通信"""
        universe_fitness = []
        for universe in self.quantum_universes:
            max_fitness = max(ind.fitness for ind in universe) if universe else 0
            universe_fitness.append(max_fitness)
        
        primary_idx = np.argmax(universe_fitness)
        secondary_indices = [i for i in range(self.U) if i != primary_idx]
        
        if secondary_indices:
            best_secondary_idx = max(secondary_indices, key=lambda x: universe_fitness[x])
            primary_universe = self.quantum_universes[primary_idx]
            best_primary_ind = max(primary_universe, key=lambda x: x.fitness)
            
            num_update = int(self.Pf * self.I)
            
            for u_idx in secondary_indices:
                if u_idx == best_secondary_idx:
                    continue
                    
                secondary_universe = self.quantum_universes[u_idx]
                individuals_to_update = sorted(secondary_universe, key=lambda x: x.fitness)[:num_update]
                
                for ind in individuals_to_update:
                    delta_theta = 0.1 * self.theta_max
                    
                    # 向量化旋转
                    alpha, beta = ind.alpha_chain, ind.beta_chain
                    best_alpha, best_beta = best_primary_ind.alpha_chain, best_primary_ind.beta_chain
                    
                    # 简化旋转方向判断
                    condition = ((alpha**2 > beta**2) & (best_alpha**2 < best_beta**2)) | \
                               ((alpha**2 < beta**2) & (best_alpha**2 > best_beta**2))
                    
                    s = np.where(condition, 
                               np.where(alpha * beta > 0, -1, 1),
                               np.where(alpha * beta > 0, 1, -1))
                    
                    theta = s * delta_theta
                    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
                    
                    ind.alpha_chain = alpha * cos_theta - beta * sin_theta
                    ind.beta_chain = alpha * sin_theta + beta * cos_theta
                    
                    ind.normalize()
    
    def run(self, debug: bool = False) -> Tuple[float, List[float], np.ndarray]:
        """优化的主运行循环"""
        best_history = []
        best_overall_matrix = None
        best_overall_fitness = 0.0

        if debug:
            print("[初始化] 开始第一轮测量与适应度计算")
        
        for universe in self.quantum_universes:
            for ind in universe:
                matrix = self.quantum_measurement(ind, debug=debug)
                ind.fitness = self.fast_robustness_calculation(matrix)

        for gen in tqdm(range(self.max_gen), desc="优化迭代中"):
            for universe_idx, universe in enumerate(self.quantum_universes):
                fitness_values = [ind.fitness for ind in universe]
                f_max = max(fitness_values)
                best_ind = max(universe, key=lambda x: x.fitness)

                new_pop = [self.quantum_rotation(ind, best_ind, fitness_values) for ind in universe]
                new_pop = [self.quantum_mutation(ind, f_max) for ind in new_pop]
                new_pop = self.quantum_crossover(new_pop)

                for ind in new_pop:
                    matrix = self.quantum_measurement(ind, debug=False)
                    ind.fitness = self.fast_robustness_calculation(matrix)
                    
                    if ind.fitness > best_overall_fitness:
                        best_overall_fitness = ind.fitness
                        best_overall_matrix = matrix.copy()

                combined = universe + new_pop
                self.quantum_universes[universe_idx] = sorted(combined, key=lambda x: x.fitness, reverse=True)[:self.I]

            if (gen + 1) % self.T == 0:
                if debug:
                    print(f"[通信] 第 {gen+1} 代触发多宇宙通信")
                self.multiverse_communication()

            best_fitness = max(ind.fitness for u in self.quantum_universes for ind in u)
            best_history.append(best_fitness)

            if debug and (gen + 1) % 10 == 0:
                print(f"[第 {gen+1} 代] 当前最优适应度: {best_fitness:.4f}")

        return best_overall_fitness, best_history, best_overall_matrix

# 原始算法类保持不变
class ScaleFreeNetwork:
    """无标度网络类"""
    def __init__(self, N: int, M: int, W: float, L: float, R: float):
        self.N = N
        self.M = M
        self.W = W
        self.L = L
        self.R = R
        self.nodes = []
        self.adjacency_matrix = np.zeros((N, N))
        self.valid_edges = []
        
    def deploy_nodes(self):
        """部署节点"""
        self.nodes = [Node(id=i, x=random.uniform(0, self.W), y=random.uniform(0, self.L), neighbors=[], degree=0) 
                      for i in range(self.N)]
    
    def calculate_distance(self, node1: Node, node2: Node) -> float:
        """计算距离"""
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def find_neighbors(self):
        """找到通信范围内的邻居"""
        for i, node1 in enumerate(self.nodes):
            node1.neighbors = [j for j, node2 in enumerate(self.nodes) 
                              if i != j and self.calculate_distance(node1, node2) <= self.R]
    
    def build_scale_free_network(self):
        """构建无标度网络（保持度分布幂律特性）"""
        self.deploy_nodes()
        self.find_neighbors()
        self.adjacency_matrix = np.zeros((self.N, self.N))
        
        # 初始节点连接
        if self.nodes[0].neighbors:
            initial_connections = min(self.M, len(self.nodes[0].neighbors))
            for neighbor in random.sample(self.nodes[0].neighbors, initial_connections):
                self.adjacency_matrix[0][neighbor] = 1
                self.adjacency_matrix[neighbor][0] = 1
                self.nodes[0].degree += 1
                self.nodes[neighbor].degree += 1
        
        # 优先连接机制
        for i in range(1, self.N):
            available_neighbors = [n for n in self.nodes[i].neighbors if n < i]
            if not available_neighbors:
                continue
            
            degrees = [sum(self.adjacency_matrix[neighbor]) + 1 for neighbor in available_neighbors]
            total_degree = sum(degrees)
            probabilities = [d / total_degree for d in degrees] if total_degree > 0 else [1/len(available_neighbors)]*len(available_neighbors)
            
            connections_to_make = min(self.M, len(available_neighbors))
            selected_neighbors = np.random.choice(available_neighbors, connections_to_make, replace=False, p=probabilities)
            for neighbor in selected_neighbors:
                self.adjacency_matrix[i][neighbor] = 1
                self.adjacency_matrix[neighbor][i] = 1
                self.nodes[i].degree += 1
                self.nodes[neighbor].degree += 1
        
        # 更新节点度数
        for i in range(self.N):
            self.nodes[i].degree = int(sum(self.adjacency_matrix[i]))
        
        # 构建有效边列表
        self.valid_edges = [(i, j) for i in range(self.N) for j in range(i+1, self.N) 
                          if j in self.nodes[i].neighbors]
    
    def get_initial_chromosome_data(self) -> Tuple[List[int], List[Tuple[int, int]]]:
        """生成染色体数据"""
        connection_states = [int(self.adjacency_matrix[i][j]) for (i, j) in self.valid_edges]
        return connection_states, self.valid_edges

class QRobustAlgorithm:
    """Q-Robust算法类（原始版本）"""
    def __init__(self, network: ScaleFreeNetwork, U: int = 5, I: int = 20, Pf: float = 0.2, max_gen: int = 500, T: int = 5):
        self.network = network
        self.U = U
        self.I = I
        self.Pf = Pf
        self.max_gen = max_gen
        self.T = T
        
        # 量子参数
        self.theta_max = 0.05 * math.pi
        self.theta_min = 0.001 * math.pi
        self.Pbm = 0.06
        self.delta_P = 0.05
        
        self.quantum_universes = self.initialize_quantum_universes()
    
    def initialize_quantum_universes(self):
        """初始化量子宇宙"""
        connection_states, valid_edges = self.network.get_initial_chromosome_data()
        chromosome_length = len(connection_states)
        universes = []
        
        for _ in range(self.U):
            universe = []
            for _ in range(self.I):
                chromosome = QuantumChromosome(chromosome_length, use_numpy=False)
                chromosome.node_mapping = valid_edges.copy()
                chromosome.measured_binary = connection_states.copy()
                
                # 初始化概率振幅
                for state in connection_states:
                    if state == 1:
                        alpha = random.uniform(0.1, 0.4)
                    else:
                        alpha = random.uniform(0.6, 0.9)
                    beta = math.sqrt(1 - alpha**2)
                    chromosome.alpha_chain.append(alpha)
                    chromosome.beta_chain.append(beta)
                
                universe.append(chromosome)
            universes.append(universe)
        return universes

    def quantum_measurement(self, chromosome: QuantumChromosome, debug: bool = False) -> np.ndarray:
        """量子测量方法"""
        original_matrix = self.network.adjacency_matrix.copy()
        N = self.network.N
        node_degrees = [node.degree for node in self.network.nodes]
        
        connected_edges = []
        unconnected_edges = []
        
        for idx, (i, j) in enumerate(chromosome.node_mapping):
            if original_matrix[i][j] > 0:
                connected_edges.append((i, j, idx))
            else:
                unconnected_edges.append((i, j, idx))
        
        edge_swap_probs = []
        for i, j, idx in connected_edges:
            prob_disconnect = chromosome.alpha_chain[idx]**2
            edge_swap_probs.append((i, j, idx, prob_disconnect))
        
        edge_swap_probs.sort(key=lambda x: x[3], reverse=True)
        
        new_matrix = original_matrix.copy()
        swapped_edges = set()
        
        for i, j, idx, prob_disconnect in edge_swap_probs:
            if (i, j) in swapped_edges or (j, i) in swapped_edges:
                continue
            
            best_candidate = None
            best_prob = -1
            
            for m, n, idx_to_connect in unconnected_edges:
                if (m, n) in swapped_edges or (n, m) in swapped_edges:
                    continue
                    
                if len({i, j, m, n}) != 4:
                    continue
                    
                prob_connect = chromosome.beta_chain[idx_to_connect]**2
                
                if prob_connect > best_prob:
                    best_prob = prob_connect
                    best_candidate = (m, n, idx_to_connect)
            
            if best_candidate and random.random() < prob_disconnect:
                m, n, idx_to_connect = best_candidate
                
                new_matrix[i][j] = new_matrix[j][i] = 0
                new_matrix[m][n] = new_matrix[n][m] = 1
                
                swapped_edges.add((i, j))
                swapped_edges.add((m, n))
                
                if debug:
                    print(f"交换边: ({i},{j}) 断开, ({m},{n}) 连接")
        
        new_degrees = [sum(new_matrix[i]) for i in range(N)]
        if debug:
            for i in range(N):
                if new_degrees[i] != node_degrees[i]:
                    print(f"警告: 节点 {i} 度数变化: {node_degrees[i]} -> {new_degrees[i]}")
        
        chromosome.measured_binary = [int(new_matrix[i][j]) for (i, j) in chromosome.node_mapping]
        
        return new_matrix
    
    def calculate_robustness(self, matrix: np.ndarray) -> float:
        """计算鲁棒性R值"""
        G = nx.from_numpy_array(matrix)
        N = G.number_of_nodes()
        total = 0.0
        
        for _ in range(N):
            if G.number_of_nodes() == 0:
                break
            largest_cc = max((len(c) for c in nx.connected_components(G)), default=0)
            total += largest_cc / N
            
            if G.number_of_nodes() > 0:
                degrees = dict(G.degree())
                target = max(degrees, key=degrees.get)
                G.remove_node(target)
        
        return total / N
    
    def quantum_rotation(self, chromosome: QuantumChromosome, best_chromosome: QuantumChromosome, fitness_values: List[float]) -> QuantumChromosome:
        """量子旋转"""
        new_chrom = chromosome.copy()
        f_max, f_min = max(fitness_values), min(fitness_values)
        if f_max == f_min:
            return new_chrom
        
        delta_theta = ((f_max - chromosome.fitness) / (f_max - f_min)) * (self.theta_max - self.theta_min) + self.theta_min
        
        for i in range(chromosome.length):
            alpha, beta = chromosome.alpha_chain[i], chromosome.beta_chain[i]
            
            x_i = chromosome.measured_binary[i]
            b_i = best_chromosome.measured_binary[i]
            
            f_X = chromosome.fitness
            f_B = best_chromosome.fitness
            e = alpha * beta
            
            s = 0
            
            if x_i == 0 and b_i == 0:
                if f_X >= f_B:
                    if alpha == 0:
                        s = 0
                    elif beta == 0:
                        s = 0
                    elif e > 0:
                        s = -1
                    else:
                        s = 1
                else:
                    if alpha == 0:
                        s = 0
                    elif beta == 0:
                        s = 0
                    elif e > 0:
                        s = -1
                    else:
                        s = 1
            
            elif x_i == 0 and b_i == 1:
                if f_X >= f_B:
                    if alpha == 0:
                        s = 0
                    elif beta == 0:
                        s = 0
                    elif e > 0:
                        s = -1
                    else:
                        s = 1
                else:
                    if alpha == 0:
                        s = 1
                    elif beta == 0:
                        s = 0
                    elif e > 0:
                        s = 1
                    else:
                        s = -1
            
            elif x_i == 1 and b_i == 0:
                if f_X >= f_B:
                    if alpha == 0:
                        s = 0
                    elif beta == 0:
                        s = 0
                    elif e > 0:
                        s = 1
                    else:
                        s = -1
                else:
                    if alpha == 0:
                        s = 0
                    elif beta == 0:
                        s = -1
                    elif e > 0:
                        s = -1
                    else:
                        s = 1
            
            elif x_i == 1 and b_i == 1:
                if f_X >= f_B:
                    if alpha == 0:
                        s = 0
                    elif beta == 0:
                        s = 0
                    elif e > 0:
                        s = 1
                    else:
                        s = -1
                else:
                    if alpha == 0:
                        s = 0
                    elif beta == 0:
                        s = 0
                    elif e > 0:
                        s = 1
                    else:
                        s = -1
            
            theta = s * delta_theta
            new_alpha = alpha * math.cos(theta) - beta * math.sin(theta)
            new_beta = alpha * math.sin(theta) + beta * math.cos(theta)
            new_chrom.alpha_chain[i], new_chrom.beta_chain[i] = new_alpha, new_beta
        
        new_chrom.normalize()
        return new_chrom
    
    def quantum_crossover(self, population: List[QuantumChromosome]) -> List[QuantumChromosome]:
        """量子交叉"""
        n = len(population)
        if n < 2:
            return population
        
        new_pop = [ind.copy() for ind in population]
        gene_length = len(population[0].alpha_chain)
        
        for i in range(gene_length):
            if i < n:
                shifts = i - 1
                if shifts >= 0:
                    for j in range(n):
                        target_j = (j + shifts + 1) % n
                        new_pop[target_j].alpha_chain[i] = population[j].alpha_chain[i]
                        new_pop[target_j].beta_chain[i] = population[j].beta_chain[i]
            else:
                k = i % n
                if k == 0:
                    shifts = n - 2
                else:
                    shifts = k - 1
                
                if shifts >= 0:
                    for j in range(n):
                        target_j = (j + shifts + 1) % n
                        new_pop[target_j].alpha_chain[i] = population[j].alpha_chain[i]
                        new_pop[target_j].beta_chain[i] = population[j].beta_chain[i]
        
        for ind in new_pop:
            ind.normalize()
            
        return new_pop
    
    def quantum_mutation(self, chromosome: QuantumChromosome, f_max: float) -> QuantumChromosome:
        """量子变异"""
        new_chrom = chromosome.copy()
        
        mutation_prob = self.Pbm + self.delta_P * (1 - chromosome.fitness / f_max) if f_max > 0 else self.Pbm
        
        for i in range(len(new_chrom.alpha_chain)):
            if random.random() < mutation_prob:
                new_chrom.alpha_chain[i], new_chrom.beta_chain[i] = chromosome.beta_chain[i], chromosome.alpha_chain[i]
        
        return new_chrom
    
    def multiverse_communication(self):
        """多宇宙通信"""
        universe_fitness = []
        for universe in self.quantum_universes:
            max_fitness = max(ind.fitness for ind in universe) if universe else 0
            universe_fitness.append(max_fitness)
        
        primary_idx = np.argmax(universe_fitness)
        secondary_indices = [i for i in range(self.U) if i != primary_idx]
        
        if secondary_indices:
            best_secondary_idx = max(secondary_indices, key=lambda x: universe_fitness[x])
            
            primary_universe = self.quantum_universes[primary_idx]
            best_primary_ind = max(primary_universe, key=lambda x: x.fitness)
            
            num_update = int(self.Pf * self.I)
            
            for u_idx in secondary_indices:
                if u_idx == best_secondary_idx:
                    continue
                    
                secondary_universe = self.quantum_universes[u_idx]
                individuals_to_update = sorted(secondary_universe, key=lambda x: x.fitness)[:num_update]
                
                for ind in individuals_to_update:
                    delta_theta = 0.1 * self.theta_max
                    for i in range(ind.length):
                        alpha, beta = ind.alpha_chain[i], ind.beta_chain[i]
                        best_alpha, best_beta = best_primary_ind.alpha_chain[i], best_primary_ind.beta_chain[i]
                        
                        if (alpha**2 > beta**2 and best_alpha**2 < best_beta**2) or \
                           (alpha**2 < beta**2 and best_alpha**2 > best_beta**2):
                            s = -1 if alpha * beta > 0 else 1
                        else:
                            s = 1 if alpha * beta > 0 else -1
                            
                        theta = s * delta_theta
                        new_alpha = alpha * math.cos(theta) - beta * math.sin(theta)
                        new_beta = alpha * math.sin(theta) + beta * math.cos(theta)
                        ind.alpha_chain[i], ind.beta_chain[i] = new_alpha, new_beta
                    
                    ind.normalize()
            
            best_secondary_universe = self.quantum_universes[best_secondary_idx]
            best_secondary_ind = max(best_secondary_universe, key=lambda x: x.fitness)
            
            primary_individuals_to_update = sorted(primary_universe, key=lambda x: x.fitness)[:num_update]
            
            for ind in primary_individuals_to_update:
                delta_theta = 0.1 * self.theta_max
                for i in range(ind.length):
                    alpha, beta = ind.alpha_chain[i], ind.beta_chain[i]
                    best_alpha, best_beta = best_secondary_ind.alpha_chain[i], best_secondary_ind.beta_chain[i]
                    
                    if (alpha**2 > beta**2 and best_alpha**2 < best_beta**2) or \
                       (alpha**2 < beta**2 and best_alpha**2 > best_beta**2):
                        s = -1 if alpha * beta > 0 else 1
                    else:
                        s = 1 if alpha * beta > 0 else -1
                        
                    theta = s * delta_theta
                    new_alpha = alpha * math.cos(theta) - beta * math.sin(theta)
                    new_beta = alpha * math.sin(theta) + beta * math.cos(theta)
                    ind.alpha_chain[i], ind.beta_chain[i] = new_alpha, new_beta
                
                ind.normalize()

    def run(self, debug: bool = False) -> Tuple[float, List[float], np.ndarray]:
        """运行算法"""
        best_history = []
        best_overall_matrix = None
        best_overall_fitness = 0.0

        if debug:
            print("[初始化] 开始第一轮测量与适应度计算")
        for universe in self.quantum_universes:
            for ind in universe:
                matrix = self.quantum_measurement(ind, debug=debug)
                ind.fitness = self.calculate_robustness(matrix)

        for gen in tqdm(range(self.max_gen), desc="优化迭代中"):
            for universe_idx, universe in enumerate(self.quantum_universes):
                fitness_values = [ind.fitness for ind in universe]
                f_max = max(fitness_values)
                best_ind = max(universe, key=lambda x: x.fitness)

                new_pop = [self.quantum_rotation(ind, best_ind, fitness_values) for ind in universe]
                new_pop = [self.quantum_mutation(ind, f_max) for ind in new_pop]
                new_pop = self.quantum_crossover(new_pop)

                for ind in new_pop:
                    matrix = self.quantum_measurement(ind, debug=False)
                    ind.fitness = self.calculate_robustness(matrix)
                    
                    if ind.fitness > best_overall_fitness:
                        best_overall_fitness = ind.fitness
                        best_overall_matrix = matrix.copy()

                combined = universe + new_pop
                self.quantum_universes[universe_idx] = sorted(combined, key=lambda x: x.fitness, reverse=True)[:self.I]

            if (gen + 1) % self.T == 0:
                if debug:
                    print(f"[通信] 第 {gen+1} 代触发多宇宙通信")
                self.multiverse_communication()

            best_fitness = max(ind.fitness for u in self.quantum_universes for ind in u)
            best_history.append(best_fitness)

            if debug and (gen + 1) % 10 == 0:
                print(f"[第 {gen+1} 代] 当前最优适应度: {best_fitness:.4f}")

        return best_overall_fitness, best_history, best_overall_matrix

# 更新函数接口
def Q_Robust_method(G, max_gen=10, use_optimized=True):
    """优化网络拓扑的鲁棒性算法接口"""
    random.seed(42)
    np.random.seed(42)
    
    if not isinstance(G, nx.Graph):
        raise ValueError("输入必须是NetworkX图对象")
    if G.number_of_nodes() == 0:
        raise ValueError("输入图不能是空图")
    
    node_positions = None
    if all('x' in G.nodes[n] and 'y' in G.nodes[n] for n in G.nodes):
        node_positions = {n: (G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes}
    
    N = G.number_of_nodes()
    adj_matrix = nx.to_numpy_array(G)
    
    if use_optimized:
        # 使用优化版本
        adj_matrix = adj_matrix.astype(np.int8)
        network = OptimizedScaleFreeNetwork(N=N, M=2, W=300, L=300, R=150)
        
        network.nodes = [
            Node(
                id=i, 
                x=node_positions[i][0] if node_positions else random.uniform(0, 300), 
                y=node_positions[i][1] if node_positions else random.uniform(0, 300),
                neighbors=[j for j in range(N) if adj_matrix[i][j] > 0],
                degree=int(np.sum(adj_matrix[i]))
            ) 
            for i in range(N)
        ]
        
        # 预计算距离矩阵
        positions = np.array([[node.x, node.y] for node in network.nodes])
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff**2, axis=2))
        network._distance_matrix = distance_matrix
        
        network.valid_edges = [
            (i, j) for i in range(N) for j in range(i+1, N) 
            if distance_matrix[i, j] <= network.R
        ]
        network.adjacency_matrix = adj_matrix
        
        q_robust = OptimizedQRobustAlgorithm(
            network, 
            U=3,
            I=30,
            max_gen=max_gen,
            T=5,
            Pf=0.2
        )
    else:
        # 使用原始版本
        network = ScaleFreeNetwork(N=N, M=2, W=300, L=300, R=150)
        
        network.nodes = [
            Node(
                id=i, 
                x=node_positions[i][0] if node_positions else random.uniform(0, 300), 
                y=node_positions[i][1] if node_positions else random.uniform(0, 300),
                neighbors=[j for j in range(N) if adj_matrix[i][j] > 0],
                degree=int(sum(adj_matrix[i]))
            ) 
            for i in range(N)
        ]
        
        network.valid_edges = [
            (i, j) for i in range(N) for j in range(i+1, N) 
            if network.calculate_distance(network.nodes[i], network.nodes[j]) <= network.R
        ]
        network.adjacency_matrix = adj_matrix
        
        q_robust = QRobustAlgorithm(
            network, 
            U=3,
            I=30,
            max_gen=max_gen,
            T=5,
            Pf=0.2
        )
    
    final_fitness, history, optimized_matrix = q_robust.run(debug=False)
    optimized_G = nx.from_numpy_array(optimized_matrix)
    
    if node_positions:
        for n in optimized_G.nodes:
            optimized_G.nodes[n]['x'] = node_positions[n][0]
            optimized_G.nodes[n]['y'] = node_positions[n][1]
    
    return optimized_G

# 保留所有可视化和分析函数
def evaluate_attack_comparison(matrices: List[Tuple[np.ndarray, str]]):
    """比较多个网络在恶意攻击下的鲁棒性"""
    plt.figure(figsize=(10, 6))

    for matrix, label in matrices:
        G = nx.from_numpy_array(matrix)
        N = G.number_of_nodes()
        sizes = []

        for _ in range(N):
            if G.number_of_nodes() == 0:
                break
            largest_cc = max((len(c) for c in nx.connected_components(G)), default=0)
            sizes.append(largest_cc / N)
            degrees = dict(G.degree())
            if degrees:
                target = max(degrees, key=degrees.get)
                G.remove_node(target)

        x = np.linspace(0, 1, len(sizes))
        plt.plot(x, sizes, marker='o', markersize=4, label=label)

    plt.xlabel("移除节点比例")
    plt.ylabel("最大连通子图比例")
    plt.title("网络抗恶意攻击能力对比")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_network(matrix: np.ndarray, title: str = "网络拓扑可视化"):
    """可视化网络拓扑"""
    G = nx.from_numpy_array(matrix)
    plt.figure(figsize=(10, 8))
    
    degrees = dict(G.degree())
    node_sizes = [v * 30 + 50 for v in degrees.values()]
    
    pos = nx.spring_layout(G, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.7, 
                          node_color='skyblue', edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_robustness_comparison(initial_graph, optimized_graph, title="Unity"):
    """绘制优化前后的鲁棒性比较图"""
    def calculate_connectivity_after_attacks(graph):
        g = graph.copy()
        n = g.number_of_nodes()
        connectivity = [1.0]
        
        for _ in range(n):
            if g.number_of_nodes() == 0:
                connectivity.extend([0] * (n - len(connectivity) + 1))
                break
                
            degrees = dict(g.degree())
            if not degrees:
                break
                
            max_degree_node = max(degrees, key=degrees.get)
            g.remove_node(max_degree_node)
            
            if g.number_of_nodes() > 0:
                components = list(nx.connected_components(g))
                largest_cc_size = max(len(comp) for comp in components) if components else 0
                connectivity.append(largest_cc_size / n)
            else:
                connectivity.append(0)
        
        return connectivity[:n+1]
    
    def calculate_robustness(graph):
        g = graph.copy()
        n = g.number_of_nodes()
        robustness_sum = 0.0

        degrees = dict(g.degree())
        node_order = sorted(degrees.keys(), key=lambda node: degrees[node], reverse=True)

        for step in range(n + 1):
            if g.number_of_nodes() > 0:
                components = list(nx.connected_components(g))
                largest_cc_size = max(len(comp) for comp in components) if components else 0
                robustness_sum += largest_cc_size / n
            else:
                robustness_sum += 0.0

            if step < n and node_order:
                node_to_remove = node_order[step]
                if g.has_node(node_to_remove):
                    g.remove_node(node_to_remove)

        return robustness_sum / (n + 1)
    
    initial_connectivity = calculate_connectivity_after_attacks(initial_graph)
    optimized_connectivity = calculate_connectivity_after_attacks(optimized_graph)
    
    initial_robustness = calculate_robustness(initial_graph)
    optimized_robustness = calculate_robustness(optimized_graph)
    
    plt.figure(figsize=(10, 6))
    x = range(len(initial_connectivity))
    plt.plot(x, initial_connectivity, 'r-', label=f'Initial Topology (R={initial_robustness:.4f})')
    plt.plot(x, optimized_connectivity, 'b-', label=f'Optimized Topology (R={optimized_robustness:.4f})')
    plt.fill_between(x, initial_connectivity, alpha=0.3, color='r')
    plt.fill_between(x, optimized_connectivity, alpha=0.3, color='b')
    
    plt.xlabel('Number of Removed Nodes')
    plt.ylabel('Largest Connected Component Ratio')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    improvement = (optimized_robustness - initial_robustness) / initial_robustness * 100
    print(f"Robustness improvement: {improvement:.2f}%")
    
    return initial_robustness, optimized_robustness, improvement

def plot_random_robustness_comparison(initial_graph, optimized_graph, title="Unity", attack_type="random", num_simulations=10):
    """绘制随机攻击下的鲁棒性比较图"""
    def calculate_connectivity_random_attacks(graph):
        n = graph.number_of_nodes()
        all_results = []
        
        for _ in range(num_simulations):
            g = graph.copy()
            connectivity = [1.0]
            nodes = list(g.nodes())
            random.shuffle(nodes)
            
            for node in nodes:
                g.remove_node(node)
                
                if g.number_of_nodes() > 0:
                    components = list(nx.connected_components(g))
                    largest_cc_size = max(len(comp) for comp in components) if components else 0
                    connectivity.append(largest_cc_size / n)
                else:
                    connectivity.append(0)
            
            if len(connectivity) < n + 1:
                connectivity.extend([0] * (n + 1 - len(connectivity)))
            
            all_results.append(connectivity[:n+1])
        
        avg_connectivity = [sum(col)/num_simulations for col in zip(*all_results)]
        return avg_connectivity
    
    def calculate_connectivity_hda(graph):
        g = graph.copy()
        n = g.number_of_nodes()
        connectivity = [1.0]
        
        for _ in range(n):
            if g.number_of_nodes() == 0:
                connectivity.extend([0] * (n - len(connectivity) + 1))
                break
                
            degrees = dict(g.degree())
            if not degrees:
                break
                
            max_degree_node = max(degrees, key=degrees.get)
            g.remove_node(max_degree_node)
            
            if g.number_of_nodes() > 0:
                components = list(nx.connected_components(g))
                largest_cc_size = max(len(comp) for comp in components) if components else 0
                connectivity.append(largest_cc_size / n)
            else:
                connectivity.append(0)
        
        return connectivity[:n+1]
    
    def calculate_robustness(connectivity):
        n = len(connectivity) - 1
        return sum(connectivity) / (n + 1)
    
    if attack_type.lower() == "random":
        initial_connectivity = calculate_connectivity_random_attacks(initial_graph)
        optimized_connectivity = calculate_connectivity_random_attacks(optimized_graph)
        attack_label = "Random Attack"
    else:
        initial_connectivity = calculate_connectivity_hda(initial_graph)
        optimized_connectivity = calculate_connectivity_hda(optimized_graph)
        attack_label = "High Degree Attack"
    
    initial_robustness = calculate_robustness(initial_connectivity)
    optimized_robustness = calculate_robustness(optimized_connectivity)
    
    plt.figure(figsize=(10, 6))
    x = range(len(initial_connectivity))
    plt.plot(x, initial_connectivity, 'r-', label=f'Initial Topology (R={initial_robustness:.4f})')
    plt.plot(x, optimized_connectivity, 'b-', label=f'Optimized Topology (R={optimized_robustness:.4f})')
    plt.fill_between(x, initial_connectivity, alpha=0.3, color='r')
    plt.fill_between(x, optimized_connectivity, alpha=0.3, color='b')
    
    plt.xlabel('Number of Removed Nodes')
    plt.ylabel('Largest Connected Component Ratio')
    plt.title(f"{title} - {attack_label}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    improvement = (optimized_robustness - initial_robustness) / initial_robustness * 100
    print(f"Robustness improvement ({attack_label}): {improvement:.2f}%")
    
    return initial_robustness, optimized_robustness, improvement

def compare_degree_distributions(initial_graph, optimized_graph, 
                                 initial_name="Initial Network", 
                                 optimized_name="Optimized Network",
                                 show_plot=True):
    """比较两个图的度分布"""
    initial_degrees = [d for _, d in initial_graph.degree()]
    optimized_degrees = [d for _, d in optimized_graph.degree()]
    
    stats = {
        "initial": {
            "nodes": initial_graph.number_of_nodes(),
            "edges": initial_graph.number_of_edges(),
            "mean_degree": sum(initial_degrees) / len(initial_degrees),
            "max_degree": max(initial_degrees)
        },
        "optimized": {
            "nodes": optimized_graph.number_of_nodes(),
            "edges": optimized_graph.number_of_edges(),
            "mean_degree": sum(optimized_degrees) / len(optimized_degrees),
            "max_degree": max(optimized_degrees)
        }
    }
    
    node_change = ((stats["optimized"]["nodes"] - stats["initial"]["nodes"]) / 
                   stats["initial"]["nodes"]) * 100
    edge_change = ((stats["optimized"]["edges"] - stats["initial"]["edges"]) / 
                   stats["initial"]["edges"]) * 100
    mean_degree_change = ((stats["optimized"]["mean_degree"] - stats["initial"]["mean_degree"]) / 
                          stats["initial"]["mean_degree"]) * 100
    max_degree_change = ((stats["optimized"]["max_degree"] - stats["initial"]["max_degree"]) / 
                         stats["initial"]["max_degree"]) * 100
    
    ks_stat, ks_pvalue = scipy_stats.ks_2samp(initial_degrees, optimized_degrees)
    
    print("Degree Distribution Analysis:")
    print(f"1. Nodes: {stats['initial']['nodes']} -> {stats['optimized']['nodes']} (Change: {node_change:.2f}%)")
    print(f"2. Edges: {stats['initial']['edges']} -> {stats['optimized']['edges']} (Change: {edge_change:.2f}%)")
    print(f"3. Mean degree: {stats['initial']['mean_degree']:.2f} -> {stats['optimized']['mean_degree']:.2f} (Change: {mean_degree_change:.2f}%)")
    print(f"4. Max degree: {stats['initial']['max_degree']} -> {stats['optimized']['max_degree']} (Change: {max_degree_change:.2f}%)")
    print(f"5. KS test: p={ks_pvalue:.4f} {'(Similar distributions)' if ks_pvalue > 0.05 else '(Different distributions)'}")
    
    if show_plot:
        plt.figure(figsize=(10, 6))
        
        max_degree = max(stats["initial"]["max_degree"], stats["optimized"]["max_degree"])
        bins = range(0, max_degree + 2)
        
        plt.hist(initial_degrees, bins=bins, alpha=0.7, density=True, 
                 label=f"{initial_name} (Mean={stats['initial']['mean_degree']:.2f})")
        plt.hist(optimized_degrees, bins=bins, alpha=0.7, density=True, 
                 label=f"{optimized_name} (Mean={stats['optimized']['mean_degree']:.2f})")
        
        plt.title("Degree Distribution Comparison")
        plt.xlabel("Degree")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    
    return {
        "initial": stats["initial"],
        "optimized": stats["optimized"],
        "change": {
            "nodes": node_change,
            "edges": edge_change,
            "mean_degree": mean_degree_change,
            "max_degree": max_degree_change
        },
        "ks_test": {
            "statistic": ks_stat,
            "pvalue": ks_pvalue
        }
    }

def adj_matrix_to_nx(adj_matrix):
    """将邻接矩阵转换为NetworkX图"""
    if not isinstance(adj_matrix, np.ndarray):
        raise ValueError("输入必须是NumPy数组")
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("输入必须是正方形矩阵")
    if not np.allclose(adj_matrix, adj_matrix.T):
        raise ValueError("邻接矩阵必须是对称的")
    if not np.all(np.diag(adj_matrix) == 0):
        raise ValueError("对角线元素必须全为0")
    
    G = nx.Graph()
    n = adj_matrix.shape[0]
    
    G.add_nodes_from(range(n))
    
    rows, cols = np.triu_indices(n, k=1)
    for i, j in zip(rows, cols):
        weight = adj_matrix[i, j]
        if weight != 0:
            G.add_edge(i, j, weight=weight)
    
    return G

def main():
    """主函数"""
    random.seed(42)
    np.random.seed(42)
    
    network = ScaleFreeNetwork(N=100, M=2, W=300, L=300, R=150)
    network.build_scale_free_network()
    
    visualize_network(network.adjacency_matrix, "初始无标度网络拓扑")
    
    q_robust = QRobustAlgorithm(
        network, 
        U=3,
        I=3,
        max_gen=10,
        T=5,
        Pf=0.2
    )
    print("1 开始运行优化网络算法代码")
    final_fitness, history, optimized_matrix = q_robust.run(debug=True)
    
    initial_matrix = network.adjacency_matrix.copy()
    initial_robustness = q_robust.calculate_robustness(initial_matrix)
    
    print(f"初始鲁棒性: {initial_robustness:.4f}")
    print(f"优化后鲁棒性: {final_fitness:.4f}")
    print(f"提升幅度: {(final_fitness - initial_robustness)/initial_robustness*100:.2f}%")
    
    visualize_network(optimized_matrix, "Q-Robust优化后的网络拓扑")
    
    plt.figure(figsize=(10,6))
    plt.plot(history, label='Q-Robust')
    plt.axhline(initial_robustness, color='r', linestyle='--', label='初始拓扑')
    plt.xlabel('迭代次数')
    plt.ylabel('鲁棒性(R值)')
    plt.title('Q-Robust算法进化曲线')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    evaluate_attack_comparison([
        (initial_matrix, "初始网络"),
        (optimized_matrix, "Q-Robust优化后网络")
    ])
    G, optim_G = adj_matrix_to_nx(initial_matrix), adj_matrix_to_nx(optimized_matrix)
    plot_robustness_comparison(G, optim_G)
    plot_random_robustness_comparison(G, optim_G)
    compare_degree_distributions(G, optim_G)

if __name__ == "__main__":
    # 测试优化版本
    print("测试优化版本:")
    G = nx.erdos_renyi_graph(n=50, p=0.1)
    optimized_G = Q_Robust_method(G, max_gen=5, use_optimized=True)
    plot_robustness_comparison(G, optimized_G, title="Optimized Version")
    
    # 测试原始版本进行对比
    print("\n测试原始版本:")
    optimized_G_original = Q_Robust_method(G, max_gen=5, use_optimized=False)
    plot_robustness_comparison(G, optimized_G_original, title="Original Version")
    plot_random_robustness_comparison(G, optimized_G, title="Original Version")
    compare_degree_distributions(G, optimized_G)