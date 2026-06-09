"""
量子驱动的高效学习模型用于IoT拓扑重构
基于量子遗传算法和量子旋转门机制
"""

import numpy as np
import networkx as nx
import random
import math
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


class QuantumEnvironment:
    """
    量子环境：管理图结构和通信约束
    基于预定义的networkx图G进行重构
    """
    def __init__(self, G, comm_range=0.3, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.r = comm_range
        
        # 从图G获取节点和位置
        self.n = G.number_of_nodes()
        self.positions = np.zeros((self.n, 2))
        
        # 验证节点ID是否为0到n-1的连续整数
        if sorted(G.nodes()) != list(range(self.n)):
            raise ValueError("Graph nodes must be labeled 0 to n-1 consecutively")
        
        # 提取节点位置
        for i in range(self.n):
            if 'pos' not in G.nodes[i]:
                raise ValueError(f"Node {i} missing 'pos' attribute")
            self.positions[i] = np.array(G.nodes[i]['pos'])
        
        # 计算所有节点对的距离并确定可能的边
        self.dist_matrix = np.zeros((self.n, self.n))
        self.possible_edges = []  # 存储物理上允许连接的边索引 (i, j)
        
        # 遍历所有节点对，找出通信半径内的所有可能边
        for u in range(self.n):
            for v in range(u + 1, self.n):
                dist = np.linalg.norm(self.positions[u] - self.positions[v])
                self.dist_matrix[u][v] = self.dist_matrix[v][u] = dist
                
                if dist <= self.r:
                    self.possible_edges.append((u, v))
        
        self.m = len(self.possible_edges)  # 有效边的数量
        self.target_edges = G.number_of_edges()  # 原始边数（保持边数约束）
        print(f"[Quantum Environment] Nodes: {self.n}, Valid Edges: {self.m}, Target Edges: {self.target_edges}, Comm Range: {self.r}")
    
    def vector_to_adj(self, binary_vector):
        """将二进制向量还原为邻接矩阵，保持边数不变"""
        adj = np.zeros((self.n, self.n), dtype=int)
        selected_edges = []
        for idx, val in enumerate(binary_vector):
            if val == 1 and idx < len(self.possible_edges):
                selected_edges.append(self.possible_edges[idx])
        
        # 保持边数约束
        if len(selected_edges) > self.target_edges:
            np.random.shuffle(selected_edges)
            selected_edges = selected_edges[:self.target_edges]
        elif len(selected_edges) < self.target_edges:
            available = [e for e in self.possible_edges if e not in selected_edges]
            needed = self.target_edges - len(selected_edges)
            if len(available) >= needed:
                additional = np.random.choice(len(available), needed, replace=False)
                for idx in additional:
                    selected_edges.append(available[idx])
        
        for u, v in selected_edges:
            adj[u][v] = adj[v][u] = 1
        return adj


class QuantumChromosome:
    """
    量子染色体：使用量子比特表示
    每个量子比特用 [alpha, beta] 表示，其中 |alpha|^2 + |beta|^2 = 1
    alpha^2 表示观测到0的概率，beta^2 表示观测到1的概率
    """
    def __init__(self, length):
        self.length = length
        # 初始化量子比特：alpha和beta
        # 初始状态为叠加态：alpha = beta = 1/sqrt(2)
        sqrt2_inv = 1.0 / np.sqrt(2.0)
        self.alpha = np.full(length, sqrt2_inv, dtype=float)  # 0态振幅
        self.beta = np.full(length, sqrt2_inv, dtype=float)   # 1态振幅
    
    def observe(self):
        """
        量子测量：根据概率幅测量得到经典二进制串
        """
        binary = []
        for i in range(self.length):
            prob_1 = self.beta[i] ** 2  # 观测到1的概率
            if random.random() < prob_1:
                binary.append(1)
            else:
                binary.append(0)
        return np.array(binary)
    
    def update_quantum_gate(self, index, rotation_angle, direction):
        """
        量子旋转门更新
        :param index: 要更新的量子比特索引
        :param rotation_angle: 旋转角度
        :param direction: 旋转方向 (1表示向1态旋转，-1表示向0态旋转)
        """
        if index >= self.length:
            return
        
        # 量子旋转门矩阵
        cos_theta = math.cos(rotation_angle * direction)
        sin_theta = math.sin(rotation_angle * direction)
        
        # 更新量子态
        alpha_new = self.alpha[index] * cos_theta - self.beta[index] * sin_theta
        beta_new = self.alpha[index] * sin_theta + self.beta[index] * cos_theta
        
        # 归一化（确保概率和为1）
        norm = math.sqrt(alpha_new**2 + beta_new**2)
        if norm > 1e-10:
            self.alpha[index] = alpha_new / norm
            self.beta[index] = beta_new / norm
        else:
            # 如果归一化失败，重置为叠加态
            sqrt2_inv = 1.0 / np.sqrt(2.0)
            self.alpha[index] = sqrt2_inv
            self.beta[index] = sqrt2_inv


class RobustnessEvaluator:
    """
    鲁棒性评估器
    计算网络在攻击下的鲁棒性指标
    """
    @staticmethod
    def calculate_robustness(adj_matrix, attack_type='malicious', sample_ratio=0.3):
        """
        计算鲁棒性指标 R（优化版本：使用采样加速）
        :param adj_matrix: 邻接矩阵
        :param attack_type: 攻击类型 ('malicious' 或 'random')
        :param sample_ratio: 采样比例（只计算部分节点的攻击，加速计算）
        :return: 鲁棒性值 R
        """
        G = nx.from_numpy_array(adj_matrix)
        n = G.number_of_nodes()
        if n == 0:
            return 0.0
        
        # 使用采样加速：只计算部分节点的攻击
        max_steps = max(5, int(n * sample_ratio))
        
        nodes = list(G.nodes())
        if attack_type == 'malicious':
            # 恶意攻击：按度从高到低移除节点
            nodes_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
            removal_queue = [node for node, degree in nodes_sorted]
        else:
            # 随机攻击
            removal_queue = list(G.nodes())
            random.shuffle(removal_queue)
        
        sum_tolerance = 0.0
        G_temp = G.copy()
        
        for theta in range(min(max_steps, n)):
            if G_temp.number_of_nodes() > 0:
                try:
                    components = list(nx.connected_components(G_temp))
                    if components:
                        largest_cc = max(components, key=len)
                        phi_theta = len(largest_cc)
                    else:
                        phi_theta = 0
                except:
                    phi_theta = 0
            else:
                phi_theta = 0
            
            remaining_count = n - theta
            t_theta = phi_theta / remaining_count if remaining_count > 0 else 0
            sum_tolerance += t_theta
            
            if theta < len(removal_queue):
                node_to_remove = removal_queue[theta]
                if G_temp.has_node(node_to_remove):
                    G_temp.remove_node(node_to_remove)
            else:
                break
        
        # 归一化到完整攻击的期望值
        return (sum_tolerance / max_steps) * (max_steps / n) if max_steps > 0 else 0.0


class QuantumLearningModel:
    """
    量子驱动的高效学习模型
    结合量子遗传算法和自适应学习机制
    """
    def __init__(self, env, pop_size=30, max_iterations=50):
        self.env = env
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.evaluator = RobustnessEvaluator()
        
        # 初始化量子种群
        self.quantum_population = [QuantumChromosome(env.m) for _ in range(pop_size)]
        
        # 经典种群（通过测量量子态得到）
        self.classical_population = []
        self.fitness_values = []
        
        # 最佳解
        self.best_fitness = -1.0
        self.best_solution = None
        self.best_adj = None
        
        # 自适应参数
        self.rotation_angle_base = 0.1 * math.pi  # 基础旋转角度
        self.learning_rate = 0.1  # 学习率
        
    def initialize_population(self):
        """初始化经典种群"""
        self.classical_population = []
        self.fitness_values = []
        
        for q_chrom in self.quantum_population:
            binary_vec = q_chrom.observe()
            adj = self.env.vector_to_adj(binary_vec)
            fitness = self.evaluator.calculate_robustness(adj)
            
            self.classical_population.append(binary_vec)
            self.fitness_values.append(fitness)
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = binary_vec.copy()
                self.best_adj = adj.copy()
    
    def quantum_rotation_gate_update(self, q_chrom, classical_solution, best_solution, index):
        """
        量子旋转门更新机制
        :param q_chrom: 量子染色体
        :param classical_solution: 当前经典解
        :param best_solution: 最佳解
        :param index: 量子比特索引
        """
        if index >= len(classical_solution) or index >= len(best_solution):
            return
        
        current_bit = classical_solution[index]
        best_bit = best_solution[index]
        
        # 确定旋转方向
        if current_bit == 0 and best_bit == 1:
            # 当前是0，最佳是1，向1态旋转
            direction = 1
        elif current_bit == 1 and best_bit == 0:
            # 当前是1，最佳是0，向0态旋转
            direction = -1
        else:
            # 当前和最佳相同，不旋转或小幅度旋转
            direction = 0
        
        if direction != 0:
            # 自适应旋转角度：根据适应度差异调整
            rotation_angle = self.rotation_angle_base * (1 + self.learning_rate)
            q_chrom.update_quantum_gate(index, rotation_angle, direction)
    
    def crossover(self, parent1, parent2):
        """交叉操作：单点交叉"""
        if len(parent1) != len(parent2):
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    
    def mutation(self, solution, mutation_rate=0.01):
        """变异操作"""
        mutated = solution.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = 1 - mutated[i]
        return mutated
    
    def selection(self):
        """选择操作：锦标赛选择"""
        tournament_size = 3
        selected_indices = []
        
        for _ in range(self.pop_size):
            candidates = random.sample(range(self.pop_size), min(tournament_size, self.pop_size))
            winner = max(candidates, key=lambda i: self.fitness_values[i])
            selected_indices.append(winner)
        
        return selected_indices
    
    def run(self, verbose=True):
        """运行量子学习算法（优化版本：使用采样加速）"""
        import time
        start_time = time.time()
        
        if verbose:
            print(f"  [QDLM] 初始化种群 (大小: {self.pop_size})...")
        
        # 初始化
        self.initialize_population()
        if verbose:
            elapsed = time.time() - start_time
            print(f"  [QDLM] 初始化完成 | Best R = {self.best_fitness:.4f} (耗时: {elapsed:.2f}s)")
        
        if verbose:
            print(f"  [QDLM] 开始优化 ({self.max_iterations} 代)...")
        
        # 主循环
        for iteration in range(self.max_iterations):
            iter_start = time.time()
            
            # 1. 选择
            selected_indices = self.selection()
            
            # 2. 更新量子种群（量子旋转门）
            for i, selected_idx in enumerate(selected_indices):
                q_chrom = self.quantum_population[i]
                classical_sol = self.classical_population[selected_idx]
                
                # 对每个量子比特应用旋转门
                for j in range(min(len(classical_sol), q_chrom.length)):
                    self.quantum_rotation_gate_update(q_chrom, classical_sol, 
                                                      self.best_solution, j)
            
            # 3. 测量量子态得到新的经典种群（使用采样加速）
            new_classical = []
            new_fitness = []
            
            for q_chrom in self.quantum_population:
                binary_vec = q_chrom.observe()
                adj = self.env.vector_to_adj(binary_vec)
                fitness = self.evaluator.calculate_robustness(adj, sample_ratio=0.3)  # 使用采样加速
                
                new_classical.append(binary_vec)
                new_fitness.append(fitness)
                
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = binary_vec.copy()
                    self.best_adj = adj.copy()
                    if verbose:
                        elapsed = time.time() - start_time
                        print(f"  [QDLM] Gen {iteration + 1}/{self.max_iterations}: New Best R = {fitness:.4f} (耗时: {elapsed:.2f}s)")
            
            # 4. 交叉和变异（可选，增强探索能力）
            if iteration % 5 == 0:  # 每5代执行一次
                for i in range(0, self.pop_size - 1, 2):
                    if random.random() < 0.7:  # 交叉概率
                        child1, child2 = self.crossover(new_classical[i], new_classical[i+1])
                        new_classical[i] = self.mutation(child1)
                        new_classical[i+1] = self.mutation(child2)
                        
                        # 重新评估（使用采样加速）
                        adj1 = self.env.vector_to_adj(new_classical[i])
                        adj2 = self.env.vector_to_adj(new_classical[i+1])
                        new_fitness[i] = self.evaluator.calculate_robustness(adj1, sample_ratio=0.3)
                        new_fitness[i+1] = self.evaluator.calculate_robustness(adj2, sample_ratio=0.3)
            
            # 更新种群
            self.classical_population = new_classical
            self.fitness_values = new_fitness
            
            # 自适应调整旋转角度（随着迭代减小，增强局部搜索）
            self.rotation_angle_base *= 0.99
            
            # 减少打印频率
            if verbose and (iteration % 5 == 0 or iteration == self.max_iterations - 1):
                elapsed = time.time() - start_time
                print(f"  [QDLM] Gen {iteration + 1}/{self.max_iterations}: Current Best R = {self.best_fitness:.4f} (耗时: {elapsed:.2f}s)")
        
        if verbose:
            total_time = time.time() - start_time
            edge_count = np.sum(self.best_adj) // 2
            print(f"  [QDLM] 优化完成 | Best R: {self.best_fitness:.4f} | Edges: {edge_count} | 总耗时: {total_time:.2f}s")
        
        return self.best_adj


# ================= 运行入口 =================
if __name__ == "__main__":
    # 创建示例图
    G = nx.random_geometric_graph(30, radius=0.3, seed=42)
    G = nx.convert_node_labels_to_integers(G)
    
    # 为节点添加位置属性
    for i in G.nodes():
        G.nodes[i]['pos'] = tuple(G.nodes[i]['pos'])
    
    # 创建环境
    env = QuantumEnvironment(G=G, comm_range=0.3)
    
    # 初始化量子学习模型
    qlm = QuantumLearningModel(env, pop_size=30, max_iterations=50)
    
    # 运行优化
    best_adj = qlm.run()
    
    # 可视化
    try:
        import matplotlib.pyplot as plt
        
        G_final = nx.from_numpy_array(best_adj)
        pos_final = {i: G.nodes[i]['pos'] for i in range(env.n)}
        
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G_final, pos_final, node_size=120, node_color='skyblue', alpha=0.9)
        nx.draw_networkx_edges(G_final, pos_final, alpha=0.7, edge_color='gray')
        plt.title(f"Quantum Learning Optimized Topology (R={qlm.best_fitness:.4f})", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("quantum_optimized_topology.png", dpi=300)
        plt.show()
        
    except ImportError:
        print("Matplotlib not found, skipping visualization.")
