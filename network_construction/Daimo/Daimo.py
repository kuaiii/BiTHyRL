import networkx as nx
import random
import copy
import itertools
import time
from typing import List, Tuple, Dict, Set
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats as scipy_stats
import numpy as np


# ---------------- 优化后的模体分析器 ----------------
class MotifAnalyzer:
    """优化后的网络三模体分析器。
    
    关键洞察：3节点模体中的边 = 总边数 - 孤立边数（两端点度均为1的边）。
    因此无需 O(N^3) 枚举，可直接 O(M) 计算。
    """
    
    def __init__(self, maxsize: int = 1024):
        self._motif_cache = {}
        self._maxsize = maxsize

    def calculate_motif_edges(self, graph: nx.Graph) -> int:
        """计算参与任意3节点模体的边数（等价于总边数 - 孤立边数）。"""
        graph_key = tuple(sorted(graph.edges()))
        if graph_key in self._motif_cache:
            return self._motif_cache[graph_key]
        
        total_edges = graph.number_of_edges()
        if total_edges == 0:
            return 0
        
        # 孤立边：两端点度均为1的边
        isolated_edges = 0
        for u, v in graph.edges():
            if graph.degree(u) == 1 and graph.degree(v) == 1:
                isolated_edges += 1
        
        result = total_edges - isolated_edges
        # 简单LRU：缓存满时清空（实际遗传算法中图差异大，命中率低）
        if len(self._motif_cache) >= self._maxsize:
            self._motif_cache.clear()
        self._motif_cache[graph_key] = result
        return result

    def clear_cache(self):
        """清理缓存"""
        self._motif_cache.clear()


# ---------------- 优化后的鲁棒性指标 ----------------
class RobustnessMetric:
    def __init__(self, maxsize: int = 2048):
        self.motif_analyzer = MotifAnalyzer(maxsize=maxsize)
        self._attack_sequence_cache = {}
        self._robustness_cache = {}
        self._maxsize = maxsize

    def calculate_robustness(self, graph: nx.Graph, failure_sequence: List[int] = None, max_failures: int = None) -> float:
        if graph.number_of_nodes() < 3:
            return 0.0

        N = graph.number_of_nodes()
        total_edges = graph.number_of_edges()
        if total_edges == 0:
            return 0.0

        if failure_sequence is None:
            failure_sequence = self._generate_high_degree_attack_sequence(graph)

        # 默认截断到 N//2，后续节点对鲁棒性积分贡献极小
        if max_failures is None:
            max_failures = max(1, N // 2)
        else:
            max_failures = min(max_failures, max(1, N // 2))

        # 鲁棒性缓存：基于(边集, 截断步数)
        cache_key = (tuple(sorted(graph.edges())), max_failures)
        if cache_key in self._robustness_cache:
            return self._robustness_cache[cache_key]

        total_mc = 0
        current_graph = graph.copy()

        for i, failed_node in enumerate(failure_sequence[:max_failures]):
            if failed_node in current_graph:
                current_graph.remove_node(failed_node)

            if current_graph.number_of_nodes() < 3:
                break

            if current_graph.number_of_edges() < 2:
                continue

            # O(M) 直接计算，替代 O(N^3) 模体枚举
            mc_k = self.motif_analyzer.calculate_motif_edges(current_graph)
            total_mc += mc_k

        I = (3 / (N - 2)) * (total_mc / total_edges) if total_edges > 0 else 0
        result = min(I, 1.0)
        
        if len(self._robustness_cache) >= self._maxsize:
            self._robustness_cache.clear()
        self._robustness_cache[cache_key] = result
        return result

    def _generate_high_degree_attack_sequence(self, graph: nx.Graph) -> List[int]:
        graph_key = tuple(sorted(graph.edges()))
        if graph_key in self._attack_sequence_cache:
            return self._attack_sequence_cache[graph_key]
        
        sequence = [node for node, _ in sorted(graph.degree, key=lambda x: x[1], reverse=True)]
        self._attack_sequence_cache[graph_key] = sequence
        return sequence


# ---------------- 优化后的拓扑染色体 ----------------
class TopologyChromosome:
    def __init__(self, graph: nx.Graph, positions=None, comm_range=None):
        self.graph = graph.copy()
        self._fitness = None
        self._graph_hash = None
        self.positions = positions  # dict: node -> (x, y)
        self.comm_range = comm_range  # float
        self._constrained_graph = None

    def _apply_comm_constraint(self) -> nx.Graph:
        """施加通信范围约束，只保留有效边，不补充新边"""
        if self.positions is None or self.comm_range is None:
            return self.graph
        
        G = nx.Graph()
        G.add_nodes_from(self.graph.nodes())
        for n in self.graph.nodes():
            G.nodes[n]['pos'] = self.positions[n]
        
        # 只保留通信范围内的边
        for u, v in self.graph.edges():
            x1, y1 = self.positions[u]
            x2, y2 = self.positions[v]
            if ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 <= self.comm_range:
                G.add_edge(u, v)
        
        return G

    def get_constrained_graph(self) -> nx.Graph:
        """获取施加约束后的图"""
        if self._constrained_graph is None:
            self._constrained_graph = self._apply_comm_constraint()
        return self._constrained_graph

    def get_fitness(self) -> float:
        if self._fitness is None:
            # 复用类级鲁棒性度量实例以命中缓存
            if not hasattr(TopologyChromosome, '_shared_robustness_metric'):
                TopologyChromosome._shared_robustness_metric = RobustnessMetric()
            metric = TopologyChromosome._shared_robustness_metric
            self._fitness = metric.calculate_robustness(self.graph)
        return self._fitness

    def get_graph_hash(self) -> int:
        """获取图的哈希值用于比较"""
        if self._graph_hash is None:
            edges = tuple(sorted(self.graph.edges()))
            self._graph_hash = hash(edges)
        return self._graph_hash

    def invalidate_fitness(self):
        """当图结构改变时使适应度失效"""
        self._fitness = None
        self._graph_hash = None
        self._constrained_graph = None


# ---------------- 优化后的遗传算子 ----------------
class GeneticOperators:
    def __init__(self, cross_prob: float = 0.7, mutation_prob: float = 0.3):
        self.cross_prob = cross_prob
        self.mutation_prob = mutation_prob

    def crossover(self, parent1: TopologyChromosome, parent2: TopologyChromosome) -> Tuple[TopologyChromosome, TopologyChromosome]:
        if random.random() > self.cross_prob:
            # 避免不必要的深拷贝
            return self._shallow_copy_chromosome(parent1), self._shallow_copy_chromosome(parent2)

        child1_graph = parent1.graph.copy()
        child2_graph = parent2.graph.copy()
        self._optimized_crossover(child1_graph, child2_graph)
        
        child1 = TopologyChromosome(child1_graph, positions=parent1.positions, comm_range=parent1.comm_range)
        child2 = TopologyChromosome(child2_graph, positions=parent2.positions, comm_range=parent2.comm_range)
        return child1, child2

    def _optimized_crossover(self, graph1: nx.Graph, graph2: nx.Graph):
        """优化的交叉操作"""
        edges1 = list(graph1.edges())
        edges2 = list(graph2.edges())
        
        if len(edges1) >= 2 and len(edges2) >= 2:
            # 随机选择多条边进行交换，增加多样性
            num_swaps = min(2, len(edges1), len(edges2))
            
            for _ in range(num_swaps):
                if not edges1 or not edges2:
                    break
                    
                edge1 = random.choice(edges1)
                edge2 = random.choice(edges2)
                
                if not graph2.has_edge(*edge1) and not graph1.has_edge(*edge2):
                    graph1.remove_edge(*edge1)
                    graph2.remove_edge(*edge2)
                    graph1.add_edge(*edge2)
                    graph2.add_edge(*edge1)
                    
                    # 从列表中移除已处理的边
                    edges1.remove(edge1)
                    edges2.remove(edge2)

    def mutation(self, chromosome: TopologyChromosome) -> TopologyChromosome:
        if random.random() > self.mutation_prob:
            return self._shallow_copy_chromosome(chromosome)
        
        mutated_graph = chromosome.graph.copy()
        self._optimized_mutation(mutated_graph, chromosome.positions, chromosome.comm_range)
        child = TopologyChromosome(mutated_graph, positions=chromosome.positions, comm_range=chromosome.comm_range)
        return child

    def _optimized_mutation(self, graph: nx.Graph, positions=None, comm_range=None):
        """优化的变异操作 - 只使用edge_swap保持边数不变"""
        edges = list(graph.edges())
        
        if len(edges) >= 2:
            # 边交换变异（保持边数不变）
            edge1, edge2 = random.sample(edges, 2)
            if len(set(edge1 + edge2)) == 4:  # 确保4个不同节点
                u1, v1 = edge1
                u2, v2 = edge2
                new_edge1 = (u1, u2)
                new_edge2 = (v1, v2)
                
                # 检查通信范围约束（如果提供了位置信息）
                valid1 = True
                valid2 = True
                if positions is not None and comm_range is not None:
                    x1, y1 = positions[u1]; x2, y2 = positions[u2]
                    valid1 = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 <= comm_range
                    x1, y1 = positions[v1]; x2, y2 = positions[v2]
                    valid2 = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 <= comm_range
                
                if valid1 and valid2 and not graph.has_edge(*new_edge1) and not graph.has_edge(*new_edge2):
                    graph.remove_edge(*edge1)
                    graph.remove_edge(*edge2)
                    graph.add_edge(*new_edge1)
                    graph.add_edge(*new_edge2)

    def _shallow_copy_chromosome(self, chromosome: TopologyChromosome) -> TopologyChromosome:
        """创建染色体的浅拷贝"""
        new_chromosome = TopologyChromosome(chromosome.graph.copy(), 
                                             positions=chromosome.positions, 
                                             comm_range=chromosome.comm_range)
        return new_chromosome


# ---------------- 优化后的选择算子 ----------------
class SelectionOperators:
    @staticmethod
    def roulette_selection(population: List[TopologyChromosome], k: int) -> List[TopologyChromosome]:
        # 预计算所有适应度
        fitnesses = [chromo.get_fitness() for chromo in population]
        total_fitness = sum(fitnesses)
        
        if total_fitness == 0:
            return random.sample(population, min(k, len(population)))
        
        selected = []
        for _ in range(k):
            rand_val = random.uniform(0, total_fitness)
            cumsum = 0
            for i, fitness in enumerate(fitnesses):
                cumsum += fitness
                if cumsum >= rand_val:
                    selected.append(TopologyChromosome(population[i].graph.copy(),
                                                        positions=population[i].positions,
                                                        comm_range=population[i].comm_range))
                    break
        return selected

    @staticmethod
    def tournament_selection(population: List[TopologyChromosome], k: int, tournament_size: int = 3) -> List[TopologyChromosome]:
        selected = []
        for _ in range(k):
            # 使用更小的锦标赛规模提高速度
            actual_tournament_size = min(tournament_size, len(population))
            tournament = random.sample(population, actual_tournament_size)
            winner = max(tournament, key=lambda x: x.get_fitness())
            selected.append(TopologyChromosome(winner.graph.copy(),
                                                positions=winner.positions,
                                                comm_range=winner.comm_range))
        return selected

    @staticmethod
    def knn_selection(population: List[TopologyChromosome], k: int) -> List[TopologyChromosome]:
        # 使用numpy进行快速排序
        fitnesses = [chromo.get_fitness() for chromo in population]
        indices = np.argsort(fitnesses)[::-1][:k]  # 降序排列取前k个
        return [TopologyChromosome(population[i].graph.copy(),
                                    positions=population[i].positions,
                                    comm_range=population[i].comm_range) for i in indices]


# ---------------- 优化后的局部优化器 ----------------
class LocalOptimizer:
    def __init__(self, selection_type: str, population_size: int = 20, generations: int = 20):
        self.selection_type = selection_type
        self.population_size = population_size
        self.generations = generations
        self.genetic_ops = GeneticOperators()
        self.selector = SelectionOperators()

    def optimize(self, initial_population: List[TopologyChromosome]) -> TopologyChromosome:
        population = initial_population[:self.population_size]
        while len(population) < self.population_size:
            chosen = random.choice(initial_population)
            population.append(TopologyChromosome(chosen.graph.copy(),
                                                  positions=chosen.positions,
                                                  comm_range=chosen.comm_range))

        best_fitness = 0
        stagnation_count = 0
        
        for generation in range(self.generations):
            if self.selection_type == "roulette":
                selected = self.selector.roulette_selection(population, self.population_size)
            elif self.selection_type == "tournament":
                selected = self.selector.tournament_selection(population, self.population_size)
            else:
                selected = self.selector.knn_selection(population, self.population_size)

            new_population = []
            for i in range(0, len(selected) - 1, 2):
                parent1, parent2 = selected[i], selected[i + 1]
                child1, child2 = self.genetic_ops.crossover(parent1, parent2)
                child1 = self.genetic_ops.mutation(child1)
                child2 = self.genetic_ops.mutation(child2)
                new_population.extend([child1, child2])

            while len(new_population) < self.population_size:
                chosen = random.choice(selected)
                new_population.append(TopologyChromosome(chosen.graph.copy(),
                                                          positions=chosen.positions,
                                                          comm_range=chosen.comm_range))

            population = new_population[:self.population_size]
            
            # 早停机制（收紧阈值以加速）
            current_best = max(population, key=lambda x: x.get_fitness()).get_fitness()
            if current_best > best_fitness:
                best_fitness = current_best
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            if stagnation_count >= 10:
                break

        best = max(population, key=lambda x: x.get_fitness())
        return best


# ---------------- 优化后的全局优化器 ----------------
class GlobalOptimizer:
    def __init__(self, population_size: int = 10, generations: int = 20):
        self.population_size = population_size
        self.generations = generations
        self.genetic_ops = GeneticOperators()
        self.selector = SelectionOperators()
        self.population = []

    def update_population(self, new_individual: TopologyChromosome):
        if len(self.population) < self.population_size:
            self.population.append(TopologyChromosome(new_individual.graph.copy(),
                                                       positions=new_individual.positions,
                                                       comm_range=new_individual.comm_range))
        else:
            worst = min(self.population, key=lambda x: x.get_fitness())
            if new_individual.get_fitness() > worst.get_fitness():
                self.population.remove(worst)
                self.population.append(TopologyChromosome(new_individual.graph.copy(),
                                                           positions=new_individual.positions,
                                                           comm_range=new_individual.comm_range))

    def optimize(self) -> TopologyChromosome:
        best_fitness = 0
        stagnation_count = 0
        
        for generation in range(self.generations):
            if len(self.population) < 2:
                break
                
            selected = self.selector.tournament_selection(self.population, len(self.population))
            new_population = []
            
            for i in range(0, len(selected) - 1, 2):
                parent1, parent2 = selected[i], selected[i + 1]
                child1, child2 = self.genetic_ops.crossover(parent1, parent2)
                child1 = self.genetic_ops.mutation(child1)
                child2 = self.genetic_ops.mutation(child2)
                new_population.extend([child1, child2])
                
            self.population = new_population[:self.population_size]
            
            # 早停机制（收紧）
            if self.population:
                current_best = max(self.population, key=lambda x: x.get_fitness()).get_fitness()
                if current_best > best_fitness:
                    best_fitness = current_best
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    
                if stagnation_count >= 15:
                    break
        
        return max(self.population, key=lambda x: x.get_fitness()) if self.population else None


# ---------------- 优化后的DAiMo主优化器 ----------------
class DAiMoOptimizer:
    def __init__(self, num_local_workers: int = 6, local_generations: int = 50, global_generations: int = 50):
        self.num_local_workers = num_local_workers
        self.local_generations = local_generations
        self.global_generations = global_generations
        self.global_optimizer = GlobalOptimizer(population_size=10, generations=global_generations)

    def optimize(self, initial_graph: nx.Graph) -> nx.Graph:
        print("开始 DAiMo 优化...")
        start_time = time.time()

        # 提取位置信息和通信范围
        positions = nx.get_node_attributes(initial_graph, 'pos')
        comm_range = None
        for n in initial_graph.nodes():
            if 'comm_range' in initial_graph.nodes[n]:
                comm_range = initial_graph.nodes[n]['comm_range']
                break
        # 如果没有 comm_range 属性，默认使用 200
        if comm_range is None:
            comm_range = 200

        initial_population = self._generate_initial_population(initial_graph, 5)  # 减少初始种群
        initial_chromosomes = [TopologyChromosome(g) for g in initial_population]

        selection_types = ["roulette", "tournament", "knn"]
        local_results = []

        for i in range(self.num_local_workers):
            print(f"运行局部优化器 {i+1}: {selection_types[i % 3]}")
            optimizer = LocalOptimizer(selection_types[i % 3], 
                                     population_size=10,  # 与论文一致
                                     generations=self.local_generations)
            best = optimizer.optimize(initial_chromosomes)
            local_results.append(best)

        for result in local_results:
            self.global_optimizer.update_population(result)

        best_global = self.global_optimizer.optimize()
        
        # 返回优化后的图
        if best_global:
            optimized_graph = best_global.graph
        else:
            optimized_graph = initial_graph

        end_time = time.time()
        print(f"优化完成！耗时 {end_time - start_time:.2f}s")
        
        # 清理缓存
        MotifAnalyzer().clear_cache()
        
        metric = RobustnessMetric()
        initial_robustness = metric.calculate_robustness(initial_graph)
        final_robustness = best_global.get_fitness() if best_global else initial_robustness
        print(f"鲁棒性提升: {initial_robustness:.4f} → {final_robustness:.4f}")
        return optimized_graph

    def _generate_initial_population(self, base_graph: nx.Graph, population_size: int) -> List[nx.Graph]:
        population = [base_graph.copy()]
        for _ in range(population_size - 1):
            g = base_graph.copy()
            self._random_edge_swap(g, num_swaps=1)  # 减少交换次数
            population.append(g)
        return population

    def _random_edge_swap(self, graph: nx.Graph, num_swaps: int = 1):
        edges = list(graph.edges())
        if len(edges) < 4:
            return
        for _ in range(num_swaps):
            if len(edges) < 2:
                break
            edge1, edge2 = random.sample(edges, 2)
            if len(set(edge1 + edge2)) == 4:
                u1, v1 = edge1
                u2, v2 = edge2
                new_edge1 = (u1, u2)
                new_edge2 = (v1, v2)
                if not graph.has_edge(*new_edge1) and not graph.has_edge(*new_edge2):
                    graph.remove_edge(*edge1)
                    graph.remove_edge(*edge2)
                    graph.add_edge(*new_edge1)
                    graph.add_edge(*new_edge2)
                    edges.remove(edge1)
                    edges.remove(edge2)


def plot_robustness_comparison(initial_graph, optimized_graph, title="Daimo"):
    """使用NetworkX绘制优化前后的鲁棒性比较图
    
    Args:
        initial_graph: NetworkX图对象，初始拓扑
        optimized_graph: NetworkX图对象，优化后的拓扑
        title: 图表标题
    """
    # 计算攻击后的网络连通性
    def calculate_connectivity_after_attacks(graph):
        # 创建图的拷贝
        g = graph.copy()
        n = g.number_of_nodes()
        connectivity = [1.0]  # 初始完全连通
        
        for _ in range(n):
            if g.number_of_nodes() == 0:
                connectivity.extend([0] * (n - len(connectivity) + 1))
                break
                
            # HDA攻击 - 删除度数最大的节点
            degrees = dict(g.degree())
            if not degrees:
                break
                
            max_degree_node = max(degrees, key=degrees.get)
            g.remove_node(max_degree_node)
            
            # 计算最大连通分量比例
            if g.number_of_nodes() > 0:
                # 获取所有连通分量
                components = list(nx.connected_components(g))
                largest_cc_size = max(len(comp) for comp in components) if components else 0
                connectivity.append(largest_cc_size / n)
            else:
                connectivity.append(0)
        
        return connectivity[:n+1]  # 限制为n+1个点（包括初始状态）
    
    # 计算鲁棒性值 - 基于NetworkX
    def calculate_robustness(graph):
        """计算网络的鲁棒性值"""
        g = graph.copy()
        n = g.number_of_nodes()
        robustness_sum = 0.0

        # 预先计算节点度数并按从大到小排序（静态攻击）
        degrees = dict(g.degree())
        node_order = sorted(degrees.keys(), key=lambda node: degrees[node], reverse=True)

        # 遍历每一步攻击，记录最大连通分量比例
        for step in range(n + 1):
            if g.number_of_nodes() > 0:
                # 获取最大连通分量大小
                components = list(nx.connected_components(g))
                largest_cc_size = max(len(comp) for comp in components) if components else 0
                robustness_sum += largest_cc_size / n
            else:
                robustness_sum += 0.0

            if step < n and node_order:
                # 删除预定的节点
                node_to_remove = node_order[step]
                if g.has_node(node_to_remove):
                    g.remove_node(node_to_remove)

        return robustness_sum / (n + 1)
    
    initial_connectivity = calculate_connectivity_after_attacks(initial_graph)
    optimized_connectivity = calculate_connectivity_after_attacks(optimized_graph)
    
    # 计算鲁棒性值
    initial_robustness = calculate_robustness(initial_graph)
    optimized_robustness = calculate_robustness(optimized_graph)
    
    # 使用英文标签绘图，避免字体问题
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
    
    # 打印鲁棒性提升信息
    improvement = (optimized_robustness - initial_robustness) / initial_robustness * 100
    print(f"Robustness improvement: {improvement:.2f}%")
    
    return initial_robustness, optimized_robustness, improvement


def plot_random_robustness_comparison(initial_graph, optimized_graph, title="Unity", attack_type="random", num_simulations=10):
    """使用NetworkX绘制优化前后的鲁棒性比较图
    
    Args:
        initial_graph: NetworkX图对象，初始拓扑
        optimized_graph: NetworkX图对象，优化后的拓扑
        title: 图表标题
        attack_type: 攻击类型，"random"为随机攻击，"hda"为高度中心性攻击
        num_simulations: 随机攻击模式下的模拟次数，求平均值
    """
    # 计算攻击后的网络连通性 - 随机攻击
    def calculate_connectivity_random_attacks(graph):
        n = graph.number_of_nodes()
        all_results = []
        
        # 进行多次随机攻击模拟
        for _ in range(num_simulations):
            g = graph.copy()
            connectivity = [1.0]  # 初始完全连通
            nodes = list(g.nodes())
            # 随机打乱节点顺序
            random.shuffle(nodes)
            
            for node in nodes:
                g.remove_node(node)
                
                # 计算最大连通分量比例
                if g.number_of_nodes() > 0:
                    components = list(nx.connected_components(g))
                    largest_cc_size = max(len(comp) for comp in components) if components else 0
                    connectivity.append(largest_cc_size / n)
                else:
                    connectivity.append(0)
            
            # 确保结果长度一致
            if len(connectivity) < n + 1:
                connectivity.extend([0] * (n + 1 - len(connectivity)))
            
            all_results.append(connectivity[:n+1])
        
        # 计算多次模拟的平均值
        avg_connectivity = [sum(col)/num_simulations for col in zip(*all_results)]
        return avg_connectivity
    
    # 计算攻击后的网络连通性 - 高度中心性攻击
    def calculate_connectivity_hda(graph):
        g = graph.copy()
        n = g.number_of_nodes()
        connectivity = [1.0]  # 初始完全连通
        
        for _ in range(n):
            if g.number_of_nodes() == 0:
                connectivity.extend([0] * (n - len(connectivity) + 1))
                break
                
            # HDA攻击 - 删除度数最大的节点
            degrees = dict(g.degree())
            if not degrees:
                break
                
            max_degree_node = max(degrees, key=degrees.get)
            g.remove_node(max_degree_node)
            
            # 计算最大连通分量比例
            if g.number_of_nodes() > 0:
                components = list(nx.connected_components(g))
                largest_cc_size = max(len(comp) for comp in components) if components else 0
                connectivity.append(largest_cc_size / n)
            else:
                connectivity.append(0)
        
        return connectivity[:n+1]  # 限制为n+1个点（包括初始状态）
    
    # 计算鲁棒性值
    def calculate_robustness(connectivity):
        """根据连通性数组计算网络的鲁棒性值"""
        n = len(connectivity) - 1  # 减去初始状态
        return sum(connectivity) / (n + 1)
    
    # 根据攻击类型选择相应的函数
    if attack_type.lower() == "random":
        initial_connectivity = calculate_connectivity_random_attacks(initial_graph)
        optimized_connectivity = calculate_connectivity_random_attacks(optimized_graph)
        attack_label = "Random Attack"
    else:  # 默认为hda攻击
        initial_connectivity = calculate_connectivity_hda(initial_graph)
        optimized_connectivity = calculate_connectivity_hda(optimized_graph)
        attack_label = "High Degree Attack"
    
    # 计算鲁棒性值
    initial_robustness = calculate_robustness(initial_connectivity)
    optimized_robustness = calculate_robustness(optimized_connectivity)
    
    # 使用英文标签绘图，避免字体问题
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
    
    # 打印鲁棒性提升信息
    improvement = (optimized_robustness - initial_robustness) / initial_robustness * 100
    print(f"Robustness improvement ({attack_label}): {improvement:.2f}%")
    
    return initial_robustness, optimized_robustness, improvement


def compare_degree_distributions(initial_graph, optimized_graph, 
                                 initial_name="Initial Network", 
                                 optimized_name="Optimized Network",
                                 show_plot=True):
    """
    比较两个NetworkX图的度分布
    
    参数:
        initial_graph (nx.Graph): 初始网络
        optimized_graph (nx.Graph): 优化后的网络
        initial_name (str): 初始网络的标签名称
        optimized_name (str): 优化后网络的标签名称
        show_plot (bool): 是否显示对比图
        
    返回:
        dict: 包含度分布统计信息的字典
    """
    # 获取度序列
    initial_degrees = [d for _, d in initial_graph.degree()]
    optimized_degrees = [d for _, d in optimized_graph.degree()]
    
    # 计算基本统计数据
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
    
    # 计算变化百分比
    node_change = ((stats["optimized"]["nodes"] - stats["initial"]["nodes"]) / 
                   stats["initial"]["nodes"]) * 100
    edge_change = ((stats["optimized"]["edges"] - stats["initial"]["edges"]) / 
                   stats["initial"]["edges"]) * 100
    mean_degree_change = ((stats["optimized"]["mean_degree"] - stats["initial"]["mean_degree"]) / 
                          stats["initial"]["mean_degree"]) * 100
    max_degree_change = ((stats["optimized"]["max_degree"] - stats["initial"]["max_degree"]) / 
                         stats["initial"]["max_degree"]) * 100
    
    # 执行KS检验比较度分布
    ks_stat, ks_pvalue = scipy_stats.ks_2samp(initial_degrees, optimized_degrees)
    
    # 打印结果
    print("Degree Distribution Analysis:")
    print(f"1. Nodes: {stats['initial']['nodes']} -> {stats['optimized']['nodes']} (Change: {node_change:.2f}%)")
    print(f"2. Edges: {stats['initial']['edges']} -> {stats['optimized']['edges']} (Change: {edge_change:.2f}%)")
    print(f"3. Mean degree: {stats['initial']['mean_degree']:.2f} -> {stats['optimized']['mean_degree']:.2f} (Change: {mean_degree_change:.2f}%)")
    print(f"4. Max degree: {stats['initial']['max_degree']} -> {stats['optimized']['max_degree']} (Change: {max_degree_change:.2f}%)")
    print(f"5. KS test: p={ks_pvalue:.4f} {'(Similar distributions)' if ks_pvalue > 0.05 else '(Different distributions)'}")
    
    # 如果需要显示图表
    if show_plot:
        # 创建度分布直方图
        plt.figure(figsize=(10, 6))
        
        # 确定合适的bins
        max_degree = max(stats["initial"]["max_degree"], stats["optimized"]["max_degree"])
        bins = range(0, max_degree + 2)
        
        # 绘制直方图
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
    
    # 返回统计信息
    return {
        "initial": {
            "nodes": stats["initial"]["nodes"],
            "edges": stats["initial"]["edges"],
            "mean_degree": stats["initial"]["mean_degree"],
            "max_degree": stats["initial"]["max_degree"]
        },
        "optimized": {
            "nodes": stats["optimized"]["nodes"],
            "edges": stats["optimized"]["edges"],
            "mean_degree": stats["optimized"]["mean_degree"],
            "max_degree": stats["optimized"]["max_degree"]
        },
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

def DaiMo_method(initial_graph: nx.Graph) -> nx.Graph:
    """
    优化网络拓扑鲁棒性的主接口函数
    
    Args:
        initial_graph: 初始网络拓扑图
        
    Returns:
        优化后的网络拓扑图
    """
    daimo = DAiMoOptimizer(
        num_local_workers=3,
        local_generations=5,
        global_generations=5
    )
    return daimo.optimize(initial_graph)

# ---------------- 主运行示例 ----------------
if __name__ == "__main__":
    # 设置随机种子以保证结果可重复
    random.seed(42)
    np.random.seed(42)
    
    # 创建测试网络
    print("创建测试网络...")
    G = nx.barabasi_albert_graph(100, 3, seed=42)
    print(f"初始网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # 运行优化
    daimo = DAiMoOptimizer(
        num_local_workers=3,
        local_generations=5,
        global_generations=5
    )
    optimized = daimo.optimize(G)
    
    print(f"优化后网络: {optimized.number_of_nodes()} 节点, {optimized.number_of_edges()} 边")
    
    # 可视化对比
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    nx.draw(G, with_labels=False, node_size=30, node_color='red', alpha=0.7)
    plt.title("Original Graph")
    plt.subplot(1, 2, 2)
    nx.draw(optimized, with_labels=False, node_size=30, node_color='blue', alpha=0.7)
    plt.title("Optimized Graph")
    plt.tight_layout()
    plt.show()
    
    # 分析结果
    print("\n=== 鲁棒性分析 ===")
    plot_robustness_comparison(G, optimized)
    plot_random_robustness_comparison(G, optimized)
    
    print("\n=== 度分布分析 ===")
    compare_degree_distributions(G, optimized)