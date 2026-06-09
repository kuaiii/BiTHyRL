# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import random
import torch
import copy
from scipy.stats import kurtosis, skew
import logging
import time
from functools import lru_cache
from src.utils.logger import get_logger
from src.topology.romen_optimization import SACAgent, NetworkTopology, ROHEMOptimizer
from src.controller.rl_optimizer import predict, calculate_reward, load_or_train_model

logger = get_logger(__name__)

# GPU 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BimodalRLOptimizer(ROHEMOptimizer):
    """
    Extends ROHEMOptimizer to include Bimodal Structure Reward and Controller Deployment.
    GPU 加速和缓存优化版本。
    """
    def __init__(self, population_size=5, generations=100, verbose=True, 
                 bimodal_weight=0.8, robustness_weight=1.0, 
                 controller_model_path='src/train/v1/checkpoints/rl_agent.pth',
                 dynamic_weight_decay=True):
        super().__init__(population_size, generations, verbose)
        self.initial_bimodal_weight = bimodal_weight
        self.bimodal_weight = bimodal_weight
        self.robustness_weight = robustness_weight
        self.controller_model_path = controller_model_path
        self.controller_model = None
        self.dynamic_weight_decay = dynamic_weight_decay
        
        # 缓存机制
        self._fitness_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 预加载控制器模型
        self.controller_model = load_or_train_model(self.controller_model_path)
        if torch.cuda.is_available():
            self.controller_model = self.controller_model.to(DEVICE)
        
    def _graph_hash(self, G):
        """计算图的哈希值用于缓存"""
        edges = tuple(sorted(G.edges()))
        return hash(edges)
        
    def _calculate_bimodality_score_fast(self, degrees_array):
        """
        快速计算双峰系数 - 使用 NumPy 向量化。
        """
        if len(degrees_array) < 3:
            return 0
        
        # 添加微小噪声避免奇异情况
        degrees = degrees_array + np.random.normal(0, 0.01, len(degrees_array))
        
        n = len(degrees)
        mean = np.mean(degrees)
        std = np.std(degrees)
        
        if std < 1e-6:
            return 0
        
        # 偏度
        s = np.mean(((degrees - mean) / std) ** 3)
        # 峰度 (Pearson's)
        k = np.mean(((degrees - mean) / std) ** 4)
        
        if k < 1e-6:
            return 0
        
        # BC calculation
        bc = (s**2 + 1) / k
        return bc

    def _calculate_combined_fitness(self, topology, use_cache=True):
        """
        Calculate combined fitness: Robustness (with controllers) + Bimodality.
        带缓存的快速版本。
        """
        G = topology.graph
        
        # 检查缓存
        if use_cache:
            graph_hash = self._graph_hash(G)
            if graph_hash in self._fitness_cache:
                self._cache_hits += 1
                return self._fitness_cache[graph_hash]
            self._cache_misses += 1
        
        # 1. Bimodality Score - 快速版本
        degrees_array = np.array([d for n, d in G.degree()])
        bimodality_score = self._calculate_bimodality_score_fast(degrees_array)
        
        # 2. 快速鲁棒性估计（不使用完整模拟）
        # 使用度方差和连通性的快速代理指标
        k = max(1, int(G.number_of_nodes() * 0.05))
        
        # 使用高度节点作为控制器（比 RL 预测快得多）
        centers = sorted(range(G.number_of_nodes()), 
                        key=lambda x: G.degree(x) if G.has_node(x) else 0, 
                        reverse=True)[:k]
        
        # 快速鲁棒性估计：使用连通分量和度分布
        # R ≈ (最大连通分量大小 / N) * (1 - 度方差归一化)
        components = list(nx.connected_components(G))
        if components:
            max_comp_size = len(max(components, key=len))
            comp_ratio = max_comp_size / G.number_of_nodes()
        else:
            comp_ratio = 0
        
        # 度方差归一化
        if len(degrees_array) > 0:
            degree_var = np.var(degrees_array)
            max_degree = np.max(degrees_array)
            var_ratio = 1 - min(1, degree_var / (max_degree ** 2 + 1))
        else:
            var_ratio = 0
        
        robustness_score = comp_ratio * 0.7 + var_ratio * 0.3
        
        # Combined Reward
        total_score = self.robustness_weight * robustness_score + self.bimodal_weight * bimodality_score
        
        # 存入缓存
        if use_cache and len(self._fitness_cache) < 5000:
            self._fitness_cache[graph_hash] = (total_score, robustness_score, bimodality_score)
        
        return total_score, robustness_score, bimodality_score

    def optimize_topology(self, initial_graph, positions=None, area_size=1000, 
                         comm_distance=None, max_time=None):
        """
        优化版本：减少 deepcopy，使用缓存，早停机制。
        """
        from tqdm import tqdm
        
        start_time = time.time()
        
        if self.verbose:
            logger.info("=== Starting Bimodal RL Topology Optimization (GPU Accelerated) ===")
        
        try:
            topology = NetworkTopology(
                graph=initial_graph,
                positions=positions,
                area_size=area_size,
                comm_distance=comm_distance
            )
            
            topology._compute_distance_matrix()
            
            # Initial evaluation
            initial_score, initial_r, initial_b = self._calculate_combined_fitness(topology, use_cache=False)
            
            state_dim = topology.num_nodes * (topology.num_nodes - 1) // 2
            action_dim = 2
            
            # 创建 SAC 代理（使用 GPU）
            master_agent = SACAgent(state_dim, action_dim)
            population = [SACAgent(state_dim, action_dim) for _ in range(self.population_size)]
            
            best_fitness = initial_score
            best_topology = copy.deepcopy(topology)
            
            # 早停参数
            no_improvement_count = 0
            early_stop_patience = 15
            
            pbar = tqdm(range(self.generations), desc="BimodalRL") if self.verbose else range(self.generations)
            
            for generation in pbar:
                # 动态权重更新
                if self.dynamic_weight_decay:
                    progress = generation / self.generations
                    self.bimodal_weight = max(0.1, self.initial_bimodal_weight * (1.0 - 0.8 * progress))
                
                if max_time and (time.time() - start_time) > max_time:
                    logger.info(f"Time limit reached at generation {generation}")
                    break
                
                try:
                    all_experiences = []
                    fitness_scores = []
                    
                    # Master agent interaction
                    master_experiences, master_fitness, master_topology, master_swaps = \
                        self._topology_interaction(master_agent, topology, steps=12)
                    all_experiences.extend(master_experiences)
                    
                    # Population interaction (减少步数)
                    for i, agent in enumerate(population):
                        experiences, fitness, agent_topology, swaps = \
                            self._topology_interaction(agent, topology, steps=8)
                        all_experiences.extend(experiences)
                        fitness_scores.append((i, fitness, agent_topology))
                        
                        # 只保留最好的经验
                        for exp in experiences[-3:]:
                            agent.replay_buffer.push(*exp)
                    
                    for exp in all_experiences[-10:]:
                        master_agent.replay_buffer.push(*exp)
                    
                    # Evolution (减少频率)
                    if generation % 15 == 0 and len(fitness_scores) >= 2:
                        fitness_scores.sort(key=lambda x: x[1], reverse=True)
                        
                        if random.random() < self.crossover_prob:
                            parent1_idx, parent2_idx = fitness_scores[0][0], fitness_scores[1][0]
                            offspring = self._distillation_crossover(population[parent1_idx], population[parent2_idx])
                            worst_idx = fitness_scores[-1][0]
                            population[worst_idx] = offspring
                    
                    # RL Update (减少频率)
                    if generation % 3 == 0 and master_agent.replay_buffer.size() > 16:
                        master_agent.update(batch_size=16)
                    
                    if generation % 8 == 0:
                        for agent in population:
                            if agent.replay_buffer.size() > 8:
                                agent.update(batch_size=8)
                    
                    # Track Best Solution
                    current_best_fitness = -float('inf')
                    current_best_topology = None
                    
                    for _, fitness, topology_candidate in fitness_scores:
                        if fitness > current_best_fitness:
                            current_best_fitness = fitness
                            current_best_topology = topology_candidate
                    
                    if current_best_topology:
                        try:
                            current_score, _, _ = self._calculate_combined_fitness(current_best_topology, use_cache=True)
                            
                            if current_score > best_fitness:
                                best_fitness = current_score
                                best_topology = copy.deepcopy(current_best_topology)
                                topology = current_best_topology  # 不用 deepcopy
                                no_improvement_count = 0
                            else:
                                no_improvement_count += 1
                        except Exception as e:
                            no_improvement_count += 1
                    else:
                        no_improvement_count += 1
                    
                    self.robustness_history.append(best_fitness)
                    
                    if self.verbose and hasattr(pbar, 'set_postfix'):
                        pbar.set_postfix({
                            'Score': f'{best_fitness:.3f}',
                            'NoImp': no_improvement_count
                        })
                    
                    # 早停检查
                    if no_improvement_count >= early_stop_patience:
                        logger.info(f"Early stopping at generation {generation}")
                        break
                
                except Exception as e:
                    logger.error(f"Generation {generation} error: {e}")
                    continue
            
            # 检查优化后的图是否与原始图相同
            if best_topology.graph.number_of_nodes() == initial_graph.number_of_nodes() and \
               best_topology.graph.number_of_edges() == initial_graph.number_of_edges():
                edges_initial = set(initial_graph.edges())
                edges_best = set(best_topology.graph.edges())
                if edges_initial == edges_best:
                    logger.warning("WARNING: BimodalRL optimization returned the same graph as input! No optimization occurred.")
                else:
                    logger.info(f"BimodalRL: Graph structure changed. Initial edges: {len(edges_initial)}, Optimized edges: {len(edges_best)}, Common: {len(edges_initial & edges_best)}")
            
            return best_topology.graph
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            logger.warning("WARNING: BimodalRL optimization failed, returning original graph!")
            return initial_graph

    def _topology_interaction(self, agent, topology, steps=15):
        """
        优化版本：减少 deepcopy 次数，使用缓存。
        """
        # 只在开始时做一次深拷贝
        current_topology = copy.deepcopy(topology)
        fitness = 0
        experiences = []
        successful_swaps = 0
        
        try:
            current_score, _, _ = self._calculate_combined_fitness(current_topology, use_cache=True)
        except:
            current_score = 0
        
        # 预计算状态向量（避免重复计算）
        current_state = current_topology.get_state_vector()
        
        for step in range(steps):
            try:
                action = agent.select_action(current_state)
                
                edge1, edge2 = self._map_action_to_edges(action, current_topology)
                if edge1 is None or edge2 is None:
                    continue
                
                # 尝试边交换（不用 deepcopy，edge_swap 内部处理回滚）
                edges_backup = list(current_topology.graph.edges())
                swap_success = current_topology.edge_swap(edge1, edge2)
                
                if not swap_success:
                    experiences.append((current_state, action, -0.1, current_state))
                    fitness -= 0.1
                    continue
                
                next_state = current_topology.get_state_vector()
                
                # 计算新的适应度（使用缓存）
                next_score, r_score, b_score = self._calculate_combined_fitness(current_topology, use_cache=True)
                
                reward = (next_score - current_score) * 50  # 减小缩放因子
                
                if b_score > 0.555:
                    reward += 0.5
                
                # 决策：接受或回滚
                if next_score > current_score:
                    accept = True
                elif next_score == current_score:
                    accept = (random.random() < 0.3)
                else:
                    accept = (random.random() < 0.1)
                
                if accept:
                    experiences.append((current_state, action, reward, next_state))
                    current_state = next_state
                    current_score = next_score
                    successful_swaps += 1
                else:
                    # 回滚
                    current_topology.graph.clear_edges()
                    current_topology.graph.add_edges_from(edges_backup)
                    experiences.append((current_state, action, -0.01, current_state))
                
                fitness += reward
                
            except Exception as e:
                continue
        
        return experiences, fitness, current_topology, successful_swaps

def construct_bimodal_rl(G, generations=35):
    """
    Construct a network using Bimodal RL Optimization.
    GPU 加速优化版本。
    
    Args:
        G: 输入图
        generations: 优化代数 (默认35，配合早停)
    """
    start_time = time.time()
    logger.info("  Starting Bimodal RL Structure Optimization (GPU Accelerated)...")
    
    optimizer = BimodalRLOptimizer(
        population_size=2,  # 进一步减少种群大小
        generations=generations,
        verbose=True,
        bimodal_weight=1.0,
        robustness_weight=2.0,
        dynamic_weight_decay=True
    )
    
    G_bimodal = optimizer.optimize_topology(
        initial_graph=G,
        area_size=500,
        comm_distance=200
    )
    
    elapsed = time.time() - start_time
    
    # 打印缓存统计
    if hasattr(optimizer, '_cache_hits'):
        total = optimizer._cache_hits + optimizer._cache_misses
        if total > 0:
            logger.info(f"  Fitness Cache: {optimizer._cache_hits}/{total} hits ({100*optimizer._cache_hits/total:.1f}%)")
    
    logger.info(f"  BimodalRL optimization completed in {elapsed:.1f}s")
    
    return G_bimodal
