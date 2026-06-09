# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import random
import copy
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
from src.utils.logger import get_logger

logger = get_logger(__name__)

class NetworkTopology:
    """网络拓扑类"""
    
    def __init__(self, num_nodes=None, density=None, area_size=None, comm_distance=None, 
                 graph=None, positions=None):
        if graph is not None:
            self.graph = graph.copy()
            self.num_nodes = graph.number_of_nodes()
            mapping = {n: i for i, n in enumerate(self.graph.nodes())}
            self.graph = nx.relabel_nodes(self.graph, mapping)
            
            self.positions = positions if positions else self._generate_positions_from_graph()
            self.density = max(1, int(np.mean([graph.degree(n) for n in graph.nodes()])))
            self.area_size = area_size if area_size else 1000
            self.comm_distance = comm_distance if comm_distance else self._estimate_comm_distance()
        else:
            self.num_nodes = num_nodes
            self.density = density
            self.area_size = area_size
            self.comm_distance = comm_distance
            self.positions = {}
            self.graph = None
            
        self._valid_connections_cache = None
        self._distance_matrix = None
        
    def _generate_positions_from_graph(self):
        """从图结构生成节点位置"""
        positions = {}
        pos = nx.spring_layout(self.graph, scale=500)
        for node, (x, y) in pos.items():
            positions[node] = (x + 500, y + 500)
        return positions
    
    def _estimate_comm_distance(self):
        """估算通信距离"""
        if not self.positions or len(self.positions) < 2:
            return 200
        
        distances = []
        for edge in self.graph.edges():
            node1, node2 = edge
            if node1 in self.positions and node2 in self.positions:
                pos1 = self.positions[node1]
                pos2 = self.positions[node2]
                dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                distances.append(dist)
        
        if distances:
            return max(distances) * 1.2
        return 200
    
    def _compute_distance_matrix(self):
        """预计算距离矩阵"""
        self._distance_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                pos1 = self.positions[i]
                pos2 = self.positions[j]
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                self._distance_matrix[i, j] = distance
                self._distance_matrix[j, i] = distance
    
    def _is_within_distance(self, node1, node2):
        """检查两个节点是否在通信距离内"""
        return self._distance_matrix[node1, node2] <= self.comm_distance
     
   
    def _rollback_graph(self, edges_backup):
        """回滚图到备份状态"""
        self.graph.clear_edges()
        self.graph.add_edges_from(edges_backup)
    
   
    def calculate_robustness_RG(self, sample_ratio=0.3):
        """计算鲁棒性指标RG（采样优化版本）"""
        graph_copy = self.graph.copy()
        total_robustness = 0
        steps = 0
        max_steps = max(5, int(self.num_nodes * sample_ratio))
        
        while graph_copy.number_of_nodes() > 0 and steps < max_steps:
            if graph_copy.number_of_edges() > 0:
                components = list(nx.connected_components(graph_copy))
                max_component_size = max(len(c) for c in components) if components else 0
            else:
                max_component_size = 1 if graph_copy.number_of_nodes() == 1 else 0
            
            total_robustness += max_component_size / self.num_nodes
            steps += 1
            
            if graph_copy.number_of_nodes() > 0:
                degrees = dict(graph_copy.degree())
                if degrees:
                    highest_degree_node = max(degrees, key=degrees.get)
                    graph_copy.remove_node(highest_degree_node)
                else:
                    break
        
        return total_robustness / (steps + 1) if steps > 0 else 0
    
    def get_state_vector(self):
        """获取状态向量（图的邻接矩阵上三角部分）"""
        if self.graph is None:
            return np.zeros(self.num_nodes * (self.num_nodes - 1) // 2)
        
        adj_matrix = nx.adjacency_matrix(self.graph, nodelist=range(self.num_nodes)).toarray()
        # 提取上三角部分（不包括对角线）
        state_vector = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                state_vector.append(adj_matrix[i, j])
        
        return np.array(state_vector, dtype=np.float32)
    
    def get_valid_connections(self):
        """获取所有有效的连接对（可以添加的边）"""
        if self._valid_connections_cache is not None:
            return self._valid_connections_cache
        
        if self.graph is None:
            return []
        
        valid_connections = []
        existing_edges = set(self.graph.edges())
        
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                # 检查是否已存在边
                if (i, j) not in existing_edges and (j, i) not in existing_edges:
                    # 检查是否在通信距离内
                    if self._distance_matrix is not None and self._distance_matrix[i, j] <= self.comm_distance:
                        valid_connections.append((i, j))
        
        self._valid_connections_cache = valid_connections
        return valid_connections
    
    def edge_swap(self, edge_to_remove, edge_to_add):
        """
        执行边交换操作：删除edge_to_remove，添加edge_to_add
        
        Args:
            edge_to_remove: 要删除的边 (node1, node2)
            edge_to_add: 要添加的边 (node3, node4)
        
        Returns:
            bool: 交换是否成功
        """
        if self.graph is None:
            return False
        
        try:
            # 确保要删除的边存在
            if edge_to_remove[0] not in self.graph or edge_to_remove[1] not in self.graph:
                return False
            
            if not self.graph.has_edge(edge_to_remove[0], edge_to_remove[1]):
                return False
            
            # 确保要添加的边不存在
            if self.graph.has_edge(edge_to_add[0], edge_to_add[1]):
                return False
            
            # 确保要添加的边在通信距离内
            if self._distance_matrix is not None:
                if self._distance_matrix[edge_to_add[0], edge_to_add[1]] > self.comm_distance:
                    return False
            
            # 执行交换：删除旧边，添加新边
            self.graph.remove_edge(edge_to_remove[0], edge_to_remove[1])
            self.graph.add_edge(edge_to_add[0], edge_to_add[1])
            
            # 清除缓存
            self._valid_connections_cache = None
            
            return True
        except Exception as e:
            logger.warning(f"Error in edge_swap: {e}")
            return False

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        if batch_size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states))
    
    def size(self):
        return len(self.buffer)

class ActorNetwork(nn.Module):
    """演员网络（策略网络）"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class CriticNetwork(nn.Module):
    """评价网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ValueNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class SACAgent:
    """SAC智能体"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic1 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic2 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)
        self.target_value = ValueNetwork(state_dim).to(self.device)
        
        self.hard_update(self.target_value, self.value)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        self.alpha = alpha
        self.replay_buffer = ReplayBuffer()
        
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
    
    def select_action(self, state, deterministic=False):
        """
        选择动作
        
        Args:
            state: 状态向量（numpy array）
            deterministic: 是否使用确定性策略（用于测试）
        
        Returns:
            numpy array: 动作向量
        """
        if isinstance(state, np.ndarray):
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor.forward(state)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state)
        
        return action.cpu().numpy().flatten()
          
    def update(self, batch_size=64, gamma=0.99, tau=0.005):
        if self.replay_buffer.size() < batch_size:
            return {"actor_loss": 0, "critic_loss": 0, "value_loss": 0}
        
        try:
            states, actions, rewards, next_states = self.replay_buffer.sample(batch_size)
            
            if len(states) == 0:
                return {"actor_loss": 0, "critic_loss": 0, "value_loss": 0}
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            
            if states.shape[0] == 0:
                return {"actor_loss": 0, "critic_loss": 0, "value_loss": 0}
            
            with torch.no_grad():
                next_state_actions, next_state_log_pi = self.actor.sample(next_states)
                target_q1 = self.critic1(next_states, next_state_actions)
                target_q2 = self.critic2(next_states, next_state_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_state_log_pi
                target_value = rewards + gamma * target_q
            
            current_value = self.value(states)
            value_loss = F.mse_loss(current_value, target_value)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)
            self.value_optimizer.step()
            
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)
            critic1_loss = F.mse_loss(current_q1, target_value)
            critic2_loss = F.mse_loss(current_q2, target_value)
            
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
            self.critic1_optimizer.step()
            
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
            self.critic2_optimizer.step()
            
            state_actions, log_pi = self.actor.sample(states)
            q1_new = self.critic1(states, state_actions)
            q2_new = self.critic2(states, state_actions)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * log_pi - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            
            self.soft_update(self.target_value, self.value, tau)
            
            return {
                "actor_loss": actor_loss.item(),
                "critic_loss": (critic1_loss + critic2_loss).item(),
                "value_loss": value_loss.item()
            }
            
        except Exception:
            return {"actor_loss": 0, "critic_loss": 0, "value_loss": 0}

class ROHEMOptimizer:
    """ROMEM优化器 - 高性能版本"""
    
    def __init__(self, population_size=2, generations=15, verbose=True):
        self.population_size = population_size
        self.generations = generations
        self.verbose = verbose
        self.crossover_prob = 0.8
        self.robustness_history = []
        
    def optimize_topology(self, initial_graph, positions=None, area_size=1000, 
                         comm_distance=None, max_time=None):
        """优化拓扑（带早停机制）"""
        start_time = time.time()
        
        try:
            topology = NetworkTopology(
                graph=initial_graph,
                positions=positions,
                area_size=area_size,
                comm_distance=comm_distance
            )
            
            topology._compute_distance_matrix()
            
            initial_robustness = topology.calculate_robustness_RG()
            
            state_dim = topology.num_nodes * (topology.num_nodes - 1) // 2
            action_dim = 2
            
            master_agent = SACAgent(state_dim, action_dim)
            population = [SACAgent(state_dim, action_dim) for _ in range(self.population_size)]
            
            best_robustness = initial_robustness
            best_topology = copy.deepcopy(topology)
            no_improvement_count = 0
            early_stop_patience = 5
            
            pbar = tqdm(range(self.generations), desc="ROMEN优化", leave=False) if self.verbose else range(self.generations)
            
            for generation in pbar:
                if max_time and (time.time() - start_time) > max_time:
                    break
                
                try:
                    all_experiences = []
                    fitness_scores = []
                    
                    master_experiences, master_fitness, master_topology, _ = \
                        self._topology_interaction(master_agent, topology, steps=10)
                    all_experiences.extend(master_experiences)
                    
                    for i, agent in enumerate(population):
                        experiences, fitness, agent_topology, _ = \
                            self._topology_interaction(agent, topology, steps=6)
                        all_experiences.extend(experiences)
                        fitness_scores.append((i, fitness, agent_topology))
                        
                        for exp in experiences[-3:]:
                            agent.replay_buffer.push(*exp)
                    
                    for exp in all_experiences[-10:]:
                        master_agent.replay_buffer.push(*exp)
                    
                    if generation % 5 == 0 and len(fitness_scores) >= 2:
                        fitness_scores.sort(key=lambda x: x[1], reverse=True)
                        if random.random() < self.crossover_prob:
                            parent1_idx, parent2_idx = fitness_scores[0][0], fitness_scores[1][0]
                            offspring = self._distillation_crossover(population[parent1_idx], population[parent2_idx])
                            worst_idx = fitness_scores[-1][0]
                            population[worst_idx] = offspring
                    
                    if generation % 2 == 0 and master_agent.replay_buffer.size() > 16:
                        master_agent.update(batch_size=16)
                    
                    if generation % 5 == 0:
                        for agent in population:
                            if agent.replay_buffer.size() > 8:
                                agent.update(batch_size=8)
                    
                    current_best_fitness = -float('inf')
                    current_best_topology = None
                    
                    for _, fitness, topology_candidate in fitness_scores:
                        if fitness > current_best_fitness:
                            current_best_fitness = fitness
                            current_best_topology = topology_candidate
                    
                    improved = False
                    if current_best_topology:
                        try:
                            current_robustness = current_best_topology.calculate_robustness_RG()
                            if current_robustness > best_robustness:
                                best_robustness = current_robustness
                                best_topology = copy.deepcopy(current_best_topology)
                                topology = current_best_topology
                                improved = True
                        except Exception:
                            pass
                    
                    if improved:
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    self.robustness_history.append(best_robustness)
                    
                    if self.verbose and hasattr(pbar, 'set_postfix'):
                        pbar.set_postfix({'适应度': f'{best_robustness:.4f}'})
                    
                    if no_improvement_count >= early_stop_patience:
                        break
                
                except Exception:
                    continue
            
            return best_topology.graph
            
        except Exception:
            return initial_graph
    
    def _topology_interaction(self, agent, topology, steps=20):
        current_topology = copy.deepcopy(topology)
        fitness = 0
        experiences = []
        successful_swaps = 0
        
        try:
            initial_robustness = current_topology.calculate_robustness_RG()
        except:
            initial_robustness = 0.1
        
        for step in range(steps):
            try:
                current_state = current_topology.get_state_vector()
                action = agent.select_action(current_state)
                
                edge_to_remove, edge_to_add = self._map_action_to_edges(action, current_topology)
                if edge_to_remove is None or edge_to_add is None:
                    continue
                
                topology_copy = copy.deepcopy(current_topology)
                swap_success = topology_copy.edge_swap(edge_to_remove, edge_to_add)
                
                if not swap_success:
                    experiences.append((current_state, action, -1, current_state))
                    fitness -= 1
                    continue
                
                next_state = topology_copy.get_state_vector()
                next_robustness = topology_copy.calculate_robustness_RG()
                current_robustness = current_topology.calculate_robustness_RG()
                
                if next_robustness > current_robustness:
                    reward = 10
                    accept = True
                elif next_robustness == current_robustness:
                    reward = 0
                    accept = (random.random() < 0.3)
                else:
                    reward = -1
                    accept = (random.random() < 0.1)
                
                if swap_success and accept:
                    experiences.append((current_state, action, reward, next_state))
                    current_topology = topology_copy
                    successful_swaps += 1
                else:
                    experiences.append((current_state, action, -1, current_state))
                
                fitness += reward
                
            except Exception as e:
                continue
        
        return experiences, fitness, current_topology, successful_swaps
    
    def _map_action_to_edges(self, action, topology):
        """
        将动作映射到两条边：一条要删除的边（现有边），一条要添加的边（有效连接）
        
        Args:
            action: 动作向量 [a1, a2]
            topology: NetworkTopology实例
        
        Returns:
            tuple: (edge_to_remove, edge_to_add) 或 (None, None) 如果失败
        """
        try:
            # 获取现有边和有效连接
            existing_edges = list(topology.graph.edges())
            valid_connections = topology.get_valid_connections()
            
            if not existing_edges or not valid_connections:
                return None, None
            
            # 归一化动作到[0, 1]
            action_normalized = (np.array(action) + 1) / 2
            action_normalized = np.clip(action_normalized, 0, 1)
            
            # 选择要删除的边（从现有边中选择）
            edge_idx = int(action_normalized[0] * len(existing_edges))
            edge_idx = np.clip(edge_idx, 0, len(existing_edges) - 1)
            edge_to_remove = existing_edges[edge_idx]
            
            # 选择要添加的边（从有效连接中选择）
            conn_idx = int(action_normalized[1] * len(valid_connections))
            conn_idx = np.clip(conn_idx, 0, len(valid_connections) - 1)
            edge_to_add = valid_connections[conn_idx]
            
            return edge_to_remove, edge_to_add
        except Exception as e:
            logger.warning(f"Error in _map_action_to_edges: {e}")
            return None, None
    
    def _distillation_crossover(self, parent1, parent2):
        try:
            state_dim = parent1.actor.fc1.in_features
            action_dim = parent1.actor.mean.out_features
            offspring = SACAgent(state_dim, action_dim)
            
            offspring.actor.load_state_dict(copy.deepcopy(parent1.actor.state_dict()))
            return offspring
        except:
            return SACAgent(parent1.actor.fc1.in_features, parent1.actor.mean.out_features)
    
    def _gaussian_mutation(self, agent):
        try:
            state_dim = agent.actor.fc1.in_features
            action_dim = agent.actor.mean.out_features
            mutated_agent = SACAgent(state_dim, action_dim)
            mutated_agent.actor.load_state_dict(copy.deepcopy(agent.actor.state_dict()))
            
            with torch.no_grad():
                for param in mutated_agent.actor.parameters():
                    if len(param.shape) > 1:
                        mask = torch.rand_like(param) < 0.03
                        noise = torch.randn_like(param) * 0.05
                        param.data = param.data + noise * mask.float()
            
            return mutated_agent
        except:
            return agent