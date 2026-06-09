# -*- coding: utf-8 -*-
import networkx as nx
import random
from tqdm import tqdm
from src.utils.logger import get_logger

logger = get_logger(__name__)

class UnionFind:
    """并查集数据结构，支持增量节点添加"""
    
    __slots__ = ['parent', 'rank', 'size', 'max_component_size', 'num_components', 'active']
    
    def __init__(self, n: int, initially_active: set = None):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [0] * n
        self.max_component_size = 0
        self.num_components = 0
        self.active = [False] * n
        
        if initially_active:
            for node in initially_active:
                self.activate_node(node)
    
    def activate_node(self, x: int):
        """激活一个节点"""
        if not self.active[x]:
            self.active[x] = True
            self.size[x] = 1
            self.parent[x] = x
            self.rank[x] = 0
            self.num_components += 1
            if self.max_component_size < 1:
                self.max_component_size = 1
    
    def find(self, x: int) -> int:
        """查找节点x所属连通分量的根节点（带路径压缩）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """合并节点x和y所在的连通分量"""
        if not self.active[x] or not self.active[y]:
            return False
        
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        self.size[root_y] = 0
        
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        if self.size[root_x] > self.max_component_size:
            self.max_component_size = self.size[root_x]
        
        self.num_components -= 1
        return True
    
    def get_component_size(self, x: int) -> int:
        """获取节点 x 所在连通分量的大小"""
        if not self.active[x]:
            return 0
        return self.size[self.find(x)]
    
    def get_max_component_size(self) -> int:
        """获取最大连通分量的大小"""
        return self.max_component_size
    
    def recalculate_max_component_size(self):
        """重新计算最大连通分量大小"""
        self.max_component_size = 0
        for i, active in enumerate(self.active):
            if active and self.parent[i] == i:
                if self.size[i] > self.max_component_size:
                    self.max_component_size = self.size[i]


class OnionOptimizer:
    """ONION结构优化器（使用逆向重组法和Union-Find）"""
    
    def __init__(self, G: nx.Graph):
        self.G = G
        self.N = G.number_of_nodes()
        
        self.node_list = list(G.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.node_list)}
        
        degrees = dict(G.degree())
        self.attack_sequence = sorted(
            range(self.N),
            key=lambda idx: degrees[self.node_list[idx]],
            reverse=True
        )
        
        self._update_edge_index_set(G)
        self._set_adaptive_params()
    
    def _update_edge_index_set(self, G: nx.Graph):
        """更新边的索引集合"""
        self.edge_set = set()
        for u, v in G.edges():
            idx_u = self.node_to_idx[u]
            idx_v = self.node_to_idx[v]
            self.edge_set.add((min(idx_u, idx_v), max(idx_u, idx_v)))
        self.edge_list = list(self.edge_set)
    
    def _set_adaptive_params(self):
        """根据图规模自适应设置参数"""
        N = self.N
        if N >= 500:
            self.sample_ratio = 0.10
            self.max_iterations = 500
            self.early_stop_patience = 80
            self.num_trials = 3
        elif N >= 300:
            self.sample_ratio = 0.15
            self.max_iterations = 600
            self.early_stop_patience = 100
            self.num_trials = 3
        elif N >= 150:
            self.sample_ratio = 0.20
            self.max_iterations = 700
            self.early_stop_patience = 100
            self.num_trials = 3
        else:
            self.sample_ratio = 0.30
            self.max_iterations = 800
            self.early_stop_patience = 100
            self.num_trials = 4
    
    def robustness_measure_reverse_union_find(self, edge_set: set) -> float:
        """使用逆向重组法和Union-Find计算鲁棒性"""
        N = self.N
        if N == 0:
            return 0.0
        
        K = max(3, int(N * self.sample_ratio))
        attack_order = self.attack_sequence[:K]
        attack_set = set(attack_order)
        reverse_order = list(reversed(attack_order))
        surviving_nodes = set(range(N)) - attack_set
        
        adj_list = [[] for _ in range(N)]
        for u, v in edge_set:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        uf = UnionFind(N, initially_active=surviving_nodes)
        for node in surviving_nodes:
            for neighbor in adj_list[node]:
                if neighbor in surviving_nodes:
                    uf.union(node, neighbor)
        
        lcc_sizes = [0] * (K + 1)
        lcc_sizes[0] = uf.get_max_component_size()
        
        for step, node in enumerate(reverse_order, start=1):
            uf.activate_node(node)
            for neighbor in adj_list[node]:
                if uf.active[neighbor]:
                    uf.union(node, neighbor)
            lcc_sizes[step] = uf.get_max_component_size()
        
        sum_lcc = sum(lcc_sizes)
        R = sum_lcc / (N * (K + 1))
        return R
    
    def _try_swap(self, edge_set: set, edge1: tuple, edge2: tuple) -> tuple:
        """尝试一次边交换并计算新的鲁棒性"""
        u, v = edge1
        x, y = edge2
        
        if len({u, v, x, y}) < 4:
            return None, None, None
        
        if random.random() < 0.5:
            new_edge1 = (min(u, x), max(u, x))
            new_edge2 = (min(v, y), max(v, y))
        else:
            new_edge1 = (min(u, y), max(u, y))
            new_edge2 = (min(v, x), max(v, x))
        
        if new_edge1 in edge_set or new_edge2 in edge_set:
            return None, None, None
        
        new_edge_set = edge_set.copy()
        new_edge_set.discard(edge1)
        new_edge_set.discard(edge2)
        new_edge_set.add(new_edge1)
        new_edge_set.add(new_edge2)
        
        R_new = self.robustness_measure_reverse_union_find(new_edge_set)
        return R_new, new_edge_set, (edge1, edge2, new_edge1, new_edge2)
    
    def optimize_network(self, max_iterations: int = None, threshold: float = 0.0, 
                        verbose: bool = True) -> nx.Graph:
        """通过边交换优化网络的鲁棒性（优化版本：减少迭代次数，添加进度显示）"""
        import time
        start_time = time.time()
        
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        # 优化：减少迭代次数（如果默认值太大）
        max_iterations = min(max_iterations, 200)
        
        current_edge_set = self.edge_set.copy()
        current_edge_list = list(current_edge_set)
        R_old = self.robustness_measure_reverse_union_find(current_edge_set)
        best_R = R_old
        best_edge_set = current_edge_set.copy()
        
        no_improvement_count = 0
        max_no_improvement = 20  # 早停机制
        
        if verbose:
            print(f"  [ONION] 开始优化 ({max_iterations} 次迭代)...")
        
        for iteration in range(max_iterations):
            if len(current_edge_list) < 2:
                break
            
            # 早停机制
            if no_improvement_count >= max_no_improvement:
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"  [ONION] 早停于迭代 {iteration + 1} (无改进 {max_no_improvement} 次) (耗时: {elapsed:.2f}s)")
                break
            
            best_swap_result = None
            best_R_new = R_old
            best_new_edge_set = None
            
            # 优化：减少尝试次数
            num_trials = min(self.num_trials, 10)
            for _ in range(num_trials):
                edge1, edge2 = random.sample(current_edge_list, 2)
                R_new, new_edge_set, swap_info = self._try_swap(current_edge_set, edge1, edge2)
                
                if R_new is not None and R_new > best_R_new:
                    best_R_new = R_new
                    best_new_edge_set = new_edge_set
                    best_swap_result = swap_info
            
            if best_swap_result and best_R_new > R_old + threshold:
                current_edge_set = best_new_edge_set
                current_edge_list = list(current_edge_set)
                R_old = best_R_new
                no_improvement_count = 0
                
                if R_old > best_R:
                    best_R = R_old
                    best_edge_set = current_edge_set.copy()
                    if verbose and (iteration % 50 == 0 or iteration == max_iterations - 1):
                        elapsed = time.time() - start_time
                        print(f"  [ONION] Iter {iteration + 1}/{max_iterations}: New Best R = {best_R:.4f} (耗时: {elapsed:.2f}s)")
            else:
                no_improvement_count += 1
        
        # 构建最终图
        G_optimized = nx.Graph()
        G_optimized.add_nodes_from(self.node_list)
        for idx_u, idx_v in best_edge_set:
            G_optimized.add_edge(self.idx_to_node[idx_u], self.idx_to_node[idx_v])
        
        if verbose:
            total_time = time.time() - start_time
            edge_count = len(best_edge_set)
            print(f"  [ONION] 优化完成 | Best R: {best_R:.4f} | Edges: {edge_count} | 总耗时: {total_time:.2f}s")
        
        return G_optimized