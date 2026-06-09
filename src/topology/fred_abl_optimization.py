import numpy as np
import networkx as nx
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm
import random
import warnings

# 忽略 GPR 的一些数值警告
warnings.filterwarnings("ignore")


class IIoTEnv:
    """
    IIoT 环境模拟 (修改版)
    基于预定义的networkx图G进行重构，验证边是否在通信半径内
    """
    def __init__(self, G, comm_range=0.3, seed=42):
        np.random.seed(seed)
        self.r = comm_range
        
        # 从图G获取节点和位置
        self.n = G.number_of_nodes()
        self.positions = np.zeros((self.n, 2))
        
        # 验证节点ID是否为0到n-1的连续整数
        if sorted(G.nodes()) != list(range(self.n)):
            raise ValueError("Graph nodes must be labeled 0 to n-1 consecutively")
        
        # 提取节点位置 (要求节点属性包含'pos')
        for i in range(self.n):
            if 'pos' not in G.nodes[i]:
                raise ValueError(f"Node {i} missing 'pos' attribute")
            self.positions[i] = np.array(G.nodes[i]['pos'])
        
        # 计算所有节点对的距离
        self.dist_matrix = np.zeros((self.n, self.n))
        self.possible_edges = []  # 存储物理上允许连接的边索引 (i, j)
        
        # 遍历图中所有边，验证是否在通信半径内
        for u, v in G.edges():
            if u == v:  # 跳过自环
                continue
            # 确保边方向一致 (小ID->大ID)
            a, b = sorted((u, v))
            dist = np.linalg.norm(self.positions[a] - self.positions[b])
            self.dist_matrix[a][b] = self.dist_matrix[b][a] = dist
            
            # 只保留满足通信半径的边
            if dist <= self.r:
                if (a, b) not in self.possible_edges:
                    self.possible_edges.append((a, b))
        
        self.m = len(self.possible_edges)  # 有效边的数量
        print(f"[Environment] Nodes: {self.n}, Valid Edges: {self.m}, Comm Range: {self.r}")

    def vector_to_adj(self, binary_vector):
        """将二进制向量还原为邻接矩阵"""
        adj = np.zeros((self.n, self.n), dtype=int)
        for idx, val in enumerate(binary_vector):
            if val == 1:
                u, v = self.possible_edges[idx]
                adj[u][v] = adj[v][u] = 1
        return adj


class ACC_Codec:
    """
    辅助连续编码 (Auxiliary Continuous Coding)
    核心逻辑：利用辅助变量 z 使得海明距离接近的拓扑在连续空间中数值也接近。
    """
    def __init__(self, dim, p1=0.625):
        self.dim = dim
        self.p1 = p1  # 1 的概率 (论文设定 0.625)
        self.p0 = 1.0 - p1

    def decode(self, value):
        """
        连续值 Y (0~1) -> 离散二进制向量
        """
        value = np.clip(value, 0.0, 0.9999999999)
        vector = []
        low = 0.0
        high = 1.0
        z = 0  # 辅助状态初始化

        for _ in range(self.dim):
            span = high - low
            if z == 0:
                mid = low + span * self.p0
                if value < mid:
                    bit = 0
                    high = mid
                else:
                    bit = 1
                    low = mid
            else:
                mid = low + span * self.p1
                if value < mid:
                    bit = 1
                    high = mid
                else:
                    bit = 0
                    low = mid

            vector.append(bit)
            z = bit ^ z  # 更新辅助状态

        return np.array(vector)


class RobustnessEvaluator:
    """
    鲁棒性评估器
    计算论文中定义的指标 R = (1/N) * sum(T_theta)
    """
    @staticmethod
    def calculate_R(adj_matrix, attack_type='malicious', sample_ratio=0.3):
        """
        计算鲁棒性指标 R（优化版本：使用采样加速）
        
        Args:
            adj_matrix: 邻接矩阵
            attack_type: 攻击类型
            sample_ratio: 采样比例（只计算部分节点的攻击，加速计算）
        """
        G = nx.from_numpy_array(adj_matrix)
        n = G.number_of_nodes()
        if n == 0: 
            return 0

        # 使用采样加速：只计算部分节点的攻击
        max_steps = max(5, int(n * sample_ratio))
        
        nodes = list(G.nodes())
        if attack_type == 'malicious':
            nodes_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
            removal_queue = [node for node, degree in nodes_sorted]
        else:
            removal_queue = list(G.nodes())
            random.shuffle(removal_queue)

        sum_tolerance = 0.0
        G_temp = G.copy()

        for theta in range(min(max_steps, n)):
            if G_temp.number_of_nodes() > 0:
                # 使用更快的连通分量计算
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
        return (sum_tolerance / max_steps) * (max_steps / n) if max_steps > 0 else 0


class FRED_ABL:
    """
    主算法类 (适配预定义图)
    """
    def __init__(self, env):
        self.env = env
        self.codec = ACC_Codec(dim=env.m)
        self.evaluator = RobustnessEvaluator()

        # 高斯过程回归模型
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1e2))
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)

        self.X_train = []   # 连续编码值 [0, 1)
        self.y_train = []   # 鲁棒性指标 R
        self.best_R = -1.0
        self.best_topology = None

    def ucb_acquisition(self, num_samples=300, kappa=2.576):
        """
        采集函数 (Upper Confidence Bound) - 优化版本
        减少采样数量以加速计算
        """
        candidates = np.random.uniform(0, 1, (num_samples, 1))
        
        if len(self.X_train) == 0:
            return candidates[0, 0]  # 随机返回一个点

        X = np.array(self.X_train).reshape(-1, 1)
        self.gpr.fit(X, np.array(self.y_train))
        
        mus, stds = self.gpr.predict(candidates, return_std=True)
        ucb_values = mus + kappa * stds
        best_idx = np.argmax(ucb_values)
        return float(candidates[best_idx, 0])

    def run(self, iterations=30, initial_samples=5, verbose=True):
        import time
        from src.utils.logger import log_algorithm_iteration
        
        start_time = time.time()
        if verbose:
            print(f"  [FRED-ABL] 初始化阶段 ({initial_samples} 个样本)...")

        # 1. 初始化阶段
        for i in range(initial_samples):
            x_sample = np.random.uniform(0, 1)
            binary_vec = self.codec.decode(x_sample)
            adj = self.env.vector_to_adj(binary_vec)
            r_val = self.evaluator.calculate_R(adj, sample_ratio=0.3)  # 使用采样加速

            self.X_train.append([float(x_sample)])
            self.y_train.append(r_val)

            if r_val > self.best_R:
                self.best_R = r_val
                self.best_topology = adj.copy()

            if verbose and (i == 0 or i == initial_samples - 1 or r_val > self.best_R * 0.95):
                elapsed = time.time() - start_time
                print(f"  [FRED-ABL] Init {i + 1}/{initial_samples}: R = {r_val:.4f} (耗时: {elapsed:.2f}s)")

        if verbose:
            print(f"  [FRED-ABL] 贝叶斯优化阶段 ({iterations} 次迭代)...")

        # 2. 贝叶斯优化循环（减少打印频率）
        for i in range(iterations):
            x_next = self.ucb_acquisition()
            binary_vec_next = self.codec.decode(x_next)
            adj_next = self.env.vector_to_adj(binary_vec_next)
            r_next = self.evaluator.calculate_R(adj_next, sample_ratio=0.3)  # 使用采样加速

            self.X_train.append([float(x_next)])
            self.y_train.append(r_next)

            if r_next > self.best_R:
                self.best_R = r_next
                self.best_topology = adj_next.copy()
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"  [FRED-ABL] Iter {i + 1}/{iterations}: New Best R = {r_next:.4f} (耗时: {elapsed:.2f}s)")
            elif verbose and (i % 5 == 0 or i == iterations - 1):
                elapsed = time.time() - start_time
                print(f"  [FRED-ABL] Iter {i + 1}/{iterations}: R = {r_next:.4f} (耗时: {elapsed:.2f}s)")

        if verbose:
            total_time = time.time() - start_time
            edge_count = np.sum(self.best_topology) // 2
            print(f"  [FRED-ABL] 优化完成 | Best R: {self.best_R:.4f} | Edges: {edge_count} | 总耗时: {total_time:.2f}s")
        return self.best_topology


# ================= 运行入口 (使用预定义图) =================
if __name__ == "__main__":
    # 创建示例图 (30节点网格)
    G = nx.grid_2d_graph(5, 6)  # 5x6网格
    G = nx.convert_node_labels_to_integers(G)  # 重命名为0-29
    
    # 为节点添加位置属性 (归一化到[0,1]x[0,1])
    pos = {}
    for i, (x, y) in enumerate(G.nodes()):
        # 归一化位置
        norm_x = x / 4.0  # x范围0-4
        norm_y = y / 5.0  # y范围0-5
        G.nodes[i]['pos'] = (norm_x, norm_y)
    
    # 创建环境 (使用预定义图，通信半径0.4)
    env = IIoTEnv(G=G, comm_range=0.4)
    
    # 初始化优化器
    fred = FRED_ABL(env)
    
    # 运行算法
    best_adj = fred.run(iterations=20, initial_samples=5)
    
    # 可视化
    try:
        import matplotlib.pyplot as plt
        
        # 从邻接矩阵创建结果图
        G_final = nx.from_numpy_array(best_adj)
        # 复用原始位置
        pos_final = {i: G.nodes[i]['pos'] for i in range(env.n)}
        
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G_final, pos_final, node_size=120, node_color='skyblue', alpha=0.9)
        nx.draw_networkx_edges(G_final, pos_final, alpha=0.7, edge_color='gray')
        plt.title(f"Optimized Topology (R={fred.best_R:.4f})", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("optimized_topology.png", dpi=300)
        plt.show()
        
        # 同时显示原始图和优化图对比
        plt.figure(figsize=(12, 5))
        
        plt.subplot(121)
        nx.draw_networkx_nodes(G, pos_final, node_size=100, node_color='lightgreen')
        nx.draw_networkx_edges(G, pos_final, alpha=0.5, edge_color='blue')
        plt.title("Original Topology", fontsize=12)
        plt.axis('off')
        
        plt.subplot(122)
        nx.draw_networkx_nodes(G_final, pos_final, node_size=100, node_color='salmon')
        nx.draw_networkx_edges(G_final, pos_final, alpha=0.7, edge_color='red')
        plt.title(f"Optimized Topology (R={fred.best_R:.4f})", fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("topology_comparison.png", dpi=300)
        plt.show()
        
    except ImportError:
        print("Matplotlib not found, skipping visualization.")