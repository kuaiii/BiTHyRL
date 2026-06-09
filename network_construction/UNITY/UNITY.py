import numpy as np
import networkx as nx
import random
import math
from typing import Tuple, List, Dict, get_origin
import time
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'

class UNITY:
    def __init__(self, area_diameter: float = 500, comm_range: float = 200, epsilon: float = 1e-9):
        """
        初始化UNITY算法
        """
        self.area_diameter = area_diameter
        self.comm_range = comm_range
        self.epsilon = epsilon
        
    def generate_robust_topology(self, N: int, M: int) -> nx.Graph:
        """
        生成鲁棒网络拓扑 - 这是UNITY的核心功能
        
        参数:
            N: 节点数量
            M: 边密度（每个节点平均连接数）
            
        返回:
            G: 生成的鲁棒拓扑
        """
        # 初始化网络
        G = self._initialize_network(N)
        
        # 执行度平衡连接过程M次
        for m in range(M):
            for i in range(N):
                # 找出节点i的潜在邻居
                neighbor_list, degree_list = self._get_potential_neighbors(G, i)
                
                if not neighbor_list:
                    continue
                    
                # 应用映射操作符
                probability_list = self._mapping_operator(degree_list)
                
                # 轮盘赌选择
                selected_idx = self._roulette_selection(probability_list)
                if selected_idx is not None:
                    k = neighbor_list[selected_idx]
                    # 添加连接（避免重复）
                    if not G.has_edge(i, k):
                        G.add_edge(i, k)
        
        return G
    
    def optimize_topology_structure(self, G: nx.Graph) -> nx.Graph:
        """
        基于现有网络的结构参数生成新的鲁棒拓扑
        
        参数:
            G: 输入的NetworkX图（用于获取参数）
            
        返回:
            optim_G: 生成的鲁棒拓扑
        """
        # 从输入图中获取基本参数
        N = G.number_of_nodes()
        # 修正边密度计算：总边数除以节点数，但要考虑无向图特性
        M = max(1, int(2 * G.number_of_edges() / N))  # 确保至少为1
        
        print(f"参数: N={N}, M={M}, 原始边数={G.number_of_edges()}")
        
        # 生成新的鲁棒拓扑
        optim_G = self.generate_robust_topology(N, M)
        
        return optim_G
    
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
    
    def _get_potential_neighbors(self, G: nx.Graph, node_idx: int) -> Tuple[List[int], List[int]]:
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
    
    def _mapping_operator(self, degree_list: List[int]) -> List[float]:
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
    
    def _roulette_selection(self, probability_list: List[float]) -> int:
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
        
        参数:
            input_graph: 输入的初始网络
            
        返回:
            matched_graph: 节点数和边数与输入网络一致的鲁棒拓扑
        """
        # 获取输入网络的核心参数
        N = input_graph.number_of_nodes()  # 节点数严格匹配
        target_edges = input_graph.number_of_edges()  # 目标边数严格匹配
        
        # 初始化网络（复制输入网络的节点位置以保持部署一致性，增强对比公平性）
        matched_graph = nx.Graph()
        for node in input_graph.nodes():
            # 若输入网络有位置信息则复用，否则随机生成（与论文中部署场景一致）
            if 'pos' in input_graph.nodes[node]:
                matched_graph.add_node(node, pos=input_graph.nodes[node]['pos'])
            else:
                # 按论文设定的圆形区域生成位置
                r = self.area_diameter/2 * math.sqrt(random.random())
                theta = random.random() * 2 * math.pi
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                matched_graph.add_node(node, pos=(x, y))
        
        # 执行度平衡连接，直到达到目标边数
        edges_added = 0
        max_iter = target_edges * 10  # 防止通信范围过小时死循环
        iter_count = 0
        while edges_added < target_edges and iter_count < max_iter:
            iter_count += 1
            made_progress = False
            # 遍历所有节点，依次尝试添加边
            for i in range(N):
                if edges_added >= target_edges:
                    break
                # 获取节点i的潜在邻居（未连接且在通信范围内）
                neighbor_list, degree_list = self._get_potential_neighbors(matched_graph, i)
                if not neighbor_list:
                    continue
                # 应用映射操作符计算连接概率
                probability_list = self._mapping_operator(degree_list)
                # 轮盘赌选择目标节点
                selected_idx = self._roulette_selection(probability_list)
                if selected_idx is not None:
                    k = neighbor_list[selected_idx]
                    if not matched_graph.has_edge(i, k):
                        matched_graph.add_edge(i, k)
                        edges_added += 1
                        made_progress = True
            # 若本轮未添加任何边，提前退出（通信范围不足）
            if not made_progress:
                break
        return matched_graph


def generate_robust_network(N: int, M: int, area_diameter: float = 500, comm_range: float = 200) -> nx.Graph:
    """
    生成鲁棒网络的便捷接口函数
    
    参数:
        N: 节点数量
        M: 边密度（每个节点平均连接数）
        area_diameter: 部署区域的直径
        comm_range: 节点通信范围
        
    返回:
        G: 生成的鲁棒拓扑
    """
    unity = UNITY(area_diameter=area_diameter, comm_range=comm_range)
    return unity.generate_robust_topology(N, M)


def optimize_network_structure(G: nx.Graph, area_diameter: float = 500, comm_range: float = 200) -> nx.Graph:
    """
    基于现有网络结构生成鲁棒拓扑的便捷接口函数
    """
    unity = UNITY(area_diameter=area_diameter, comm_range=comm_range)
    return unity.optimize_topology_structure(G)


# 保持其他函数不变...
def calculate_robustness(G: nx.Graph, attack_type: str = "malicious") -> float:
    """计算网络对攻击的鲁棒性"""
    H = G.copy()
    N = H.number_of_nodes()
    
    if N == 0:
        return 0.0
    
    mcs_sizes = []
    
    # 初始状态
    if H.number_of_nodes() > 0:
        mcs_sizes.append(len(max(nx.connected_components(H), key=len)))
    else:
        mcs_sizes.append(0)
    
    # 攻击过程
    for i in range(N):
        if H.number_of_nodes() == 0:
            mcs_sizes.append(0)
            continue
            
        if attack_type == "malicious":
            # 恶意攻击: 移除度数最高的节点
            degrees = dict(H.degree())
            if not degrees:
                mcs_sizes.append(0)
                continue
            target = max(degrees, key=degrees.get)
        else:
            # 随机攻击: 随机移除一个节点
            nodes = list(H.nodes())
            if not nodes:
                mcs_sizes.append(0)
                continue
            target = random.choice(nodes)
            
        H.remove_node(target)
        
        if H.number_of_nodes() == 0:
            mcs_sizes.append(0)
        else:
            components = list(nx.connected_components(H))
            if components:
                mcs_sizes.append(len(max(components, key=len)))
            else:
                mcs_sizes.append(0)
    
    # 计算R值
    R = (1 / (N + 1)) * sum(s / N for s in mcs_sizes)
    return R


def calculate_degree_variance(G: nx.Graph) -> float:
    """计算网络节点度的方差"""
    if G.number_of_nodes() <= 1:
        return 0.0
        
    degrees = [G.degree(n) for n in G.nodes()]
    mean_degree = sum(degrees) / len(degrees)
    var = sum((d - mean_degree) ** 2 for d in degrees) / (len(degrees) - 1)
    return var

def plot_robustness_comparison(initial_graph, optimized_graph, title="Unity"):
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
    from scipy import stats as scipy_stats
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
        import matplotlib.pyplot as plt
        
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
    
    
    
def main():
    """主函数，用于测试UNITY算法"""
    print("===== UNITY算法测试 =====")
    
    # 设置随机种子以便结果可重现
    random.seed(42)
    np.random.seed(42)
        
    # 参数设置
    N = 100  # 节点数量
    M = 2    # 边密度
    seed = 42
    print(f"创建节点数为{N}，边密度为{M}的BA无标度网络...")
    
    # 创建BA无标度网络
    G_original = nx.barabasi_albert_graph(N, M, seed=seed)
    
    # 为原始网络添加位置信息（模拟实际部署）
    for i in G_original.nodes():
        r = 250 * math.sqrt(random.random())
        theta = random.random() * 2 * math.pi
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        G_original.nodes[i]['pos'] = (x, y)
    
    print("使用UNITY生成鲁棒网络拓扑...")
    start_time = time.time()
    
    # 方法1：基于原网络参数生成新的鲁棒拓扑
    G_robust = optimize_network_structure(G_original)
    
    # 方法2：直接生成指定参数的鲁棒拓扑
    # G_robust = generate_robust_network(N, M)
    
    end_time = time.time()
    
    print(f"生成完成，用时: {end_time - start_time:.4f}秒")
    
    # 网络基本信息
    print(f"\n原始BA网络: 节点数={G_original.number_of_nodes()}, 边数={G_original.number_of_edges()}")
    print(f"鲁棒网络: 节点数={G_robust.number_of_nodes()}, 边数={G_robust.number_of_edges()}")
    
    # 计算鲁棒性指标
    print("\n计算鲁棒性指标...")
    
    # 恶意攻击鲁棒性
    r_original_malicious = calculate_robustness(G_original, "malicious")
    r_robust_malicious = calculate_robustness(G_robust, "malicious")
    
    # 随机攻击鲁棒性
    r_original_random = calculate_robustness(G_original, "random")
    r_robust_random = calculate_robustness(G_robust, "random")
    
    # 计算度方差
    var_original = calculate_degree_variance(G_original)
    var_robust = calculate_degree_variance(G_robust)
    
    # 打印结果
    print("\n===== 性能对比 =====")
    print("度方差（越小表示度分布越均匀）:")
    print(f"  原始网络: {var_original:.4f}")
    print(f"  鲁棒网络: {var_robust:.4f}")
    print(f"  改善: {(var_original - var_robust) / var_original * 100:.2f}%")
    
    print("\n对恶意攻击的鲁棒性 (R值):")
    print(f"  原始网络: {r_original_malicious:.4f}")
    print(f"  鲁棒网络: {r_robust_malicious:.4f}")
    improvement_mal = (r_robust_malicious - r_original_malicious) / r_original_malicious * 100 if r_original_malicious > 0 else 0
    print(f"  提升: {improvement_mal:.2f}%")
    
    print("\n对随机攻击的鲁棒性 (R值):")
    print(f"  原始网络: {r_original_random:.4f}")
    print(f"  鲁棒网络: {r_robust_random:.4f}")
    improvement_rand = (r_robust_random - r_original_random) / r_original_random * 100 if r_original_random > 0 else 0
    print(f"  提升: {improvement_rand:.2f}%")
    
    # 可视化结果
    print("\n生成可视化图表...")
    
    # 度分布对比
    plt.figure(figsize=(15, 5))
    
    # 原始网络度分布
    degrees_original = [G_original.degree(n) for n in G_original.nodes()]
    degrees_robust = [G_robust.degree(n) for n in G_robust.nodes()]
    
    max_degree = max(max(degrees_original), max(degrees_robust))
    bins = range(0, max_degree + 2)
    
    plt.subplot(1, 3, 1)
    plt.hist(degrees_original, bins=bins, alpha=0.7, color='blue', density=True)
    plt.xlabel('节点度数')
    plt.ylabel('概率密度')
    plt.title('原始BA网络度分布')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(degrees_robust, bins=bins, alpha=0.7, color='red', density=True)
    plt.xlabel('节点度数')
    plt.ylabel('概率密度')
    plt.title('UNITY鲁棒网络度分布')
    plt.grid(True, alpha=0.3)
    
    # 鲁棒性对比
    plt.subplot(1, 3, 3)
    categories = ['恶意攻击', '随机攻击']
    original_values = [r_original_malicious, r_original_random]
    robust_values = [r_robust_malicious, r_robust_random]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, original_values, width, label='原始网络', color='blue', alpha=0.7)
    plt.bar(x + width/2, robust_values, width, label='鲁棒网络', color='red', alpha=0.7)
    
    plt.xlabel('攻击类型')
    plt.ylabel('鲁棒性 (R值)')
    plt.title('鲁棒性对比')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return G_original, G_robust

def UNITY_method(G=nx.erdos_renyi_graph(100, 0.06, seed=42)):
    unity = UNITY()
    # G_original = nx.erdos_renyi_graph(100, 0.06, seed=42)
    G_robust = unity.generate_matched_topology(G)
    
    # plot_robustness_comparison(G_original, G_robust)
    # plot_random_robustness_comparison(G_original, G_robust)
    # compare_degree_distributions(G_original, G_robust)
    print("\n=== 优化完成 ===")
    return G_robust


if __name__ == "__main__":
    # G_original, _ = main()
    # 生成与初始网络节点数、边数完全一致的鲁棒拓扑
    unity = UNITY()
    G_original = nx.erdos_renyi_graph(100, 0.06, seed=42)
    G_robust = unity.generate_matched_topology(G_original)
    
    plot_robustness_comparison(G_original, G_robust)
    plot_random_robustness_comparison(G_original, G_robust)
    compare_degree_distributions(G_original, G_robust)
    print("\n=== 优化完成 ===")