import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import matplotlib

# 添加中文字体配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（Windows常用中文字体）
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.use('TkAgg')  # 使用Tkinter作为后端

class OnionNetworkOptimizer:
    def __init__(self, G):
        """
        初始化洋葱网络优化器
        
        Parameters:
        G: networkx.Graph - 输入网络
        """
        self.G = G.copy()
        self.N = len(G.nodes())
        self.original_degree_sequence = [G.degree(node) for node in G.nodes()]
        
    def robustness_measure(self, G, attack_type="HDA"):
        """
        计算网络的鲁棒性度量R
        
        Parameters:
        G: networkx.Graph - 网络
        attack_type: str - 攻击类型 ("HDA" 或 "random")
        
        Returns:
        float: 鲁棒性度量值R
        """
        if len(G.nodes()) == 0:
            return 0.0
        
        G_copy = G.copy()
        N = len(G_copy.nodes())
        
        # 计算在节点移除过程中最大连通集群的平均大小
        s_values = []
        
        # 初始状态
        largest_cc = max(nx.connected_components(G_copy), key=len)
        s_values.append(len(largest_cc) / N)
        
        # 创建攻击序列
        if attack_type == "HDA":
            # 高度自适应攻击（HDA）：每次移除当前度数最高的节点
            nodes_removed = 0
            while nodes_removed < N and len(G_copy.nodes()) > 0:
                # 找出当前度数最高的节点
                degrees = dict(G_copy.degree())
                max_degree = max(degrees.values())
                max_degree_nodes = [n for n, d in degrees.items() if d == max_degree]
                
                # 如果有多个度数相同的最高度节点，随机选择一个
                target_node = random.choice(max_degree_nodes)
                
                # 移除目标节点
                G_copy.remove_node(target_node)
                nodes_removed += 1
                
                # 计算最大连通分量
                if len(G_copy.nodes()) > 0:
                    connected_components = list(nx.connected_components(G_copy))
                    if connected_components:
                        largest_cc = max(connected_components, key=len)
                        s_values.append(len(largest_cc) / N)
                    else:
                        s_values.append(0.0)
                else:
                    s_values.append(0.0)
        else:
            # 随机攻击：随机顺序移除节点
            nodes_list = list(G_copy.nodes())
            random.shuffle(nodes_list)  # 随机打乱节点顺序
            
            for target_node in nodes_list:
                if target_node in G_copy.nodes():
                    G_copy.remove_node(target_node)
                
                # 计算最大连通分量
                if len(G_copy.nodes()) > 0:
                    connected_components = list(nx.connected_components(G_copy))
                    if connected_components:
                        largest_cc = max(connected_components, key=len)
                        s_values.append(len(largest_cc) / N)
                    else:
                        s_values.append(0.0)
                else:
                    s_values.append(0.0)
        
        # 确保s_values的长度为N+1
        while len(s_values) < N + 1:
            s_values.append(0.0)
        
        # 截断多余的值
        s_values = s_values[:N + 1]
        
        # 计算鲁棒性度量
        R = sum(s_values) / (N + 1)
        return R
    
    def optimize_network(self, max_iterations=10000, threshold=0.0, tolerance=0.01, verbose=True):
        """
        通过边交换优化网络的鲁棒性
        
        Parameters:
        max_iterations: int - 最大迭代次数
        threshold: float - 接受交换的阈值
        tolerance: float - 收敛容差
        verbose: bool - 是否显示进度
        
        Returns:
        networkx.Graph - 优化后的网络
        """
        G_optimized = self.G.copy()
        R_old = self.robustness_measure(G_optimized, attack_type="HDA")
        
        if verbose:
            print(f"初始鲁棒性: {R_old:.4f}")
            print(f"开始优化...")
            
        # 跟踪最近的改进
        recent_improvements = []
        last_check_point = 0
        
        for i in tqdm(range(max_iterations), disable=not verbose):
            # 随机选择两条边
            edges = list(G_optimized.edges())
            if len(edges) < 2:
                break
                
            edge1, edge2 = random.sample(edges, 2)
            i, j = edge1
            k, l = edge2
            
            # 避免选择相邻的边和自环
            if len({i, j, k, l}) < 4:
                continue
                
            # 确保新边不会形成重复边
            if G_optimized.has_edge(i, k) or G_optimized.has_edge(j, l):
                continue
            
            # 临时移除旧边
            G_optimized.remove_edge(i, j)
            G_optimized.remove_edge(k, l)
            
            # 添加新边
            G_optimized.add_edge(i, k)
            G_optimized.add_edge(j, l)
            
            # 计算新的鲁棒性
            R_new = self.robustness_measure(G_optimized, attack_type="HDA")
            
            # 判断是否接受交换
            if R_new > R_old + threshold:
                # 接受交换
                R_old = R_new
                recent_improvements.append(R_new - R_old)
            else:
                # 恢复原始边
                G_optimized.remove_edge(i, k)
                G_optimized.remove_edge(j, l)
                G_optimized.add_edge(i, j)
                G_optimized.add_edge(k, l)
            
            # 每1000次迭代检查一次收敛情况
            if i - last_check_point >= 1000:
                last_check_point = i
                recent_improvement = sum(recent_improvements[-1000:] if recent_improvements else [0])
                
                if recent_improvement < tolerance * R_old:
                    if verbose:
                        print(f"\n收敛于迭代 {i}: 最近1000次迭代的总改进 = {recent_improvement:.6f}")
                    break
                    
                if verbose and i % 10000 == 0:
                    print(f"迭代 {i}: 当前鲁棒性 = {R_old:.4f}")
        
        if verbose:
            print(f"最终鲁棒性: {R_old:.4f}")
            
        return G_optimized
    
    def analyze_onion_structure(self, G):
        """
        分析网络的洋葱结构特性
        
        Parameters:
        G: networkx.Graph - 网络
        
        Returns:
        dict - 洋葱结构特性
        """
        # 1. 计算节点的平均邻居度数
        k_nn = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if neighbors:
                k_nn[node] = sum(G.degree(neighbor) for neighbor in neighbors) / len(neighbors)
            else:
                k_nn[node] = 0
        
        # 2. 计算节点的聚类系数
        clustering = nx.clustering(G)
        
        # 3. 计算每个度数k的节点的平均邻居度数
        k_nn_k = {}
        for node, degree in G.degree():
            if degree not in k_nn_k:
                k_nn_k[degree] = []
            k_nn_k[degree].append(k_nn[node])
        
        for k in k_nn_k:
            k_nn_k[k] = sum(k_nn_k[k]) / len(k_nn_k[k])
        
        # 4. 计算每个度数k的节点的平均聚类系数
        clustering_k = {}
        for node, degree in G.degree():
            if degree not in clustering_k:
                clustering_k[degree] = []
            clustering_k[degree].append(clustering[node])
        
        for k in clustering_k:
            if clustering_k[k]:
                clustering_k[k] = sum(clustering_k[k]) / len(clustering_k[k])
            else:
                clustering_k[k] = 0
        
        # 5. 计算洋葱特性：同度数节点连通性
        onion_property = {}
        degrees = dict(G.degree())
        
        for k in set(degrees.values()):
            nodes_with_degree_k = [node for node, deg in degrees.items() if deg == k]
            
            if len(nodes_with_degree_k) <= 1:
                onion_property[k] = 0
                continue
            
            # 创建子图只包含度数<=k的节点
            low_degree_nodes = [node for node, deg in degrees.items() if deg <= k]
            subgraph = G.subgraph(low_degree_nodes)
            
            # 检查度数为k的节点在子图中的连通性
            connected_components = list(nx.connected_components(subgraph))
            
            # 找出包含度数为k的节点的最大连通分量
            max_component_size = 0
            for component in connected_components:
                k_nodes_in_component = [node for node in component if degrees[node] == k]
                if len(k_nodes_in_component) > max_component_size:
                    max_component_size = len(k_nodes_in_component)
            
            # 计算连通比例
            onion_property[k] = max_component_size / len(nodes_with_degree_k)
        
        return {
            'k_nn_k': k_nn_k,
            'clustering_k': clustering_k,
            'onion_property': onion_property
        }

def attack_simulation_comparison(original_G, onion_G, attack_type="HDA", num_simulations=1):
    """
    比较原始网络和洋葱网络在攻击下的表现
    
    Parameters:
    original_G: networkx.Graph - 原始网络
    onion_G: networkx.Graph - 优化后的网络
    attack_type: str - 攻击类型 ("HDA" 或 "random")
    num_simulations: int - 随机攻击的模拟次数
    
    Returns:
    tuple - 原始网络和洋葱网络的攻击曲线
    """
    N = len(original_G.nodes())
    
    if attack_type == "random" and num_simulations > 1:
        # 多次随机攻击
        original_curves = []
        onion_curves = []
        
        for _ in range(num_simulations):
            # 原始网络随机攻击
            G_orig_copy = original_G.copy()
            nodes_list_orig = list(G_orig_copy.nodes())
            random.shuffle(nodes_list_orig)
            
            s_values_orig = []
            largest_cc = max(nx.connected_components(G_orig_copy), key=len)
            s_values_orig.append(len(largest_cc) / N)
            
            for target_node in nodes_list_orig:
                if target_node in G_orig_copy.nodes():
                    G_orig_copy.remove_node(target_node)
                
                if len(G_orig_copy.nodes()) > 0:
                    connected_components = list(nx.connected_components(G_orig_copy))
                    if connected_components:
                        largest_cc = max(connected_components, key=len)
                        s_values_orig.append(len(largest_cc) / N)
                    else:
                        s_values_orig.append(0.0)
                else:
                    s_values_orig.append(0.0)
            
            original_curves.append(s_values_orig)
            
            # 洋葱网络随机攻击
            G_onion_copy = onion_G.copy()
            nodes_list_onion = list(G_onion_copy.nodes())
            random.shuffle(nodes_list_onion)
            
            s_values_onion = []
            largest_cc = max(nx.connected_components(G_onion_copy), key=len)
            s_values_onion.append(len(largest_cc) / N)
            
            for target_node in nodes_list_onion:
                if target_node in G_onion_copy.nodes():
                    G_onion_copy.remove_node(target_node)
                
                if len(G_onion_copy.nodes()) > 0:
                    connected_components = list(nx.connected_components(G_onion_copy))
                    if connected_components:
                        largest_cc = max(connected_components, key=len)
                        s_values_onion.append(len(largest_cc) / N)
                    else:
                        s_values_onion.append(0.0)
                else:
                    s_values_onion.append(0.0)
            
            onion_curves.append(s_values_onion)
        
        # 计算平均曲线和标准差
        original_curves = np.array(original_curves)
        onion_curves = np.array(onion_curves)
        
        original_mean = np.mean(original_curves, axis=0)
        original_std = np.std(original_curves, axis=0)
        onion_mean = np.mean(onion_curves, axis=0)
        onion_std = np.std(onion_curves, axis=0)
        
        return original_mean, original_std, onion_mean, onion_std
    
    else:
        # 单次HDA攻击或单次随机攻击
        # 原始网络攻击
        G_orig_copy = original_G.copy()
        s_values_orig = []
        
        largest_cc = max(nx.connected_components(G_orig_copy), key=len)
        s_values_orig.append(len(largest_cc) / N)
        
        if attack_type == "HDA":
            # 高度自适应攻击
            nodes_removed = 0
            while nodes_removed < N and len(G_orig_copy.nodes()) > 0:
                # 找出当前度数最高的节点
                degrees = dict(G_orig_copy.degree())
                if not degrees:
                    break
                max_degree = max(degrees.values())
                max_degree_nodes = [n for n, d in degrees.items() if d == max_degree]
                
                # 如果有多个度数相同的最高度节点，随机选择一个
                target_node = random.choice(max_degree_nodes)
                
                # 移除目标节点
                G_orig_copy.remove_node(target_node)
                nodes_removed += 1
                
                # 计算最大连通分量
                if len(G_orig_copy.nodes()) > 0:
                    connected_components = list(nx.connected_components(G_orig_copy))
                    if connected_components:
                        largest_cc = max(connected_components, key=len)
                        s_values_orig.append(len(largest_cc) / N)
                    else:
                        s_values_orig.append(0.0)
                else:
                    s_values_orig.append(0.0)
        else:
            # 随机攻击
            nodes_list = list(G_orig_copy.nodes())
            random.shuffle(nodes_list)
            
            for target_node in nodes_list:
                if target_node in G_orig_copy.nodes():
                    G_orig_copy.remove_node(target_node)
                
                if len(G_orig_copy.nodes()) > 0:
                    connected_components = list(nx.connected_components(G_orig_copy))
                    if connected_components:
                        largest_cc = max(connected_components, key=len)
                        s_values_orig.append(len(largest_cc) / N)
                    else:
                        s_values_orig.append(0.0)
                else:
                    s_values_orig.append(0.0)
        
        # 洋葱网络攻击
        G_onion_copy = onion_G.copy()
        s_values_onion = []
        
        largest_cc = max(nx.connected_components(G_onion_copy), key=len)
        s_values_onion.append(len(largest_cc) / N)
        
        if attack_type == "HDA":
            # 高度自适应攻击
            nodes_removed = 0
            while nodes_removed < N and len(G_onion_copy.nodes()) > 0:
                # 找出当前度数最高的节点
                degrees = dict(G_onion_copy.degree())
                if not degrees:
                    break
                max_degree = max(degrees.values())
                max_degree_nodes = [n for n, d in degrees.items() if d == max_degree]
                
                # 如果有多个度数相同的最高度节点，随机选择一个
                target_node = random.choice(max_degree_nodes)
                
                # 移除目标节点
                G_onion_copy.remove_node(target_node)
                nodes_removed += 1
                
                # 计算最大连通分量
                if len(G_onion_copy.nodes()) > 0:
                    connected_components = list(nx.connected_components(G_onion_copy))
                    if connected_components:
                        largest_cc = max(connected_components, key=len)
                        s_values_onion.append(len(largest_cc) / N)
                    else:
                        s_values_onion.append(0.0)
                else:
                    s_values_onion.append(0.0)
        else:
            # 随机攻击
            nodes_list = list(G_onion_copy.nodes())
            random.shuffle(nodes_list)
            
            for target_node in nodes_list:
                if target_node in G_onion_copy.nodes():
                    G_onion_copy.remove_node(target_node)
                
                if len(G_onion_copy.nodes()) > 0:
                    connected_components = list(nx.connected_components(G_onion_copy))
                    if connected_components:
                        largest_cc = max(connected_components, key=len)
                        s_values_onion.append(len(largest_cc) / N)
                    else:
                        s_values_onion.append(0.0)
                else:
                    s_values_onion.append(0.0)
        
        # 确保s_values的长度一致
        max_len = max(len(s_values_orig), len(s_values_onion))
        s_values_orig = s_values_orig + [0.0] * (max_len - len(s_values_orig))
        s_values_onion = s_values_onion + [0.0] * (max_len - len(s_values_onion))
        
        # 截断到N+1
        s_values_orig = s_values_orig[:N+1]
        s_values_onion = s_values_onion[:N+1]
        
        # 计算攻击曲线的标准差（单次攻击时为0）
        original_std = np.zeros_like(s_values_orig)
        onion_std = np.zeros_like(s_values_onion)
        
        return s_values_orig, original_std, s_values_onion, onion_std

def visualize_onion_comparison(original_G, onion_G, original_R, onion_R, attack_type="HDA"):
    """
    可视化原始网络和洋葱结构网络的对比
    
    Parameters:
    original_G: networkx.Graph - 原始网络
    onion_G: networkx.Graph - 优化后的网络
    original_R: float - 原始网络的鲁棒性
    onion_R: float - 优化后网络的鲁棒性
    attack_type: str - 攻击类型
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 创建网格布局
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
    
    # 1. 原始网络
    ax1 = fig.add_subplot(gs[0, 0])
    degrees_orig = dict(original_G.degree())
    pos_orig = nx.spring_layout(original_G, k=0.3, iterations=50, seed=42)
    
    # 节点颜色：按度数着色
    max_degree = max(degrees_orig.values())
    node_colors_orig = [degrees_orig[node] / max_degree for node in original_G.nodes()]
    node_sizes_orig = [degrees_orig[node] * 30 + 50 for node in original_G.nodes()]
    
    nx.draw(original_G, pos_orig, ax=ax1, 
           node_color=node_colors_orig, node_size=node_sizes_orig, 
           cmap=plt.cm.Reds, with_labels=False, 
           edge_color='gray', alpha=0.7, width=0.5)
    attack_name = "高度自适应攻击" if attack_type == "HDA" else "随机攻击"
    ax1.set_title(f'原始网络\n{attack_name}鲁棒性 R = {original_R:.4f}', fontsize=14, fontweight='bold')
    
    # 2. 洋葱结构网络 - 使用径向布局突出核心
    ax2 = fig.add_subplot(gs[0, 1])
    degrees_onion = dict(onion_G.degree())
    
    # 创建径向布局：高度数节点在中心
    def create_radial_layout(G):
        degrees = dict(G.degree())
        pos = {}
        
        # 按度数分组
        degree_groups = {}
        for node, degree in degrees.items():
            if degree not in degree_groups:
                degree_groups[degree] = []
            degree_groups[degree].append(node)
        
        # 按度数排序
        sorted_degrees = sorted(degree_groups.keys(), reverse=True)
        
        # 计算每个度数组的半径
        num_layers = len(sorted_degrees)
        for layer_idx, degree in enumerate(sorted_degrees):
            nodes_in_layer = degree_groups[degree]
            radius = layer_idx * 0.8  # 度数越高，半径越小（越靠近中心）
            
            if len(nodes_in_layer) == 1:
                # 单个节点放在中心或该层的中心
                pos[nodes_in_layer[0]] = (0, 0) if radius == 0 else (radius, 0)
            else:
                # 多个节点均匀分布在圆周上
                angles = np.linspace(0, 2*np.pi, len(nodes_in_layer), endpoint=False)
                for i, node in enumerate(nodes_in_layer):
                    x = radius * np.cos(angles[i])
                    y = radius * np.sin(angles[i])
                    pos[node] = (x, y)
        
        return pos
    
    pos_onion = create_radial_layout(onion_G)
    
    node_colors_onion = [degrees_onion[node] / max_degree for node in onion_G.nodes()]
    node_sizes_onion = [degrees_onion[node] * 30 + 50 for node in onion_G.nodes()]
    
    nx.draw(onion_G, pos_onion, ax=ax2,
           node_color=node_colors_onion, node_size=node_sizes_onion,
           cmap=plt.cm.Reds, with_labels=False,
           edge_color='gray', alpha=0.7, width=0.5)
    ax2.set_title(f'洋葱结构网络\n{attack_name}鲁棒性 R = {onion_R:.4f}', fontsize=14, fontweight='bold')
    
    # 添加度数标签到高度数节点
    high_degree_nodes = [node for node, degree in degrees_onion.items() 
                        if degree >= sorted(degrees_onion.values(), reverse=True)[min(5, len(degrees_onion)-1)]]
    labels = {node: str(degrees_onion[node]) for node in high_degree_nodes}
    nx.draw_networkx_labels(onion_G, pos_onion, labels, ax=ax2, font_size=8, font_color='white', font_weight='bold')
    
    # 3. 网络信息统计
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    # 计算网络统计信息
    avg_clustering_orig = nx.average_clustering(original_G)
    avg_clustering_onion = nx.average_clustering(onion_G)
    
    # 度数分布统计
    degrees_orig_list = list(degrees_orig.values())
    degrees_onion_list = list(degrees_onion.values())
    
    # 核心节点分析（前30%高度数节点）
    top_30_percent = max(1, len(onion_G) // 3)
    core_nodes_orig = sorted(degrees_orig.items(), key=lambda x: x[1], reverse=True)[:top_30_percent]
    core_nodes_onion = sorted(degrees_onion.items(), key=lambda x: x[1], reverse=True)[:top_30_percent]
    
    # 计算核心节点间的连接
    core_nodes_set_orig = set([node for node, _ in core_nodes_orig])
    core_nodes_set_onion = set([node for node, _ in core_nodes_onion])
    
    core_edges_orig = sum(1 for u, v in original_G.edges() if u in core_nodes_set_orig and v in core_nodes_set_orig)
    core_edges_onion = sum(1 for u, v in onion_G.edges() if u in core_nodes_set_onion and v in core_nodes_set_onion)
    
    max_core_edges = top_30_percent * (top_30_percent - 1) // 2
    core_density_orig = core_edges_orig / max_core_edges if max_core_edges > 0 else 0
    core_density_onion = core_edges_onion / max_core_edges if max_core_edges > 0 else 0
    
    # 计算洋葱特性
    optimizer = OnionNetworkOptimizer(original_G)
    onion_metrics = optimizer.analyze_onion_structure(onion_G)
    
    # 计算相同度数节点的连通性
    onion_connectivity = np.mean(list(onion_metrics['onion_property'].values()))
    
    info_text = f"""
网络对比信息

基本信息:
• 节点数: {len(original_G)}
• 边数: {len(original_G.edges())}
• 平均度数: {np.mean(degrees_orig_list):.2f}

{attack_name}鲁棒性对比:
• 原始网络: {original_R:.4f}
• 洋葱结构: {onion_R:.4f}
• 改进幅度: {onion_R - original_R:.4f}
• 相对改进: {((onion_R - original_R) / original_R * 100):.1f}%

结构特征对比:
• 平均聚类系数:
  - 原始: {avg_clustering_orig:.3f}
  - 洋葱: {avg_clustering_onion:.3f}

• 核心节点连接密度:
  - 原始: {core_density_orig:.3f}
  - 洋葱: {core_density_onion:.3f}

• 洋葱特性 (相同度数节点连通性): {onion_connectivity:.3f}

• 最高度数: {max(degrees_orig_list)}
• 最低度数: {min(degrees_orig_list)}

洋葱结构特点:
• 高度数节点形成致密核心
• 节点按度数分层排列
• 同度数节点有较高连通性
• 在{attack_name}下更加鲁棒
    """
    
    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # 4. 攻击模拟曲线对比
    ax4 = fig.add_subplot(gs[1, :])
    
    print(f"进行{attack_name}模拟对比...")
    if attack_type == "random":
        num_sims = 15
        original_mean, original_std, onion_mean, onion_std = attack_simulation_comparison(
            original_G, onion_G, attack_type="random", num_simulations=num_sims)
        title_suffix = f"({num_sims}次模拟平均)"
    else:
        original_mean, original_std, onion_mean, onion_std = attack_simulation_comparison(
            original_G, onion_G, attack_type="HDA")
        title_suffix = ""
    
    N = len(original_G.nodes())
    q_values = np.arange(len(original_mean)) / N
    
    # 绘制攻击曲线
    ax4.plot(q_values, original_mean, 'r-', label='原始网络', linewidth=2)
    ax4.plot(q_values, onion_mean, 'b-', label='洋葱结构网络', linewidth=2)
    
    # 添加标准差阴影（如果有多次模拟）
    if attack_type == "random":
        ax4.fill_between(q_values, original_mean - original_std, original_mean + original_std, 
                         color='red', alpha=0.2, label='原始网络 ±1σ')
        ax4.fill_between(q_values, onion_mean - onion_std, onion_mean + onion_std, 
                         color='blue', alpha=0.2, label='洋葱结构 ±1σ')
    
    ax4.set_title(f'{attack_name}下的网络鲁棒性对比 {title_suffix}', fontsize=14, fontweight='bold')
    ax4.set_xlabel('移除节点比例 q', fontsize=12)
    ax4.set_ylabel('最大连通分量比例 s(q)', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数：测试洋葱网络优化算法"""
    print("=== 洋葱网络结构优化算法（HDA攻击鲁棒性）===\n")
    
    # 生成测试网络
    networks = {
        'BA网络': nx.barabasi_albert_graph(100, 3, seed=42),
        'WS网络': nx.watts_strogatz_graph(100, 6, 0.3, seed=42),
        'ER网络': nx.erdos_renyi_graph(100, 0.06, seed=42)
    }
    
    for name, G in networks.items():
        print(f"\n{'='*50}")
        print(f"测试网络: {name}")
        print(f"{'='*50}")
        
        # 确保网络连通
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            print(f"取最大连通分量: 节点数 {len(G.nodes())}, 边数 {len(G.edges())}")
        
        print(f"节点数: {len(G.nodes())}, 边数: {len(G.edges())}")
        
        # 创建优化器
        optimizer = OnionNetworkOptimizer(G)
        
        # 计算原始网络鲁棒性（HDA攻击）
        print("计算原始网络HDA攻击鲁棒性...")
        original_R = optimizer.robustness_measure(G, attack_type="HDA")
        
        # 优化网络
        print("优化网络结构...")
        max_iterations = 5000  # 减少迭代次数以加快演示
        onion_G = optimizer.optimize_network(max_iterations=max_iterations, threshold=0.0, tolerance=0.01)
        
        # 计算优化后网络鲁棒性（HDA攻击）
        print("计算优化后网络HDA攻击鲁棒性...")
        onion_R = optimizer.robustness_measure(onion_G, attack_type="HDA")
        
        # 输出结果
        print(f"\n结果:")
        print(f"原始网络HDA攻击鲁棒性: {original_R:.4f}")
        print(f"优化后网络HDA攻击鲁棒性: {onion_R:.4f}")
        print(f"改进幅度: {onion_R - original_R:.4f}")
        print(f"相对改进: {((onion_R - original_R) / original_R * 100):.1f}%")
        compare_degree_distributions(G, onion_R)
        # 可视化第一个网络
        if name == 'BA网络':
            print(f"\n生成可视化图表...")
            visualize_onion_comparison(G, onion_G, original_R, onion_R, attack_type="HDA")
        
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
   
   
# 添加接口函数，供外部调用
def Onion_method(G, max_iterations=10000, threshold=0.0, verbose=False):
    """
    洋葱网络优化算法接口函数
    
    Parameters:
    G: networkx.Graph - 输入网络
    max_iterations: int - 最大迭代次数
    threshold: float - 接受交换的阈值
    verbose: bool - 是否输出详细信息
    
    Returns:
    tuple - (优化后的网络, 原始鲁棒性, 优化后鲁棒性)
    """
    # 确保输入网络是连通的
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        if verbose:
            print(f"取最大连通分量: 节点数 {len(G.nodes())}, 边数 {len(G.edges())}")
    
    # 创建优化器
    optimizer = OnionNetworkOptimizer(G)
    
    # 计算原始网络鲁棒性
    if verbose:
        print("计算原始网络HDA攻击鲁棒性...")
    original_R = optimizer.robustness_measure(G, attack_type="HDA")
    
    # 优化网络
    if verbose:
        print("优化网络结构...")
    
    onion_G = optimizer.optimize_network(max_iterations=max_iterations, 
                                        threshold=threshold, 
                                        tolerance=0.01, 
                                        verbose=verbose)
    return onion_G

if __name__ == "__main__":
    # main()
    test_graph = nx.barabasi_albert_graph(100, 3, seed=42)
    onion_G = Onion_method(test_graph)
    plot_robustness_comparison(test_graph, onion_G)
    plot_random_robustness_comparison(test_graph, onion_G)
    compare_degree_distributions(test_graph, onion_G)
    