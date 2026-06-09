import random
import numpy as np
import networkx as nx


def compute_all_metrics(gcc_sizes, n_nodes, threshold=0.5):
    """
    计算所有核心指标

    Args:
        gcc_sizes: List[int], 每步移除后的GCC大小
        n_nodes: int, 初始节点数
        threshold: float, 渗流判定阈值

    Returns:
        dict: 包含 remove_num, auc, percolation_threshold
    """
    gcc_ratios = np.array([1.0] + [s / n_nodes for s in gcc_sizes])

    # remove_num: 使GCC比例低于阈值的移除节点数
    remove_num = n_nodes  # 默认全部移除
    for i, ratio in enumerate(gcc_ratios):
        if ratio < threshold:
            remove_num = i
            break

    # AUC: 攻击曲线下面积（梯形法则）
    f_values = np.arange(len(gcc_sizes) + 1) / n_nodes
    auc = np.trapz(gcc_ratios, f_values)

    # 渗流阈值
    percolation_threshold = remove_num / n_nodes

    return {
        'remove_num': int(remove_num),
        'auc': float(auc),
        'percolation_threshold': float(percolation_threshold),
        'attack_curve': list(zip(f_values.tolist(), gcc_ratios.tolist()))
    }


def validate_reproduction(original_results, reproduced_results, tolerance=0.03):
    """
    验证复现结果是否在允许误差范围内

    Args:
        original_results: dict, 论文中的指标值
        reproduced_results: dict, 复现的指标值
        tolerance: float, AUC相对误差容忍度

    Returns:
        (bool, dict): 是否通过验证，以及各项检查详情
    """
    checks = {}

    if 'auc' in original_results and 'auc' in reproduced_results:
        checks['auc'] = abs(original_results['auc'] - reproduced_results['auc']) / original_results['auc'] < tolerance
    else:
        checks['auc'] = None

    if 'remove_num' in original_results and 'remove_num' in reproduced_results:
        checks['remove_num'] = abs(original_results['remove_num'] - reproduced_results['remove_num']) <= 2
    else:
        checks['remove_num'] = None

    if 'percolation_threshold' in original_results and 'percolation_threshold' in reproduced_results:
        checks['percolation_threshold'] = abs(original_results['percolation_threshold'] - reproduced_results['percolation_threshold']) <= 0.02
    else:
        checks['percolation_threshold'] = None

    return all(v for v in checks.values() if v is not None), checks


def robustness_metric_r(G, attack_type='malicious', dynamic=True, seed=None):
    """
    Schneider 鲁棒性度量 R
    
    R = (1/(N+1)) * Σ_{i=0}^{N} s(i)/N
    其中 s(i) 是移除 i 个节点后的最大连通分量大小
    
    Args:
        G: networkx.Graph
        attack_type: 'malicious'(HDA) 或 'random'
        dynamic: True=动态攻击(每步重算度数), False=静态攻击(按初始度数排序)
        seed: 随机种子(仅影响random攻击和并列打破)
    
    Returns:
        float: R值
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    H = G.copy()
    N = H.number_of_nodes()
    if N == 0:
        return 0.0
    
    s_values = [len(max(nx.connected_components(H), key=len)) / N]
    
    if attack_type == 'malicious':
        if dynamic:
            # 动态HDA：每步移除当前度数最高的节点
            for _ in range(N):
                if H.number_of_nodes() == 0:
                    s_values.append(0.0)
                    continue
                d = dict(H.degree())
                max_d = max(d.values())
                candidates = [n for n, deg in d.items() if deg == max_d]
                target = random.choice(candidates) if len(candidates) > 1 else candidates[0]
                H.remove_node(target)
                s_values.append(len(max(nx.connected_components(H), key=len)) / N if H.number_of_nodes() > 0 else 0.0)
        else:
            # 静态HDA：按初始度数排序后依次移除
            degrees = dict(G.degree())
            node_order = sorted(degrees.keys(), key=lambda n: degrees[n], reverse=True)
            for node in node_order:
                if node in H.nodes():
                    H.remove_node(node)
                s_values.append(len(max(nx.connected_components(H), key=len)) / N if H.number_of_nodes() > 0 else 0.0)
    else:
        # 随机攻击
        nodes = list(H.nodes())
        random.shuffle(nodes)
        for node in nodes:
            if node in H.nodes():
                H.remove_node(node)
            s_values.append(len(max(nx.connected_components(H), key=len)) / N if H.number_of_nodes() > 0 else 0.0)
    
    return sum(s_values) / (N + 1)


def network_statistics(G):
    """网络基础统计信息"""
    stats = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'avg_degree': 2 * G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        'clustering_coeff': nx.average_clustering(G),
        'density': nx.density(G),
    }

    if nx.is_connected(G):
        stats['diameter'] = nx.diameter(G)
        stats['avg_shortest_path'] = nx.average_shortest_path_length(G)
    else:
        stats['diameter'] = None
        stats['avg_shortest_path'] = None

    return stats
