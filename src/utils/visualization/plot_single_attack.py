import network_construction as nc  # 统一重构接口
from src.topology.reconstruction import create_bimodal_network_exact
from src.topology.generators import load_graph, construct_random, construct_ba
import os
from matplotlib import pyplot as plt
import networkx as nx


def load_or_generate_graph(dataset_name):
    """
    从 GML 文件加载图。
    训练用的数据集在 dataset/all 中；测试的数据集在 dataset/testdata 中。
    
    Args:
        args (argparse.Namespace): 解析后的参数。
        
    Returns:
        tuple: (G, dataset_name)
    """
    gml_path = os.path.join('dataset', 'testdata', f'{dataset_name}.gml')
    if not os.path.exists(gml_path):
        # 如果 testdata 中未找到，尝试在 all 中查找（训练数据集）
        gml_path = os.path.join('dataset', 'all', f'{dataset_name}.gml')
    G, _ = load_graph(gml_path)
    
    return G

def construct_graph(G, node_num, edge_num, cover_rate):
    # 生成对比拓扑
    G_RA = construct_random(node_num, edge_num)
  
    G_BA = construct_ba(node_num, edge_num)
    
    G_BiT = create_bimodal_network_exact(G, hub_ratio=0.15)

    G_GA = nc.construct(G, algorithm='GA')

    G_ONION = nc.construct(G, algorithm='Onion', max_iter=100)

    G_ROMEM = nc.construct(G, algorithm='ROMEN')

    G_UNITY = nc.construct(G, algorithm='UNITY')


    # 将拓扑组织成字典
    all_G = {
        "G": G,   # 原始
        "G_RA": G_RA, # 随机
        "G_BA": G_BA, # 无标度
        "G_GA": G_GA, # GA 优化
        "G_ONION": G_ONION, # ONION 优化
        "G_ROMEM": G_ROMEM, # ROMEN 优化
        "G_UNITY": G_UNITY,  # UNITY 优化
        "G_BiT": G_BiT, # 双模态
    }
    return all_G

def get_attack_sequence(G, attack_mode):
    if attack_mode == "degree":
        return node_attack_degree(G)
    elif attack_mode == "random":
        return node_attack_random(G)
    else:
        raise ValueError(f"Invalid attack mode: {attack_mode}")

def simulate(all_G, attack_mode):
    x_results = {
        "G": [],   # 原始
        "G_RA": [], # 随机
        "G_BA": [], # 无标度
        "G_GA": [], # GA 优化
        "G_ONION": [], # ONION 优化
        "G_ROMEM": [], # ROMEN 优化
        "G_UNITY": [],  # UNITY 优化
        "G_BiT": [], # 双模态
    }
    y_results = {
        "G": [],   # 原始
        "G_RA": [], # 随机
        "G_BA": [], # 无标度
        "G_GA": [], # GA 优化
        "G_ONION": [], # ONION 优化
        "G_ROMEM": [], # ROMEN 优化
        "G_UNITY": [],  # UNITY 优化
        "G_BiT": [], # 双模态
    }
    
    for name, G in all_G.items():
        sequence = get_attack_sequence(G, attack_mode)
        x_results[name], y_results[name] = simulate_attack(G, sequence)
        
    return x_results, y_results

def simulate_attack(G, sequence):
    G = G.copy()
    initial_nodes = G.number_of_nodes()
    x_results = []
    y_results = []
    for node in sequence:
        edges_lists = G.neighbors(node)
        G.remove_edges_from(edges_lists)
        G.remove_node(node)
        x, y = calculate_metrics(G, initial_nodes)
        x_results.append(x)
        y_results.append(y)
    return x_results, y_results

def calculate_metrics(G, initial_nodes):
    gcc = calculate_gcc(G, initial_nodes)
    now_nodes = G.number_of_nodes()
    x = (initial_nodes - now_nodes) / initial_nodes
    return x, gcc

def calculate_gcc(G):
    return len(max(nx.connected_components(G), key=len))

def node_attack_degree(G):
    degree_list = list(G.degree())
    degree_list.sort(key=lambda x: x[1], reverse=True)
    return degree_list

def node_attack_random(all_G):
    return node_attack_random(all_G)

def plot_results(x_results, y_results):
    plt.figure(figsize=(10, 5))
    for name, x_result, y_result in zip(x_results.keys(), x_results.values(), y_results.values()):
        plt.plot(x_result, y_result, label=name)
    plt.legend()
    plt.show()
    
    
def main():
    dataset_name = "BA-100"
    cover_rate = 0.1
    G = load_or_generate_graph(dataset_name)
    all_G = construct_graph(G, G.number_of_nodes(), G.number_of_edges(), cover_rate)
    x_results, y_results = simulate(all_G, "degree")
    plot_results(x_results, y_results)