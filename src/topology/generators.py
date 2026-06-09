# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import random
from bisect import bisect

# 根据经纬度计算地理距离
def disN(lon1, lat1, lon2, lat2):
    coords = np.radians([lon1, lat1, lon2, lat2])
    lon1, lat1, lon2, lat2 = coords
    d_lon = lon2 - lon1 
    d_lat = lat2 - lat1
    a = np.sin(d_lat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon/2)**2
    return 2 * 6371 * 1000 * np.arcsin(np.sqrt(a))

def load_graph(name, verbose=True):
    """
    加载图文件（支持 .gml 和 .graphml 格式）。
    
    Args:
        name: 图文件路径
        verbose: 是否输出详细信息（默认True，批量加载时建议设为False）
        
    Returns:
        (G, pos): 图对象和节点位置字典
    """
    if verbose:
        print(f"Loading graph from {name}...")
    
    G = None
    
    # 根据文件扩展名选择加载方式
    if name.lower().endswith('.graphml'):
        # 加载 .graphml 文件
        try:
            G = nx.read_graphml(name)
            # 转换为简单图（如果是多重图）
            if isinstance(G, nx.MultiGraph) or isinstance(G, nx.MultiDiGraph):
                G = nx.Graph(G)
            elif isinstance(G, nx.DiGraph):
                G = G.to_undirected()
        except Exception as e:
            if verbose:
                print(f"Failed to read graphml {name}: {e}")
            return None, None
    else:
        # 加载 .gml 文件
        try:
            G = nx.read_gml(name, label='id')
        except Exception as e:
            if verbose:
                print(f"Standard read failed for {name}: {e}. Retrying with MultiGraph patch...")
            try:
                with open(name, 'r', encoding='utf-8') as f:
                    content = f.read()
                if "graph [" in content and "multigraph 1" not in content:
                    content = content.replace("graph [", "graph [\n  multigraph 1", 1)
                G = nx.parse_gml(content, label='id')
                G = nx.Graph(G) # Convert to simple graph
            except Exception as e2:
                if verbose:
                    print(f"Failed to read {name} even with patch: {e2}")
                return None, None
    
    if G is None:
        return None, None

    nodes = G.nodes(data=True)
    pos = dict()
    for node, data in nodes:
        longitude = data.get('Longitude')
        latitude = data.get('Latitude')
        pos[node] = (longitude, latitude)

    edge_weight = dict()
    sum_weight = 0.0
    valid_edges_count = 0
    edges = G.edges(data=True)
    
    use_geo_dist = True
    nodes_with_pos = sum(1 for n in pos if pos[n] != (None, None))
    if nodes_with_pos < G.number_of_nodes() * 0.9: 
         use_geo_dist = False
         # 仅在verbose模式下输出警告
    
    for data in edges:
        node1, node2, _ = data
        (x1, y1) = pos[node1]
        (x2, y2) = pos[node2]
        
        if use_geo_dist and x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            try:
                dist = disN(x1, y1, x2, y2)
                edge_weight[(node1, node2)] = dist
                sum_weight += dist
                valid_edges_count += 1
            except:
                edge_weight[(node1, node2)] = None
        else:
            edge_weight[(node1, node2)] = None
            
    if valid_edges_count < G.number_of_edges() * 0.9:
         # 使用跳数作为权重（这是正常情况，不需要警告）
         for u, v in G.edges():
             edge_weight[(u, v)] = 1.0
         nx.set_edge_attributes(G, edge_weight, 'weight')
         return G, pos

    if valid_edges_count > 0:
        avg_weight = sum_weight / valid_edges_count
    else:
        avg_weight = 1.0
        
    for key in edge_weight.keys():
        if edge_weight[key] is not None:
            edge_weight[key] = round(edge_weight[key] / avg_weight, 2)
        else:
            edge_weight[key] = 1.0
            
    nx.set_edge_attributes(G, edge_weight, 'weight')
    return G, pos

def generate_BA_network(N, gamma, L):
    b = 1
    alpha = b / (gamma - 1)
    n = np.linspace(1, N, N)
    eta = n**(-alpha) 
    sum_eta = np.sum(eta)
    nom_eta = eta / sum_eta 
    random.shuffle(nom_eta) 
    cum_eta = np.array([np.sum(nom_eta[:i]) for i in range(N)])
    G = nx.Graph()

    c = 0
    G.add_nodes_from([x for x in range(N)])
    while c < L:
        i = bisect(cum_eta, np.random.rand(2)[0])
        j = bisect(cum_eta, np.random.rand(2)[1])
        if i >= N or j >= N or i == j or G.has_edge(i, j):
            continue
        G.add_edge(i, j)
        c += 1
    return G

def construct_random(nodes_num, edges_num):
    return nx.gnm_random_graph(nodes_num, edges_num)

def construct_ba(nodes_num, edges_num):
    return generate_BA_network(nodes_num, 2.5, edges_num)
