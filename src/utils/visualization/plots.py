# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt

def plt_graph(graph):
    nx.draw(graph, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_color="black",
            edge_color="gray")
    plt.show()

def plot_network(G, pos):
    # 调整布局使其更紧凑
    pos_compressed = {node: (x*0.5, y*0.5) for node, (x, y) in pos.items()}
    
    plt.figure(figsize=(10, 10), facecolor='white')
    
    # 绘制边
    nx.draw_networkx_edges(G, pos_compressed, 
                          edge_color='k',
                          alpha=0.6,
                          width=5)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos_compressed,
                          node_color='#2B5B84',
                          node_size=500,
                          alpha=0.8)
    
    # 添加节点标签
    nx.draw_networkx_labels(G, pos_compressed, 
                           font_size=12,
                           font_color='white',
                           font_weight='bold')
    
    plt.title("Network Topology", fontsize=14, pad=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
