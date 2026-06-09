"""
Networkx-based heuristic sorters for network dismantling.
Each function returns a list of node IDs sorted by removal priority.
Higher priority = earlier in the list.
"""
import numpy as np
import networkx as nx


def degree_scores(G: nx.Graph) -> list:
    degree_node = G.degree()
    sorted_nodes = sorted(degree_node, key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_nodes]


def pagerank_scores(G: nx.Graph, **kwargs) -> list:
    pr = nx.pagerank(G, **kwargs)
    sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_nodes]


def betweenness_scores(G: nx.Graph, **kwargs) -> list:
    bc = nx.betweenness_centrality(G, **kwargs)
    sorted_nodes = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_nodes]


def eigenvector_scores(G: nx.Graph, **kwargs) -> list:
    try:
        ec = nx.eigenvector_centrality(G, **kwargs)
    except nx.PowerIterationFailedConvergence:
        ec = nx.eigenvector_centrality(G, max_iter=1000, **kwargs)
    sorted_nodes = sorted(ec.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_nodes]


def random_scores(G: nx.Graph, seed: int = None) -> list:
    nodes = list(G.nodes())
    rng = np.random.default_rng(seed)
    rng.shuffle(nodes)
    return nodes
