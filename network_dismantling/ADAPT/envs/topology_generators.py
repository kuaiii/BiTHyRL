"""
Topology generators covering the full sparsity spectrum from 0 to 1000 nodes.
"""
import networkx as nx
import numpy as np


def _ensure_connected(G: nx.Graph, seed: int = None) -> nx.Graph:
    """Return the largest connected component if graph is not connected."""
    if G.number_of_nodes() == 0:
        return G
    if nx.is_connected(G):
        return nx.convert_node_labels_to_integers(G, first_label=0)
    nodes = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(nodes).copy(), first_label=0)


def generate_tree(n: int, seed: int = None) -> nx.Graph:
    """Random tree (extremely sparse, avg degree ~2)."""
    if n <= 1:
        return nx.empty_graph(n)
    # networkx >=3.0 renamed random_tree -> random_labeled_tree
    if hasattr(nx, "random_labeled_tree"):
        return nx.random_labeled_tree(n, seed=seed)
    return nx.random_tree(n, seed=seed)


def generate_er(n: int, p: float, seed: int = None) -> nx.Graph:
    """Erdős–Rényi graph, ensured connected."""
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return _ensure_connected(G, seed=seed)


def generate_ba(n: int, m: int, seed: int = None) -> nx.Graph:
    """Barabási–Albert graph."""
    if n <= m:
        return nx.complete_graph(n)
    return nx.barabasi_albert_graph(n, m, seed=seed)


def generate_ws(n: int, k: int, p: float, seed: int = None) -> nx.Graph:
    """Watts–Strogatz small-world graph, ensured connected."""
    if n <= k:
        k = max(2, n - 1)
        k = k if k % 2 == 0 else k - 1
    G = nx.connected_watts_strogatz_graph(n, k, p, tries=100, seed=seed)
    return G


def generate_geometric(n: int, radius: float, seed: int = None) -> nx.Graph:
    """Random geometric graph in unit square, ensured connected."""
    if n <= 1:
        return nx.empty_graph(n)
    G = nx.random_geometric_graph(n, radius, seed=seed)
    return _ensure_connected(G, seed=seed)


def generate_complete(n: int) -> nx.Graph:
    """Complete graph K_n."""
    return nx.complete_graph(n)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
SPARSITY_PARAMS = {
    "extremely_sparse": {
        "tree": {},
        "er": {"p": 0.005},
        "ba": {"m": 1},
    },
    "sparse": {
        "er": {"p": 0.01},
        "ba": {"m": 2},
        "ws": {"k": 4, "p": 0.1},
    },
    "medium": {
        "er": {"p": 0.05},
        "ba": {"m": 4},
        "geometric": {"radius": 0.15},
    },
    "dense": {
        "er": {"p": 0.2},
        "geometric": {"radius": 0.5},
    },
    "complete": {
        "complete": {},
    },
}


def generate_graph(
    dataset: str = "ER",
    n_nodes: int = 100,
    sparsity: str = "sparse",
    seed: int = None,
) -> nx.Graph:
    """
    Generate a synthetic graph according to dataset type and sparsity level.

    Parameters
    ----------
    dataset : str
        One of {"ER", "BA", "WS", "tree", "geometric", "complete"}.
    n_nodes : int
        Number of nodes (0–1000).
    sparsity : str
        One of {"extremely_sparse", "sparse", "medium", "dense", "complete"}.
    seed : int, optional
        Random seed.

    Returns
    -------
    nx.Graph
        Connected undirected simple graph.
    """
    rng = np.random.default_rng(seed)
    seed_int = int(rng.integers(0, 2**31))

    if n_nodes <= 0:
        return nx.empty_graph(0)
    if n_nodes == 1:
        return nx.empty_graph(1)

    dataset = dataset.lower()
    sparsity = sparsity.lower()

    if dataset == "complete" or sparsity == "complete":
        return generate_complete(n_nodes)

    params = SPARSITY_PARAMS.get(sparsity, SPARSITY_PARAMS["sparse"]).get(dataset, {})

    if dataset == "er":
        p = params.get("p", 0.01)
        return generate_er(n_nodes, p, seed=seed_int)
    elif dataset == "ba":
        m = params.get("m", 2)
        return generate_ba(n_nodes, m, seed=seed_int)
    elif dataset == "ws":
        k = params.get("k", 4)
        p = params.get("p", 0.1)
        return generate_ws(n_nodes, k, p, seed=seed_int)
    elif dataset == "tree":
        return generate_tree(n_nodes, seed=seed_int)
    elif dataset == "geometric":
        radius = params.get("radius", 0.15)
        return generate_geometric(n_nodes, radius, seed=seed_int)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")


def generate_training_graphs(
    n_graphs: int = 1000,
    n_range: tuple = (20, 120),
    seed: int = 42,
):
    """
    Generate a diverse set of training graphs with mixed types and sparsities.
    Useful for non-stationary training.
    """
    rng = np.random.default_rng(seed)
    graphs = []
    datasets = ["ER", "BA", "WS", "tree", "geometric"]
    sparsities = ["extremely_sparse", "sparse", "medium", "dense"]

    for i in range(n_graphs):
        n = rng.integers(n_range[0], n_range[1] + 1)
        ds = rng.choice(datasets)
        sp = rng.choice(sparsities)
        g_seed = int(rng.integers(0, 2**31))
        G = generate_graph(dataset=ds, n_nodes=n, sparsity=sp, seed=g_seed)
        graphs.append(G)

    return graphs
