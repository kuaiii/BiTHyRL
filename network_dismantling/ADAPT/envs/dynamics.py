"""
Dynamic topology evolution rules for network dismantling environment.
"""
import networkx as nx
import numpy as np


def apply_static(G: nx.Graph, rng: np.random.Generator = None) -> nx.Graph:
    """No change."""
    return G


def apply_random_mobility(
    G: nx.Graph,
    p_m: float = 0.1,
    rng: np.random.Generator = None,
    seed: int = None,
) -> nx.Graph:
    """
    Random mobility: each node moves with probability p_m.
    For geometric graphs this means rewiring edges based on new positions.
    For other graphs we simulate by rewiring a fraction of edges.
    """
    if G.number_of_nodes() <= 1:
        return G
    if rng is None:
        rng = np.random.default_rng(seed)

    # Simple model: for each node, with prob p_m, remove a random edge and add a new one
    G = G.copy()
    nodes = list(G.nodes())
    for node in nodes:
        if rng.random() < p_m:
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 0 and G.number_of_edges() > G.number_of_nodes():
                # remove one random edge
                nb = rng.choice(neighbors)
                G.remove_edge(node, nb)
            # add one new edge to a non-neighbor
            non_neighbors = [v for v in nodes if v != node and not G.has_edge(node, v)]
            if non_neighbors:
                new_nb = rng.choice(non_neighbors)
                G.add_edge(node, new_nb)
    return G


def apply_targeted_churn(
    G: nx.Graph,
    rho: float = 0.05,
    rng: np.random.Generator = None,
    seed: int = None,
) -> nx.Graph:
    """
    Targeted churn: replace rho fraction of non-critical (low-degree) nodes.
    A replaced node keeps its ID but loses all edges and gains new random edges.
    """
    if G.number_of_nodes() <= 1:
        return G
    if rng is None:
        rng = np.random.default_rng(seed)

    G = G.copy()
    nodes = list(G.nodes())
    n_replace = max(1, int(round(rho * len(nodes))))

    # sort by degree ascending -> non-critical nodes first
    nodes_by_deg = sorted(nodes, key=lambda v: G.degree(v))
    to_replace = nodes_by_deg[:n_replace]

    for node in to_replace:
        # Remove all edges of this node
        neighbors = list(G.neighbors(node))
        for nb in neighbors:
            if G.has_edge(node, nb):
                G.remove_edge(node, nb)
        # Add new random edges (number equal to previous degree, or at least 1)
        new_deg = max(1, len(neighbors))
        candidates = [v for v in nodes if v != node]
        if candidates:
            new_nbs = rng.choice(candidates, size=min(new_deg, len(candidates)), replace=False)
            for nb in new_nbs:
                G.add_edge(node, nb)

    return G


def apply_dynamics(
    G: nx.Graph,
    mode: str = "static",
    rng: np.random.Generator = None,
    seed: int = None,
    **kwargs,
) -> nx.Graph:
    """
    Apply dynamic rule to graph.

    Parameters
    ----------
    G : nx.Graph
        Current graph.
    mode : str
        One of {"static", "random_mobility", "targeted_churn"}.
    rng : np.random.Generator
        Random number generator.
    seed : int
        Fallback seed if rng is None.
    **kwargs : extra parameters for specific dynamics.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    mode = mode.lower()
    if mode == "static":
        return apply_static(G, rng)
    elif mode == "random_mobility":
        return apply_random_mobility(G, p_m=kwargs.get("p_m", 0.1), rng=rng)
    elif mode == "targeted_churn":
        return apply_targeted_churn(G, rho=kwargs.get("rho", 0.05), rng=rng)
    else:
        raise ValueError(f"Unknown dynamics mode '{mode}'")
