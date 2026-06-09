"""
Evaluation metrics for network dismantling.
"""
from typing import List, Tuple, Optional

import networkx as nx
import numpy as np
from scipy import integrate


def gcc_size(G: nx.Graph) -> int:
    """Giant connected component size."""
    if G.number_of_nodes() == 0:
        return 0
    return max(len(c) for c in nx.connected_components(G))


def accumulated_reward(rewards: List[float]) -> float:
    return float(np.sum(rewards))


def gcc_ratios(G: nx.Graph, removal_sequence: List[int]) -> List[Tuple[float, float]]:
    """
    Compute (q, sigma(q)) curve.
    Returns list of (fraction_removed, gcc_ratio).
    """
    n = G.number_of_nodes()
    if n == 0:
        return []
    G_tmp = G.copy()
    results = [(0.0, gcc_size(G_tmp) / n)]
    for i, node in enumerate(removal_sequence):
        if node in G_tmp:
            G_tmp.remove_node(node)
        if G_tmp.number_of_nodes() == 0:
            results.append(((i + 1) / n, 0.0))
            break
        results.append(((i + 1) / n, gcc_size(G_tmp) / n))
    return results


def dismantling_auc(G: nx.Graph, removal_sequence: List[int], q_max: float = 1.0) -> float:
    """
    Area under the dismantling curve sigma(q).
    Lower is better.
    """
    curve = gcc_ratios(G, removal_sequence)
    if not curve:
        return 0.0
    qs = np.array([c[0] for c in curve])
    sigmas = np.array([c[1] for c in curve])
    # Clip to q_max
    mask = qs <= q_max
    qs = qs[mask]
    sigmas = sigmas[mask]
    if len(qs) < 2:
        return float(sigmas[0]) if len(sigmas) > 0 else 0.0
    return float(integrate.simpson(y=sigmas, x=qs))


def critical_overlap_ratio(
    seq: List[int],
    oracle_seq: List[int],
    k: Optional[int] = None,
) -> float:
    """
    Fraction of top-k removals that overlap with oracle (e.g., Global CI).
    """
    if k is None:
        k = min(len(seq), len(oracle_seq))
    k = max(1, min(k, len(seq), len(oracle_seq)))
    top_seq = set(seq[:k])
    top_oracle = set(oracle_seq[:k])
    if len(top_oracle) == 0:
        return 0.0
    return len(top_seq & top_oracle) / len(top_oracle)


def time_to_fragmentation(
    G: nx.Graph,
    removal_sequence: List[int],
    threshold: float = 0.5,
) -> int:
    """
    Number of removals until GCC drops below threshold * initial_GCC.
    Returns -1 if threshold never reached.
    """
    n = G.number_of_nodes()
    if n == 0:
        return -1
    initial_gcc = gcc_size(G)
    target = threshold * initial_gcc
    G_tmp = G.copy()
    for i, node in enumerate(removal_sequence):
        if node in G_tmp:
            G_tmp.remove_node(node)
        if G_tmp.number_of_nodes() == 0:
            return i + 1
        if gcc_size(G_tmp) < target:
            return i + 1
    return -1


def compute_all_metrics(
    G: nx.Graph,
    removal_sequence: List[int],
    oracle_sequence: Optional[List[int]] = None,
) -> dict:
    """
    Compute all metrics for a single dismantling run.
    """
    n = G.number_of_nodes()
    curve = gcc_ratios(G, removal_sequence)
    metrics = {
        "n_nodes": n,
        "seq_length": len(removal_sequence),
        "auc": dismantling_auc(G, removal_sequence),
        "tau": time_to_fragmentation(G, removal_sequence, threshold=0.5),
    }
    if oracle_sequence is not None:
        metrics["critical_overlap"] = critical_overlap_ratio(removal_sequence, oracle_sequence)
    if curve:
        metrics["final_gcc_ratio"] = curve[-1][1]
    return metrics
