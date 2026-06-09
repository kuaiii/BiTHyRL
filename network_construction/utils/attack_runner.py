import random
import numpy as np
import networkx as nx
from tqdm import tqdm


def run_attack_batch(G, algorithms, stop_condition='threshold', stop_threshold=0.5, verbose=True):
    """
    批量运行多个攻击算法

    Args:
        G: networkx.Graph
        algorithms: dict, {name: algorithm_instance}
        stop_condition: str
        stop_threshold: float
        verbose: bool

    Returns:
        dict: {name: result_dict}
    """
    results = {}
    iterator = tqdm(algorithms.items(), desc="Running attacks") if verbose else algorithms.items()

    for name, algo in iterator:
        try:
            result = algo.dismantle(G, stop_condition=stop_condition, stop_threshold=stop_threshold)
            results[name] = result
        except Exception as e:
            if verbose:
                print(f"[ERROR] {name} failed: {e}")
            results[name] = {'error': str(e)}

    return results
