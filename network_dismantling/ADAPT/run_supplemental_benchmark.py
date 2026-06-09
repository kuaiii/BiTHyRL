"""
Supplemental benchmark: Add GDM, GDM+R, and skipped methods for n>=500.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path
from multiprocessing import Process, Queue

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import networkx as nx
import torch
import pandas as pd

from network_dismantling.ADAPT.envs.topology_generators import generate_graph
from network_dismantling.ADAPT.envs.network_env import NetworkDismantleEnv
from network_dismantling.ADAPT.models.actor_critic import ActorCritic
from network_dismantling.ADAPT.evaluate import arch_dismantle_sequence
from network_dismantling.ADAPT.utils.metrics import dismantling_auc, time_to_fragmentation
from network_dismantling.unified_interface import METHOD_REGISTRY

warnings.filterwarnings("ignore")

# Methods to supplement
SUPPLEMENTAL_METHODS = {
    "GDM": "GDM",
    "GDM+R": "GDM+R",
}

# For n>=500, also run previously skipped methods
LARGE_GRAPH_METHODS = {
    "Betweenness": "betweenness",
    "Eigenvector": "eigenvector",
    "Entanglement_S": "entanglement_small",
    "Entanglement_M": "entanglement_mid",
    "Entanglement_L": "entanglement_large",
    "Vertex_Entanglement": "vertex_entanglement",
}


def _run_method_worker(name, method_key, G_dict, stop_condition, result_queue):
    import networkx as nx
    try:
        G = nx.node_link_graph(G_dict)
        from network_dismantling.unified_interface import METHOD_REGISTRY
        func = METHOD_REGISTRY.get(method_key)
        if func is None:
            result_queue.put((None, None, "not_registered"))
            return
        t0 = time.time()
        seq = func(G, stop_condition=stop_condition)
        elapsed = time.time() - t0
        result_queue.put((seq, elapsed, None))
    except Exception as e:
        result_queue.put((None, None, str(e)))


def run_method(name, method_key, G, stop_condition=1, timeout=60):
    G_dict = nx.node_link_data(G)
    result_queue = Queue()
    p = Process(target=_run_method_worker, args=(name, method_key, G_dict, stop_condition, result_queue))
    p.start()
    p.join(timeout=timeout)
    
    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join(timeout=5)
        return None, None, "timeout"
    
    if not result_queue.empty():
        return result_queue.get_nowait()
    else:
        return None, None, "no_result"


def supplement_scale(n_nodes, n_graphs_per_type, output_dir, existing_df=None):
    graph_types = [
        ("BA", "sparse"),
        ("ER", "sparse"),
        ("WS", "sparse"),
        ("BA", "medium"),
        ("ER", "medium"),
        ("tree", "extremely_sparse"),
        ("complete", "complete"),
    ]

    if n_nodes <= 100:
        timeout = 60
    elif n_nodes <= 200:
        timeout = 120
    elif n_nodes <= 500:
        timeout = 300
    else:
        timeout = 600

    # Determine which methods to run
    methods_to_run = dict(SUPPLEMENTAL_METHODS)
    if n_nodes >= 500:
        methods_to_run.update(LARGE_GRAPH_METHODS)

    records = []

    for dataset, sparsity in graph_types:
        for seed_offset in range(n_graphs_per_type):
            seed = n_nodes * 1000 + hash(dataset + sparsity) % 10000 + seed_offset
            try:
                G = generate_graph(dataset=dataset, n_nodes=n_nodes, sparsity=sparsity, seed=seed)
            except Exception as e:
                print(f"  Skip {dataset}/{sparsity} n={n_nodes} seed={seed}: {e}")
                continue

            if G.number_of_nodes() == 0:
                continue

            print(f"[n={n_nodes}] {dataset}/{sparsity} seed={seed} | nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

            for method_name, method_key in methods_to_run.items():
                # Skip if already exists in existing_df
                if existing_df is not None:
                    mask = (
                        (existing_df["n_nodes"] == n_nodes)
                        & (existing_df["dataset"] == dataset)
                        & (existing_df["sparsity"] == sparsity)
                        & (existing_df["seed"] == seed)
                        & (existing_df["method"] == method_name)
                        & (existing_df["error"].isna())
                    )
                    if mask.any():
                        print(f"  Skip existing {method_name}")
                        continue

                seq, elapsed, error = run_method(
                    method_name, method_key, G,
                    stop_condition=1, timeout=timeout,
                )

                if error:
                    records.append({
                        "n_nodes": n_nodes,
                        "dataset": dataset,
                        "sparsity": sparsity,
                        "seed": seed,
                        "method": method_name,
                        "auc": None,
                        "tau": None,
                        "time": None,
                        "error": error,
                    })
                else:
                    try:
                        auc = dismantling_auc(G, seq)
                        tau = time_to_fragmentation(G, seq, threshold=0.5)
                        records.append({
                            "n_nodes": n_nodes,
                            "dataset": dataset,
                            "sparsity": sparsity,
                            "seed": seed,
                            "method": method_name,
                            "auc": auc,
                            "tau": tau if tau >= 0 else None,
                            "time": elapsed,
                            "error": None,
                        })
                    except Exception as e:
                        records.append({
                            "n_nodes": n_nodes,
                            "dataset": dataset,
                            "sparsity": sparsity,
                            "seed": seed,
                            "method": method_name,
                            "auc": None,
                            "tau": None,
                            "time": elapsed,
                            "error": f"metric_err:{e}",
                        })

    if records:
        df = pd.DataFrame(records)
        # Merge with existing
        if existing_df is not None:
            combined = pd.concat([existing_df, df], ignore_index=True)
        else:
            combined = df
        out_path = Path(output_dir) / f"benchmark_n{n_nodes}.csv"
        combined.to_csv(out_path, index=False)
        print(f"Saved {out_path}")
        return combined
    else:
        print(f"No new records for n={n_nodes}")
        return existing_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=int, nargs="+", default=[50, 100, 200, 400, 500, 1000])
    parser.add_argument("--n_graphs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    for n in args.scales:
        print(f"\n{'='*60}")
        print(f"Supplementing n={n}")
        print(f"{'='*60}")
        
        existing_path = Path(args.output_dir) / f"benchmark_n{n}.csv"
        existing_df = None
        if existing_path.exists():
            existing_df = pd.read_csv(existing_path)
            print(f"Loaded existing {existing_path} with {len(existing_df)} rows")
        
        supplement_scale(n, args.n_graphs, args.output_dir, existing_df)


if __name__ == "__main__":
    main()
