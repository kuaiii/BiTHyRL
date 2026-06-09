"""
Full benchmark: ARCH vs all network_dismantling baselines across scales.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path
from concurrent.futures import TimeoutError as FutureTimeoutError
from functools import partial
import multiprocessing as mp
from multiprocessing import Process, Queue

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import networkx as nx
import torch
import pandas as pd
from tqdm import tqdm

from network_dismantling.ADAPT.envs.topology_generators import generate_graph
from network_dismantling.ADAPT.envs.network_env import NetworkDismantleEnv
from network_dismantling.ADAPT.models.actor_critic import ActorCritic
from network_dismantling.ADAPT.evaluate import arch_dismantle_sequence
from network_dismantling.ADAPT.utils.metrics import dismantling_auc, time_to_fragmentation, gcc_ratios
from network_dismantling.unified_interface import METHOD_REGISTRY

warnings.filterwarnings("ignore")

# Methods to benchmark
METHODS = {
    "ADAPT": None,  # handled separately
    "Random": "random",
    "Degree": "degree",
    "Betweenness": "betweenness",
    "Pagerank": "pagerank",
    "Eigenvector": "eigenvector",
    "CoreHD": "CoreHD",
    "GND": "GND",
    "EI_s1": "EI_s1",
    "EI_s2": "EI_s2",
    "EGND": "EGND",
    "CI_L1": "CI_L1",
    "CI_L2": "CI_L2",
    "CI_L3": "CI_L3",
    "Entanglement_S": "entanglement_small",
    "Entanglement_M": "entanglement_mid",
    "Entanglement_L": "entanglement_large",
    "Vertex_Entanglement": "vertex_entanglement",
    "GDM": "GDM",
    "GDM+R": "GDM+R",
}


def _run_method_worker(name, method_key, G_dict, stop_condition, result_queue):
    """Worker function that runs in a separate process."""
    import networkx as nx
    try:
        G = nx.node_link_graph(G_dict)
        if name == "ADAPT":
            result_queue.put((None, None, "no_model_in_worker"))
            return
        
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


def run_method(name, method_key, G, stop_condition=1, timeout=60, arch_model=None, arch_env=None, arch_device=None):
    """Run a single method with timeout using a separate process."""
    if name == "ADAPT":
        if arch_model is None:
            return None, None, "no_model"
        t0 = time.time()
        try:
            seq, _ = arch_dismantle_sequence(G, arch_model, arch_env, arch_device, max_removals_ratio=0.2)
            elapsed = time.time() - t0
            return seq, elapsed, None
        except Exception as e:
            return None, None, str(e)

    # Use multiprocessing for timeout enforcement
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


def benchmark_scale(n_nodes, n_graphs_per_type, arch_model, arch_env, arch_device, output_dir):
    """Benchmark all methods at a given scale."""
    graph_types = [
        ("BA", "sparse"),
        ("ER", "sparse"),
        ("WS", "sparse"),
        ("BA", "medium"),
        ("ER", "medium"),
        ("tree", "extremely_sparse"),
        ("complete", "complete"),
    ]

    # Timeout per method based on scale
    if n_nodes <= 100:
        timeout = 60
    elif n_nodes <= 200:
        timeout = 120
    elif n_nodes <= 500:
        timeout = 300
    else:
        timeout = 600

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

            for method_name, method_key in METHODS.items():
                seq, elapsed, error = run_method(
                    method_name, method_key, G,
                    stop_condition=1, timeout=timeout,
                    arch_model=arch_model, arch_env=arch_env, arch_device=arch_device,
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

    df = pd.DataFrame(records)
    out_path = Path(output_dir) / f"benchmark_n{n_nodes}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/arch_best.pt")
    parser.add_argument("--scales", type=int, nargs="+", default=[50, 100, 200, 400, 500, 1000])
    parser.add_argument("--n_graphs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load ARCH model
    print("Loading ARCH model...")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    arch_model = ActorCritic(
        node_feat_dim=config.get("node_feat_dim", 7),
        hidden_dim=config.get("hidden_dim", 64),
        gat_layers=config.get("gat_layers", 2),
        gat_heads=config.get("gat_heads", 4),
        gru_hidden=config.get("gru_hidden", 128),
        action_dim=config.get("action_dim", 32),
        max_nodes=config.get("max_nodes", 1000),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    arch_model.load_state_dict(ckpt["model_state_dict"])
    arch_model.eval()
    arch_env = NetworkDismantleEnv(
        k_hop=config.get("k_hop", 2),
        reward_metric=config.get("reward_metric", "gcc"),
        dynamics_mode=config.get("dynamics", "static"),
        max_nodes=config.get("max_nodes", 1000),
        device=device,
    )
    print("ARCH model loaded.")

    all_dfs = []
    for n in args.scales:
        print(f"\n{'='*60}")
        print(f"Benchmarking n={n}")
        print(f"{'='*60}")
        df = benchmark_scale(n, args.n_graphs, arch_model, arch_env, device, args.output_dir)
        all_dfs.append(df)

    # Aggregate summary
    combined = pd.concat(all_dfs, ignore_index=True)
    summary = combined.groupby(["n_nodes", "method"]).agg({
        "auc": "mean",
        "tau": "mean",
        "time": "mean",
    }).reset_index()
    summary_path = Path(args.output_dir) / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
