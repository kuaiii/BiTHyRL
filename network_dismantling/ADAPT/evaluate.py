"""
Evaluation script for ARCH.
Compares ARCH against baselines (Random, Degree, Betweenness, Global CI).
Example:
    python network_dismantling/ARCH/evaluate.py --model_path checkpoints/arch_best.pt --dataset ER --n_nodes 100 --sparsity sparse
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import networkx as nx

from network_dismantling.ADAPT.envs.network_env import NetworkDismantleEnv
from network_dismantling.ADAPT.envs.topology_generators import generate_graph
from network_dismantling.ADAPT.models.actor_critic import ActorCritic
from network_dismantling.ADAPT.utils.metrics import compute_all_metrics, gcc_ratios
from network_dismantling.ADAPT.utils.visualizer import plot_dismantling_curves
from network_dismantling.unified_interface import dismantle


def collective_influence_dismantle(G: nx.Graph, l: int = 2, stop_condition: int = 1) -> list:
    """
    Pure Python implementation of Collective Influence (CI) dismantler.
    Returns removal sequence.
    """
    G = G.copy()
    seq = []
    while G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        ci_scores = {}
        for node in G.nodes():
            # BFS ball of radius l
            ball = nx.single_source_shortest_path_length(G, node, cutoff=l)
            ci = (degrees[node] - 1) * sum((degrees.get(j, 0) - 1) for j in ball if j != node)
            ci_scores[node] = ci
        if not ci_scores:
            break
        best = max(ci_scores, key=ci_scores.get)
        seq.append(best)
        G.remove_node(best)
        if G.number_of_nodes() == 0:
            break
        lcc = max(len(c) for c in nx.connected_components(G))
        if lcc <= stop_condition:
            break
    return seq


@torch.no_grad()
def arch_dismantle_sequence(
    G: nx.Graph,
    model: ActorCritic,
    env: NetworkDismantleEnv,
    device: torch.device,
    max_removals_ratio: float = 1.0,
    stop_condition: int = 1,
) -> tuple:
    """
    Run trained ARCH policy on a graph and return removal sequence + inference times.
    """
    model.eval()
    obs = env.reset(G)
    h = model.init_hidden(device=device)
    prev_action = model.max_nodes + 1
    prev_reward = 0.0
    done = False
    seq = []
    max_removals = max(1, int(max_removals_ratio * G.number_of_nodes()))
    inference_times = []

    while not done and len(seq) < max_removals:
        t0 = time.time()
        action, _, _, _, h_next = model.get_action_and_value(
            obs, h, prev_action, prev_reward
        )
        inference_times.append((time.time() - t0) * 1000)  # ms
        action = int(action.item())
        obs_next, reward, done, info = env.step(action)

        if action < len(obs["node_map"]):
            removed_node = obs["node_map"][action]
            seq.append(removed_node)
        # if terminate action, stop
        if action == len(obs["node_map"]):
            break

        if not done:
            node_map = obs["node_map"]
            prev_action = node_map[action] if action < len(node_map) else model.max_nodes
            prev_reward = reward
            h = h_next
            obs = obs_next

        # Check stop_condition independently
        if env.G_nx is not None and env.G_nx.number_of_nodes() > 0:
            lcc = max(len(c) for c in nx.connected_components(env.G_nx))
            if lcc <= stop_condition:
                break

    return seq, inference_times


def _fill_sequence(G: nx.Graph, seq: list) -> list:
    """Append remaining nodes by degree descending to make a full sequence."""
    removed = set(seq)
    remaining = [v for v in G.nodes() if v not in removed]
    remaining.sort(key=lambda v: G.degree(v), reverse=True)
    return list(seq) + remaining


def evaluate_arch(args):
    device = torch.device(args.device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    # Load model
    model = ActorCritic(
        node_feat_dim=config.get("node_feat_dim", 7),
        hidden_dim=config.get("hidden_dim", 64),
        gat_layers=config.get("gat_layers", 2),
        gat_heads=config.get("gat_heads", 4),
        gru_hidden=config.get("gru_hidden", 128),
        action_dim=config.get("action_dim", 32),
        max_nodes=config.get("max_nodes", 1000),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = NetworkDismantleEnv(
        k_hop=config.get("k_hop", 2),
        reward_metric=config.get("reward_metric", "gcc"),
        dynamics_mode=config.get("dynamics", "static"),
        max_nodes=config.get("max_nodes", 1000),
        seed=args.seed,
    )

    max_removals_ratio = config.get("attack_budget_ratio", 0.2)

    # Generate test graphs
    graphs = []
    for i in range(args.n_graphs):
        G = generate_graph(
            dataset=args.dataset,
            n_nodes=args.n_nodes,
            sparsity=args.sparsity,
            seed=args.seed + i,
        )
        graphs.append(G)

    methods = {
        "ADAPT": lambda G: arch_dismantle_sequence(G, model, env, device, max_removals_ratio)[0],
        "Random": lambda G: dismantle(G, method="random", stop_condition=1),
        "Degree": lambda G: dismantle(G, method="degree", stop_condition=1),
        "Betweenness": lambda G: dismantle(G, method="betweenness", stop_condition=1),
        "Global_CI": lambda G: collective_influence_dismantle(G, l=2, stop_condition=1),
    }

    all_results = {m: [] for m in methods}
    all_curves = {m: [] for m in methods}
    all_times = {m: [] for m in methods}

    for idx, G in enumerate(graphs):
        print(f"Evaluating graph {idx + 1}/{len(graphs)} (n={G.number_of_nodes()}, m={G.number_of_edges()})...")
        oracle_seq = methods["Global_CI"](G)

        for method_name, func in methods.items():
            t0 = time.time()
            if method_name == "ADAPT":
                seq, inf_times = arch_dismantle_sequence(G, model, env, device, max_removals_ratio)
                all_times[method_name].extend(inf_times)
            else:
                seq = func(G)
            elapsed = (time.time() - t0) * 1000  # total ms
            if method_name != "ADAPT":
                # approximate per-step time
                per_step = elapsed / max(len(seq), 1)
                all_times[method_name].append(per_step)

            # Ensure full sequence for fair AUC comparison
            seq = _fill_sequence(G, seq)

            metrics = compute_all_metrics(G, seq, oracle_sequence=oracle_seq)
            all_results[method_name].append(metrics)
            all_curves[method_name].append(gcc_ratios(G, seq))

    # Aggregate and print
    print("\n" + "=" * 80)
    print(f"Evaluation Results ({args.dataset}, n={args.n_nodes}, sparsity={args.sparsity}, {args.n_graphs} graphs)")
    print("=" * 80)
    print(f"{'Method':<15} {'AUC':>10} {'tau':>8} {'Overlap':>10} {'InfTime(ms)':>12}")
    print("-" * 80)

    avg_curves = {}
    for method_name in methods:
        aucs = [r["auc"] for r in all_results[method_name]]
        taus = [r["tau"] for r in all_results[method_name] if r["tau"] >= 0]
        overlaps = [r.get("critical_overlap", 0) for r in all_results[method_name]]
        times = all_times[method_name]
        print(f"{method_name:<15} {np.mean(aucs):>10.4f} {np.mean(taus) if taus else -1:>8.1f} {np.mean(overlaps):>10.4f} {np.mean(times):>12.2f}")

        # Average curves across graphs
        # Align by q (fraction removed) and average sigma
        all_qs = []
        all_sigs = []
        for curve in all_curves[method_name]:
            qs = [c[0] for c in curve]
            sigs = [c[1] for c in curve]
            all_qs.extend(qs)
            all_sigs.extend(sigs)
        if all_qs:
            # bin by q
            bins = np.linspace(0, 1, 51)
            bin_sigs = []
            for i in range(len(bins) - 1):
                mask = [(bins[i] <= q < bins[i + 1]) for q in all_qs]
                vals = [s for m, s in zip(mask, all_sigs) if m]
                bin_sigs.append(np.mean(vals) if vals else 0.0)
            avg_curves[method_name] = list(zip(bins[:-1], bin_sigs))

    # Plot
    if avg_curves:
        plot_path = Path(args.output_dir) / f"eval_{args.dataset}_{args.n_nodes}_{args.sparsity}.png"
        plot_dismantling_curves(
            avg_curves,
            title=f"{args.dataset} n={args.n_nodes} sparsity={args.sparsity}",
            save_path=str(plot_path),
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARCH")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="ER")
    parser.add_argument("--n_nodes", type=int, default=100)
    parser.add_argument("--sparsity", type=str, default="sparse")
    parser.add_argument("--n_graphs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate_arch(args)


if __name__ == "__main__":
    main()
