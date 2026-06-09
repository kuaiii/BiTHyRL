"""
Main training script for ARCH.
Example:
    python network_dismantling/ARCH/train.py --dataset ER --n_nodes 100 --sparsity sparse --k_hop 2 --epochs 500
"""
import argparse
import sys
import os
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
import numpy as np
import torch
from tqdm import tqdm

import networkx as nx
from network_dismantling.ADAPT.envs.network_env import NetworkDismantleEnv
from dataset.loader import load_split
from network_dismantling.ADAPT.models.actor_critic import ActorCritic
from network_dismantling.ADAPT.agents.ppo_agent import PPOAgent


def load_real_world_graphs(data_dir="dataset/test/real"):
    """Load real-world networks from testdata, excluding synthetic BA graphs."""
    import re, tempfile, os
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    graphs = []
    for gml_file in sorted(data_path.glob("*.gml")):
        if gml_file.stem.startswith("BA-"):
            continue
        try:
            G = nx.read_gml(str(gml_file), label="id")
        except Exception:
            try:
                G = nx.read_gml(str(gml_file))
            except Exception:
                # Dedup edges manually
                with open(gml_file) as fh:
                    content = fh.read()
                edge_pattern = re.compile(r'edge\s*\[([^\]]*)\]', re.DOTALL)
                seen_edges = set()
                def dedup_edge(m):
                    text = m.group(1)
                    s = re.search(r'source\s+(\S+)', text)
                    t = re.search(r'target\s+(\S+)', text)
                    if s and t:
                        key = (s.group(1), t.group(1))
                        if key in seen_edges:
                            return ''
                        seen_edges.add(key)
                    return m.group(0)
                new_content = edge_pattern.sub(dedup_edge, content)
                tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(gml_file))
                with open(tmp_path, 'w') as fh:
                    fh.write(new_content)
                G = nx.read_gml(tmp_path, label="id")
                os.remove(tmp_path)
        G = nx.Graph(G)
        if not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()
        G = nx.convert_node_labels_to_integers(G, label_attribute="original_label")
        graphs.append((gml_file.stem, G))
    return graphs


@torch.no_grad()
def evaluate_policy(
    model: ActorCritic,
    env: NetworkDismantleEnv,
    graphs: list,
    device: torch.device,
    max_removals_ratio: float = 1.0,
) -> dict:
    model.eval()
    all_rewards = []
    all_lengths = []
    all_remove_nums = []
    all_aucs = []
    for G in graphs:
        if G.number_of_nodes() == 0:
            continue
        n = G.number_of_nodes()
        stop_10 = max(1, int(0.1 * n))
        obs = env.reset(G)
        h = model.init_hidden(device=device)
        prev_action = model.max_nodes + 1
        prev_reward = 0.0
        done = False
        ep_reward = 0.0
        ep_len = 0
        trace_lcc = []
        max_removals = max(1, int(max_removals_ratio * G.number_of_nodes()))
        while not done and ep_len < max_removals:
            action, _, _, _, h_next = model.get_action_and_value(
                obs, h, prev_action, prev_reward
            )
            action = int(action.item())
            obs_next, reward, done, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            gcc = info.get("gcc", 0)
            trace_lcc.append(gcc / n if n > 0 else 0.0)
            if not done:
                node_map = obs["node_map"]
                prev_action = node_map[action] if action < len(node_map) else model.max_nodes
                prev_reward = reward
                h = h_next
                obs = obs_next
        all_rewards.append(ep_reward)
        all_lengths.append(ep_len)
        # Compute remove_num and AUC
        rem_num = n
        for i, lcc_frac in enumerate(trace_lcc):
            if lcc_frac <= 0.1:
                rem_num = i + 1
                break
        q_vals = np.arange(1, len(trace_lcc) + 1) / n if n > 0 else np.array([])
        auc = float(np.trapz(trace_lcc, q_vals)) if len(trace_lcc) > 0 else 0.0
        all_remove_nums.append(rem_num)
        all_aucs.append(auc)
    return {
        "mean_return": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "mean_length": float(np.mean(all_lengths)) if all_lengths else 0.0,
        "mean_remove_num": float(np.mean(all_remove_nums)) if all_remove_nums else float("inf"),
        "mean_auc": float(np.mean(all_aucs)) if all_aucs else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser(description="Train ARCH agent")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default=None, help="Training dataset type (mixed if None)")
    parser.add_argument("--n_nodes", type=int, default=None, help="Number of nodes")
    parser.add_argument("--sparsity", type=str, default=None, help="Sparsity level")
    parser.add_argument("--k_hop", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    # Training hyperparameter overrides (for easy grid search on another machine)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--update_epochs", type=int, default=None)
    parser.add_argument("--update_steps", type=int, default=None)
    parser.add_argument("--entropy_coef", type=float, default=None)
    parser.add_argument("--value_coef", type=float, default=None)
    parser.add_argument("--p_clip", type=float, default=None)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--attack_budget_ratio", type=float, default=None)
    args = parser.parse_args()

    # Load config
    if args.config is None:
        config_path = Path(__file__).parent / "configs" / "default.yaml"
    else:
        config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override with CLI args
    if args.k_hop is not None:
        config["k_hop"] = args.k_hop
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.lr is not None:
        config["lr"] = args.lr
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.update_epochs is not None:
        config["update_epochs"] = args.update_epochs
    if args.update_steps is not None:
        config["update_steps"] = args.update_steps
    if args.entropy_coef is not None:
        config["entropy_coef"] = args.entropy_coef
    if args.value_coef is not None:
        config["value_coef"] = args.value_coef
    if args.p_clip is not None:
        config["p_clip"] = args.p_clip
    if args.gae_lambda is not None:
        config["gae_lambda"] = args.gae_lambda
    if args.gamma is not None:
        config["gamma"] = args.gamma
    if args.max_grad_norm is not None:
        config["max_grad_norm"] = args.max_grad_norm
    if args.attack_budget_ratio is not None:
        config["attack_budget_ratio"] = args.attack_budget_ratio

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    max_nodes = config.get("max_nodes", 1000)

    # Load pre-split dataset
    print("Loading training dataset...")
    train_syn = load_split("train", "syn", max_nodes=max_nodes)
    train_real = load_split("train", "real", max_nodes=max_nodes)
    graphs = list(train_syn.values()) + list(train_real.values())
    print(f"Total training graphs: {len(graphs)} (syn={len(train_syn)}, real={len(train_real)})")

    eval_syn = load_split("validate", "syn", max_nodes=max_nodes)
    eval_real = load_split("validate", "real", max_nodes=max_nodes)
    eval_graphs = list(eval_syn.values()) + list(eval_real.values())
    print(f"Total eval graphs: {len(eval_graphs)} (syn={len(eval_syn)}, real={len(eval_real)})")

    # CUDA diagnostics
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")

    # Environment
    env = NetworkDismantleEnv(
        k_hop=config["k_hop"],
        reward_metric=config.get("reward_metric", "gcc"),
        dynamics_mode=config.get("dynamics", "static"),
        max_nodes=config.get("max_nodes", 1000),
        seed=args.seed,
        device=device,
        reward_weights=config.get("reward_weights", None),
        node_feat_dim=config.get("node_feat_dim", 7),
    )

    # Model
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

    # Agent
    agent = PPOAgent(
        model=model,
        lr=config.get("lr", 3e-4),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        p_clip=config.get("p_clip", 0.2),
        entropy_coef=config.get("entropy_coef", 0.01),
        value_coef=config.get("value_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
    )

    # Resume from checkpoint if provided
    start_epoch = 1
    global_step = 0
    best_eval_return = -float("inf")
    best_eval_remove_num = float("inf")
    if args.resume is not None:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print(f"Resuming from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_eval_return = ckpt.get("best_eval_return") or -float("inf")
            global_step = (start_epoch - 1) * config.get("update_steps", 2048)
            print(f"Resumed at epoch {start_epoch}, best_eval_return={best_eval_return:.4f}")
        else:
            print(f"Warning: checkpoint {ckpt_path} not found, starting from scratch")

    # Training loop
    epochs = args.epochs if args.epochs is not None else config.get("epochs", 500)
    update_steps = config.get("update_steps", 2048)
    base_eval_interval = config.get("eval_interval", 10)
    save_dir = Path(config.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(start_epoch, epochs + 1), desc="Training", initial=start_epoch, total=epochs)
    for epoch in pbar:
        # Dynamic eval interval: every 10 epochs for first 50, then every 100
        eval_interval = base_eval_interval if epoch <= 50 else 100

        # LR decay: halve every 50 epochs (Phase 2)
        if epoch > start_epoch and epoch % 50 == 0:
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"[LR decay @ epoch {epoch}] LR = {agent.optimizer.param_groups[0]['lr']:.2e}")

        # Collect rollouts
        collect_stats = agent.collect_rollouts(
            env=env,
            graphs=graphs,
            n_steps=update_steps,
            device=device,
            max_removals_ratio=config.get("attack_budget_ratio", 1.0),
        )

        # Update
        update_stats = agent.update(
            update_epochs=config.get("update_epochs", 4),
            batch_size=config.get("batch_size", 64),
        )

        global_step += update_steps
        pbar.set_postfix({
            "ep_return": f"{collect_stats['mean_ep_return']:.3f}",
            "policy_loss": f"{update_stats.get('policy_loss', 0):.4f}",
            "value_loss": f"{update_stats.get('value_loss', 0):.4f}",
            "eval_int": eval_interval,
        })

        # Eval
        if epoch % eval_interval == 0:
            eval_stats = evaluate_policy(
                model, env, eval_graphs, device,
                max_removals_ratio=config.get("attack_budget_ratio", 1.0),
            )
            print(f"\n[Eval @ epoch {epoch}] mean_return={eval_stats['mean_return']:.4f}, mean_length={eval_stats['mean_length']:.2f}, mean_remove_num={eval_stats['mean_remove_num']:.2f}, mean_auc={eval_stats['mean_auc']:.4f}")

            # Save best by mean_return (legacy)
            if eval_stats["mean_return"] > best_eval_return:
                best_eval_return = eval_stats["mean_return"]
            # Save best by mean_remove_num (primary metric, lower is better)
            if eval_stats["mean_remove_num"] < best_eval_remove_num:
                best_eval_remove_num = eval_stats["mean_remove_num"]
                ckpt_path = save_dir / "arch_best.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_eval_return": best_eval_return,
                    "best_eval_remove_num": best_eval_remove_num,
                }, ckpt_path)
                print(f"Saved best checkpoint to {ckpt_path} (remove_num={best_eval_remove_num:.2f})")

        # Periodic save
        if epoch % 50 == 0:
            ckpt_path = save_dir / f"arch_epoch_{epoch}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
            }, ckpt_path)

    print("Training complete.")
    final_ckpt = save_dir / "arch_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "config": config,
        "epoch": epochs,
    }, final_ckpt)
    print(f"Final checkpoint saved to {final_ckpt}")


if __name__ == "__main__":
    main()
