"""
Curriculum Learning Training for ADAPT with Rolling-Window Environment.

Trains ADAPT with a k-hop decreasing curriculum:
  Phase 0: k=20 (near-global)  -> Phase 4: k=1 (extremely local)

Each phase resumes from the previous phase's best checkpoint.
Evaluation uses remove_num (primary) and AUC (secondary).
"""
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
import numpy as np
import torch
from tqdm import tqdm

import networkx as nx
from network_dismantling.ADAPT.envs.rolling_env import RollingDismantleEnv
from dataset.loader import load_split
from network_dismantling.ADAPT.models.actor_critic import ActorCritic
from network_dismantling.ADAPT.agents.ppo_agent import PPOAgent


def load_real_world_graphs(data_dir="dataset/test/real"):
    """Load real-world networks from testdata, excluding synthetic BA graphs."""
    import re, tempfile
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
    env: RollingDismantleEnv,
    graphs: list,
    device: torch.device,
    max_removals_ratio: float = 1.0,
) -> dict:
    model.eval()
    all_rewards = []
    all_lengths = []
    all_remove_nums = []
    all_aucs = []
    for idx, G in enumerate(graphs):
        if G.number_of_nodes() == 0:
            continue
        n = G.number_of_nodes()
        try:
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
        except Exception as e:
            print(f"  [Eval Warning] Graph {idx} (n={n}) failed: {e}")
            continue
        # Periodic GPU memory cleanup to prevent OOM on large eval sets
        if device.type == "cuda" and (idx + 1) % 20 == 0:
            torch.cuda.empty_cache()
    return {
        "mean_return": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "mean_length": float(np.mean(all_lengths)) if all_lengths else 0.0,
        "mean_remove_num": float(np.mean(all_remove_nums)) if all_remove_nums else float("inf"),
        "mean_auc": float(np.mean(all_aucs)) if all_aucs else float("inf"),
    }


def run_training_phase(
    phase_idx: int,
    phase_config: dict,
    base_config: dict,
    model: ActorCritic,
    agent: PPOAgent,
    train_graphs: list,
    eval_graphs: list,
    device: torch.device,
    save_dir: Path,
    resume_ckpt: Path = None,
    seed: int = 42,
) -> Path:
    """
    Run a single curriculum phase.
    Returns path to the best checkpoint.
    """
    k_hop = phase_config["k_hop"]
    epochs = phase_config["epochs"]
    lr = phase_config.get("lr", base_config.get("lr", 3e-4))
    entropy_coef = phase_config.get("entropy_coef", base_config.get("entropy_coef", 0.01))

    print(f"\n{'='*70}")
    print(f"Curriculum Phase {phase_idx}: k_hop={k_hop}, epochs={epochs}, lr={lr}, entropy={entropy_coef}")
    print(f"{'='*70}")

    # Update optimizer LR
    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = lr
    # Update entropy coefficient
    agent.entropy_coef = entropy_coef

    # Create environment for this phase
    env = RollingDismantleEnv(
        k_hop=k_hop,
        reward_metric=base_config.get("reward_metric", "gcc"),
        dynamics_mode=base_config.get("dynamics", "static"),
        max_nodes=base_config.get("max_nodes", 1000),
        seed=seed,
        device=device,
        reward_weights=base_config.get("reward_weights", None),
        terminal_reward_weights=base_config.get("terminal_reward_weights", None),
        node_feat_dim=base_config.get("node_feat_dim", 10),
    )

    # Resume if checkpoint provided
    start_epoch = 1
    best_eval_remove_num = float("inf")
    best_eval_return = -float("inf")
    if resume_ckpt is not None and resume_ckpt.exists():
        print(f"Resuming from: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_eval_remove_num = ckpt.get("best_eval_remove_num", float("inf"))
        best_eval_return = ckpt.get("best_eval_return", -float("inf"))
        print(f"Resumed at epoch {start_epoch}, best_remove_num={best_eval_remove_num:.2f}")

    update_steps = base_config.get("update_steps", 2048)
    eval_interval = base_config.get("eval_interval", 10)
    save_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(start_epoch, epochs + 1), desc=f"Phase {phase_idx} (k={k_hop})", initial=start_epoch, total=epochs)
    for epoch in pbar:
        eval_interval_dynamic = eval_interval if epoch <= 50 else 100

        # LR decay: halve every 50 epochs within phase
        if epoch > start_epoch and epoch % 50 == 0:
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"[LR decay @ epoch {epoch}] LR = {agent.optimizer.param_groups[0]['lr']:.2e}")

        # Collect rollouts
        collect_stats = agent.collect_rollouts(
            env=env,
            graphs=train_graphs,
            n_steps=update_steps,
            device=device,
            max_removals_ratio=base_config.get("attack_budget_ratio", 1.0),
        )

        # Update
        update_stats = agent.update(
            update_epochs=base_config.get("update_epochs", 4),
            batch_size=base_config.get("batch_size", 64),
        )

        pbar.set_postfix({
            "ep_return": f"{collect_stats['mean_ep_return']:.3f}",
            "policy_loss": f"{update_stats.get('policy_loss', 0):.4f}",
            "value_loss": f"{update_stats.get('value_loss', 0):.4f}",
        })

        # Eval
        if epoch % eval_interval_dynamic == 0:
            eval_stats = evaluate_policy(
                model, env, eval_graphs, device,
                max_removals_ratio=base_config.get("attack_budget_ratio", 1.0),
            )
            print(f"\n[Eval @ epoch {epoch}] return={eval_stats['mean_return']:.3f}, len={eval_stats['mean_length']:.1f}, remove_num={eval_stats['mean_remove_num']:.2f}, auc={eval_stats['mean_auc']:.4f}")

            if eval_stats["mean_remove_num"] < best_eval_remove_num:
                best_eval_remove_num = eval_stats["mean_remove_num"]
                best_eval_return = eval_stats["mean_return"]
                ckpt_path = save_dir / f"curriculum_k{k_hop}_best.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "config": base_config,
                    "epoch": epoch,
                    "best_eval_return": best_eval_return,
                    "best_eval_remove_num": best_eval_remove_num,
                    "phase": phase_idx,
                    "k_hop": k_hop,
                }, ckpt_path)
                print(f"Saved best checkpoint: {ckpt_path} (remove_num={best_eval_remove_num:.2f})")

        # Periodic save
        if epoch % 50 == 0:
            ckpt_path = save_dir / f"curriculum_k{k_hop}_epoch_{epoch}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "config": base_config,
                "epoch": epoch,
                "best_eval_return": best_eval_return,
                "best_eval_remove_num": best_eval_remove_num,
                "phase": phase_idx,
                "k_hop": k_hop,
            }, ckpt_path)

    # Final save
    final_ckpt = save_dir / f"curriculum_k{k_hop}_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "config": base_config,
        "epoch": epochs,
        "best_eval_return": best_eval_return,
        "best_eval_remove_num": best_eval_remove_num,
        "phase": phase_idx,
        "k_hop": k_hop,
    }, final_ckpt)
    print(f"Phase {phase_idx} complete. Final checkpoint: {final_ckpt}")

    return save_dir / f"curriculum_k{k_hop}_best.pt"


def main():
    parser = argparse.ArgumentParser(description="Train ADAPT with curriculum learning")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--start_phase", type=int, default=0, help="Which curriculum phase to start from")
    args = parser.parse_args()

    # Load config
    if args.config is None:
        config_path = Path(__file__).parent / "configs" / "curriculum.yaml"
    else:
        config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    max_nodes = config.get("max_nodes", 1000)

    # Load datasets
    print("Loading training dataset...")
    train_syn = load_split("train", "syn", max_nodes=max_nodes, num_workers=0)
    train_real = load_split("train", "real", max_nodes=max_nodes, num_workers=0)
    train_graphs = list(train_syn.values()) + list(train_real.values())
    print(f"Total training graphs: {len(train_graphs)} (syn={len(train_syn)}, real={len(train_real)})")

    eval_only_syn = config.get("eval_only_syn", False)
    eval_syn = load_split("validate", "syn", max_nodes=max_nodes)
    if eval_only_syn:
        eval_graphs = list(eval_syn.values())
        print(f"Total eval graphs: {len(eval_graphs)} (syn only, real-world skipped for speed)")
    else:
        eval_real = load_split("validate", "real", max_nodes=max_nodes)
        eval_graphs = list(eval_syn.values()) + list(eval_real.values())
        print(f"Total eval graphs: {len(eval_graphs)} (syn={len(eval_syn)}, real={len(eval_real)})")

    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    # Model (shared across phases)
    model = ActorCritic(
        node_feat_dim=config.get("node_feat_dim", 10),
        hidden_dim=config.get("hidden_dim", 64),
        gat_layers=config.get("gat_layers", 2),
        gat_heads=config.get("gat_heads", 4),
        gru_hidden=config.get("gru_hidden", 128),
        action_dim=config.get("action_dim", 32),
        max_nodes=config.get("max_nodes", 1000),
        dropout=config.get("dropout", 0.1),
    ).to(device)

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

    save_dir = Path(config.get("save_dir", "network_dismantling/ADAPT/checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    curriculum = config.get("curriculum", [])
    if not curriculum:
        print("ERROR: No curriculum schedule found in config!")
        return

    resume_ckpt = Path(args.resume) if args.resume else None

    # Run curriculum phases
    for phase_idx, phase_cfg in enumerate(curriculum):
        if phase_idx < args.start_phase:
            print(f"Skipping phase {phase_idx} (k={phase_cfg['k_hop']})")
            continue

        best_ckpt = run_training_phase(
            phase_idx=phase_idx,
            phase_config=phase_cfg,
            base_config=config,
            model=model,
            agent=agent,
            train_graphs=train_graphs,
            eval_graphs=eval_graphs,
            device=device,
            save_dir=save_dir,
            resume_ckpt=resume_ckpt,
            seed=args.seed,
        )

        # Next phase resumes from this phase's best checkpoint
        resume_ckpt = best_ckpt

    print(f"\n{'='*70}")
    print("Curriculum training complete!")
    print(f"Best checkpoints saved in: {save_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
