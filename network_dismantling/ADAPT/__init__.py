"""
ADAPT: Attentive Graph-based Agent-driven network dismantling via RL.
"""
from typing import List

import networkx as nx
import torch

from network_dismantling.unified_interface import register_method
from network_dismantling.ADAPT.envs.network_env import NetworkDismantleEnv
from network_dismantling.ADAPT.models.actor_critic import ActorCritic


def adapt_dismantle(
    G: nx.Graph,
    model_path: str = "network_dismantling/ADAPT/checkpoints/adaptive_best.pt",
    device: str = "cpu",
    k_hop: int = 2,
    max_removals_ratio: float = 1.0,
    **kwargs,
) -> List[int]:
    """
    Dismantle a graph using a trained ADAPT agent.

    Parameters
    ----------
    G : nx.Graph
        Input graph (will be standardized internally).
    model_path : str
        Path to trained checkpoint.
    device : str
        'cpu' or 'cuda'.
    k_hop : int
        Agent observation radius.
    max_removals_ratio : float
        Budget as fraction of initial nodes.

    Returns
    -------
    List[int]
        Removal sequence (aligned with standardized node IDs 0..n-1).
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    model = ActorCritic(
        node_feat_dim=config.get("node_feat_dim", 7),
        hidden_dim=config.get("hidden_dim", 64),
        gat_layers=config.get("gat_layers", 2),
        gat_heads=config.get("gat_heads", 4),
        gru_hidden=config.get("gru_hidden", 128),
        action_dim=config.get("action_dim", 32),
        max_nodes=config.get("max_nodes", 1000),
        dropout=config.get("dropout", 0.1),
    ).to(torch.device(device))

    # Compatibility: map old PyG GATConv parameter names to new ones
    state_dict = ckpt["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if "gat_encoder.convs." in k:
            if ".att_l" in k:
                k = k.replace(".att_l", ".att_src")
            elif ".att_r" in k:
                k = k.replace(".att_r", ".att_dst")
            elif ".lin_l.weight" in k:
                k = k.replace(".lin_l.weight", ".lin.weight")
            elif ".lin_r.weight" in k:
                # New GATConv has a single lin; skip the right projection
                continue
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    env = NetworkDismantleEnv(
        k_hop=k_hop,
        reward_metric=config.get("reward_metric", "gcc"),
        dynamics_mode=config.get("dynamics", "static"),
        max_nodes=config.get("max_nodes", 1000),
        node_feat_dim=config.get("node_feat_dim", 7),
    )

    max_removals = max(1, int(max_removals_ratio * G.number_of_nodes()))
    seq = []
    with torch.no_grad():
        obs = env.reset(G)
        h = model.init_hidden(torch.device(device))
        prev_action = model.max_nodes + 1
        prev_reward = 0.0
        done = False
        while not done and len(seq) < max_removals:
            action, _, _, _, h_next = model.get_action_and_value(
                obs, h, prev_action, prev_reward
            )
            action = int(action.item())
            obs_next, reward, done, info = env.step(action)
            if action < len(obs["node_map"]):
                seq.append(obs["node_map"][action])
            if action == len(obs["node_map"]):
                break
            if not done:
                node_map = obs["node_map"]
                prev_action = node_map[action] if action < len(node_map) else model.max_nodes
                prev_reward = reward
                h = h_next
                obs = obs_next

    return seq


# Register to unified interface (primary name)
register_method("adapt")(adapt_dismantle)

# Backward-compatible alias
register_method("adaptive")(adapt_dismantle)
