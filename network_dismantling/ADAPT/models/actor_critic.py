"""
Actor-Critic network for ARCH.
Combines GAT encoder, history GRU, and action/value heads.
"""
import math
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from network_dismantling.ADAPT.models.gat_encoder import GATEncoder
from network_dismantling.ADAPT.models.history_gru import HistoryGRU


class ActorCritic(nn.Module):
    """
    Actor-Critic for network dismantling with local observations.
    """

    def __init__(
        self,
        node_feat_dim: int = 7,
        hidden_dim: int = 64,
        gat_layers: int = 2,
        gat_heads: int = 4,
        gru_hidden: int = 128,
        action_dim: int = 32,
        max_nodes: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden
        self.action_dim = action_dim
        self.max_nodes = max_nodes

        self.gat_encoder = GATEncoder(
            in_channels=node_feat_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=gat_layers,
            heads=gat_heads,
            dropout=dropout,
        )

        # Action embedding: map global node IDs + terminate index to vectors
        # index max_nodes   -> terminate action
        # index max_nodes+1 -> padding / no previous action
        self.action_embedding = nn.Embedding(max_nodes + 2, action_dim)

        self.history_gru = HistoryGRU(
            graph_dim=hidden_dim,
            action_dim=action_dim,
            hidden_dim=gru_hidden,
        )

        # Actor MLP: input = [node_emb || graph_emb || global_feat || h_t]
        actor_in = hidden_dim * 2 + 1 + gru_hidden  # node + graph + global_est(1) + h_t
        self.actor_mlp = nn.Sequential(
            nn.Linear(actor_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Terminate action head (only depends on h_t)
        self.terminate_head = nn.Sequential(
            nn.Linear(gru_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Critic MLP
        self.critic_mlp = nn.Sequential(
            nn.Linear(gru_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GRUCell):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0.0)

    def _get_device(self):
        return next(self.parameters()).device

    def forward(
        self,
        obs: Dict[str, Any],
        h_prev: torch.Tensor,
        prev_action_global: Optional[torch.Tensor] = None,
        prev_reward: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        obs : dict
            Observation from environment.
        h_prev : Tensor, shape (gru_hidden,)
        prev_action_global : Tensor or int, optional
            Global node ID of previous action, or max_nodes for terminate.
        prev_reward : Tensor or float, optional
            Previous step reward.

        Returns
        -------
        action_logits : Tensor, shape (num_valid_actions,)
        value : Tensor, scalar
        h_t : Tensor, shape (gru_hidden,)
        """
        device = self._get_device()

        # Convert obs to tensors (handle both numpy and torch inputs)
        x = obs["x"]
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        x = x.to(device)

        edge_index = obs["edge_index"]
        if not torch.is_tensor(edge_index):
            edge_index = torch.from_numpy(edge_index).long()
        edge_index = edge_index.to(device)

        action_mask = obs["action_mask"]
        if not torch.is_tensor(action_mask):
            action_mask = torch.from_numpy(action_mask).bool()
        action_mask = action_mask.to(device)

        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # GAT encoder
        node_emb, graph_emb = self.gat_encoder(x, edge_index, num_nodes, num_edges)
        # node_emb: (N, hidden_dim), graph_emb: (hidden_dim,)

        # Global estimate feature per node (last column of x)
        if num_nodes > 0:
            global_feat = x[:, 6:7]  # (N, 1)
        else:
            global_feat = torch.zeros((0, 1), device=device)

        # History GRU inputs
        if prev_action_global is None:
            prev_action_idx = torch.tensor(self.max_nodes + 1, dtype=torch.long, device=device)
        else:
            if not torch.is_tensor(prev_action_global):
                prev_action_idx = torch.tensor(prev_action_global, dtype=torch.long, device=device)
            else:
                prev_action_idx = prev_action_global.long().to(device)

        action_emb = self.action_embedding(prev_action_idx)  # (action_dim,)

        if prev_reward is None:
            prev_reward_t = torch.tensor(0.0, dtype=torch.float, device=device)
        else:
            if not torch.is_tensor(prev_reward):
                prev_reward_t = torch.tensor(prev_reward, dtype=torch.float, device=device)
            else:
                prev_reward_t = prev_reward.float().to(device)

        h_t = self.history_gru(graph_emb, action_emb, prev_reward_t, h_prev.to(device))

        # Actor logits for each node
        if num_nodes > 0:
            h_t_expanded = h_t.unsqueeze(0).expand(num_nodes, -1)  # (N, gru_hidden)
            actor_input = torch.cat([
                node_emb,
                graph_emb.unsqueeze(0).expand(num_nodes, -1),
                global_feat,
                h_t_expanded,
            ], dim=-1)  # (N, actor_in)
            node_logits = self.actor_mlp(actor_input).squeeze(-1)  # (N,)
        else:
            node_logits = torch.zeros(0, device=device)

        # Terminate logit
        terminate_logit = self.terminate_head(h_t).squeeze(-1)  # scalar

        # Concatenate
        action_logits = torch.cat([node_logits, terminate_logit.unsqueeze(0)], dim=0)  # (N+1,)

        # Apply action mask
        action_logits = action_logits.masked_fill(~action_mask, float("-inf"))

        # Critic value
        value = self.critic_mlp(h_t).squeeze(-1)

        return action_logits, value, h_t

    def get_action_and_value(
        self,
        obs: Dict[str, Any],
        h_prev: torch.Tensor,
        prev_action_global: Optional[int] = None,
        prev_reward: Optional[float] = None,
        action: Optional[int] = None,
    ):
        """
        Convenience wrapper for PPO: returns action, log_prob, entropy, value, h_t.
        If action is provided, computes log_prob for that action (for updates).
        """
        action_logits, value, h_t = self.forward(
            obs, h_prev, prev_action_global, prev_reward
        )

        dist = torch.distributions.Categorical(logits=action_logits)

        if action is None:
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            action_t = torch.tensor(action, dtype=torch.long, device=action_logits.device)
            log_prob = dist.log_prob(action_t)

        entropy = dist.entropy()
        return action, log_prob, entropy, value, h_t

    def forward_batch(
        self,
        obs_batch,
        h_prev: torch.Tensor,
        prev_actions: torch.Tensor,
        prev_rewards: torch.Tensor,
    ):
        """
        Batch forward for PPO update.

        Parameters:
          obs_batch: PyG Batch object with x, edge_index, batch, ptr
          h_prev: (B, gru_hidden)
          prev_actions: (B,)
          prev_rewards: (B,)

        Returns:
          node_logits: (total_nodes,)
          terminate_logits: (B,)
          values: (B,)
          h_t: (B, gru_hidden)
        """
        device = self._get_device()
        x = obs_batch.x.to(device)
        edge_index = obs_batch.edge_index.to(device)
        batch = obs_batch.batch.to(device)
        ptr = obs_batch.ptr.to(device)

        node_emb, graph_emb = self.gat_encoder(x, edge_index, batch=batch)
        global_feat = x[:, 6:7]

        prev_actions = prev_actions.long().to(device)
        prev_rewards = prev_rewards.float().to(device)
        action_emb = self.action_embedding(prev_actions)
        h_t = self.history_gru(graph_emb, action_emb, prev_rewards, h_prev.to(device))

        B = h_t.size(0)
        h_t_expanded = h_t[batch]
        graph_emb_expanded = graph_emb[batch]

        actor_input = torch.cat([
            node_emb,
            graph_emb_expanded,
            global_feat,
            h_t_expanded,
        ], dim=-1)
        node_logits = self.actor_mlp(actor_input).squeeze(-1)

        if hasattr(obs_batch, 'node_mask'):
            node_mask = obs_batch.node_mask.to(device)
            node_logits = node_logits.masked_fill(~node_mask, float("-inf"))

        terminate_logits = self.terminate_head(h_t).squeeze(-1)
        values = self.critic_mlp(h_t).squeeze(-1)

        return node_logits, terminate_logits, values, h_t

    def init_hidden(self, device=None):
        """Initialize hidden state to zeros."""
        if device is None:
            device = self._get_device()
        return torch.zeros(self.gru_hidden, device=device)
