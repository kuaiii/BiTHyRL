"""
PPO Agent with GAE for ARCH.
"""
import math
from typing import List, Dict, Any, Optional

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, Batch

from network_dismantling.ADAPT.models.actor_critic import ActorCritic
from network_dismantling.ADAPT.envs.network_env import NetworkDismantleEnv
from network_dismantling.ADAPT.agents.rollout_buffer import RolloutBuffer


class PPOAgent:
    """
    Proximal Policy Optimization agent for network dismantling.
    """

    def __init__(
        self,
        model: ActorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        p_clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.model = model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.p_clip = p_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    @torch.no_grad()
    def collect_rollouts(
        self,
        env: NetworkDismantleEnv,
        graphs: List[nx.Graph],
        n_steps: int,
        device: torch.device,
        max_removals_ratio: float = 0.2,
    ) -> Dict[str, float]:
        """
        Collect rollouts from a list of graphs until buffer reaches n_steps.

        Returns
        -------
        stats : dict with episode returns and lengths.
        """
        self.model.eval()
        step_count = 0
        ep_returns = []
        ep_lengths = []

        while step_count < n_steps:
            # Sample a graph
            G = graphs[np.random.randint(len(graphs))]
            if G.number_of_nodes() == 0:
                continue

            obs = env.reset(G)
            h = self.model.init_hidden(device=device)
            prev_action_global = self.model.max_nodes + 1  # no previous action
            prev_reward = 0.0
            done = False
            ep_return = 0.0
            ep_len = 0
            max_removals = max(1, int(max_removals_ratio * G.number_of_nodes()))

            while not done and step_count < n_steps and ep_len < max_removals:
                action, log_prob, entropy, value, h_next = self.model.get_action_and_value(
                    obs, h, prev_action_global, prev_reward
                )
                action = int(action.item())
                log_prob_val = float(log_prob.item())
                value_val = float(value.item())

                obs_next, reward, done, info = env.step(action)

                self.buffer.add(
                    obs=obs,
                    action=action,
                    log_prob=log_prob_val,
                    reward=reward,
                    value=value_val,
                    done=done,
                    hidden=h.cpu().numpy(),
                    prev_action=prev_action_global if prev_action_global <= self.model.max_nodes else self.model.max_nodes,
                    prev_reward=prev_reward,
                )

                step_count += 1
                ep_len += 1
                ep_return += reward

                # Prepare for next step
                if not done:
                    node_map = obs["node_map"]
                    if action < len(node_map):
                        prev_action_global = node_map[action]
                    else:
                        prev_action_global = self.model.max_nodes  # terminate
                    prev_reward = reward
                    h = h_next
                    obs = obs_next

            ep_returns.append(ep_return)
            ep_lengths.append(ep_len)

        return {
            "mean_ep_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "mean_ep_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        }

    def update(
        self,
        update_epochs: int = 4,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Perform PPO update on the collected rollout buffer.

        Returns
        -------
        stats : dict with losses.
        """
        if len(self.buffer) == 0:
            return {}

        self.model.train()
        self.buffer.compute_returns_and_advantages(self.gamma, self.gae_lambda)

        (
            obs_list,
            actions_arr,
            old_log_probs_arr,
            _,
            _,
            _,
            hiddens_arr,
            prev_actions_arr,
            prev_rewards_arr,
            advantages_arr,
            returns_arr,
        ) = self.buffer.get()

        device = next(self.model.parameters()).device
        actions_t = torch.from_numpy(actions_arr).long().to(device)
        old_log_probs_t = torch.from_numpy(old_log_probs_arr).float().to(device)
        advantages_t = torch.from_numpy(advantages_arr).float().to(device)
        returns_t = torch.from_numpy(returns_arr).float().to(device)
        hiddens_t = torch.from_numpy(hiddens_arr).float().to(device)
        prev_actions_t = torch.from_numpy(prev_actions_arr).long().to(device)
        prev_rewards_t = torch.from_numpy(prev_rewards_arr).float().to(device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        T = len(obs_list)
        indices = np.arange(T)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        n_updates = 0

        for epoch in range(update_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                mb_idx = indices[start:end]

                mb_size = len(mb_idx)
                if mb_size == 0:
                    continue

                mb_obs = [obs_list[i] for i in mb_idx]
                data_list = []
                for obs in mb_obs:
                    data = Data(
                        x=obs["x"],
                        edge_index=obs["edge_index"],
                    )
                    data.num_nodes = obs["x"].size(0)
                    data_list.append(data)
                obs_batch = Batch.from_data_list(data_list)
                node_masks = [obs["action_mask"][:-1] for obs in mb_obs]
                obs_batch.node_mask = torch.cat(node_masks)
                ptr = obs_batch.ptr.to(device)

                mb_actions = actions_t[mb_idx]
                mb_h_prev = hiddens_t[mb_idx]
                mb_prev_actions = prev_actions_t[mb_idx]
                mb_prev_rewards = prev_rewards_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                node_logits, terminate_logits, values, _ = self.model.forward_batch(
                    obs_batch, mb_h_prev, mb_prev_actions, mb_prev_rewards
                )

                num_nodes = ptr[1:] - ptr[:-1]
                node_lse = torch.stack([
                    node_logits[ptr[i]:ptr[i+1]].logsumexp(0)
                    for i in range(mb_size)
                ])
                full_lse = torch.logaddexp(node_lse, terminate_logits)

                is_terminate = mb_actions == num_nodes
                sample_logits = torch.zeros(mb_size, device=device)
                if (~is_terminate).any():
                    sample_logits[~is_terminate] = node_logits[
                        ptr[:-1][~is_terminate] + mb_actions[~is_terminate]
                    ]
                if is_terminate.any():
                    sample_logits[is_terminate] = terminate_logits[is_terminate]

                log_probs = sample_logits - full_lse

                entropies = torch.zeros(mb_size, device=device)
                for i in range(mb_size):
                    g_logits = torch.cat([
                        node_logits[ptr[i]:ptr[i+1]],
                        terminate_logits[i].unsqueeze(0)
                    ])
                    probs = F.softmax(g_logits, dim=0)
                    entropies[i] = -(probs * torch.log(probs + 1e-10)).sum()

                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.p_clip, 1.0 + self.p_clip) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_returns)
                entropy_loss = entropies.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1

        self.buffer.clear()

        if n_updates == 0:
            return {}

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy_loss / n_updates,
        }
