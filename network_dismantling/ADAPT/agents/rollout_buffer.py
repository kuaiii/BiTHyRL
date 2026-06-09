"""
On-policy Rollout Buffer for PPO.
Stores trajectories and computes GAE advantages.
"""
from typing import List, Dict, Any, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """
    Simple list-based rollout buffer for variable-size graph observations.
    """

    def __init__(self):
        self.observations: List[Dict[str, Any]] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.hiddens: List[np.ndarray] = []
        self.prev_actions: List[int] = []
        self.prev_rewards: List[float] = []

        self.advantages: np.ndarray = None
        self.returns: np.ndarray = None

    def add(
        self,
        obs: Dict[str, Any],
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        hidden: np.ndarray,
        prev_action: int,
        prev_reward: float,
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.hiddens.append(hidden)
        self.prev_actions.append(prev_action)
        self.prev_rewards.append(prev_reward)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.hiddens.clear()
        self.prev_actions.clear()
        self.prev_rewards.clear()
        self.advantages = None
        self.returns = None

    def compute_returns_and_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute GAE advantages and returns in-place.
        """
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_advantage = 0.0

        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # Bootstrap from last value if episode didn't end
        # For simplicity, we use 0 as bootstrap when done=True
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0 if dones[t] else 0.0  # no bootstrap from buffer
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = delta + gamma * gae_lambda * next_non_terminal * last_advantage
            last_advantage = advantages[t]

        self.advantages = advantages
        self.returns = advantages + values

    def get(self) -> Tuple:
        """
        Return all stored data.
        """
        return (
            self.observations,
            np.array(self.actions, dtype=np.int64),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
            np.stack(self.hiddens),  # (T, hidden_dim)
            np.array(self.prev_actions, dtype=np.int64),
            np.array(self.prev_rewards, dtype=np.float32),
            self.advantages,
            self.returns,
        )

    def __len__(self):
        return len(self.observations)
