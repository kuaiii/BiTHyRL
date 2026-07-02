# -*- coding: utf-8 -*-
"""
统一 PPO 训练器：收集 B+K 两阶段轨迹并执行 PPO 更新。
"""
from collections import namedtuple
import random
import numpy as np
import torch
import torch.nn.functional as F

from src.bit_hyrl import config
from src.unified_ppo.observation import GraphState, build_observation
from src.unified_ppo.action_space import CandidatePoolGenerator, build_topology_action_mask
from src.unified_ppo.reward_unified import calculate_unified_reward

DEVICE = config.DEVICE


UnifiedExperience = namedtuple('UnifiedExperience', [
    'x',              # 节点特征
    'edge_index',     # 当前边索引
    'action',         # 动作索引
    'log_prob',       # 旧 log_prob
    'value',          # 旧 value
    'reward',         # 奖励
    'selected_mask',  # 已选控制器 mask
    'done',           # 是否完成
    'phase',          # 1 or 2
    'candidate_pool', # Phase 1 候选边池
    'action_mask',    # Phase 1 动作掩码
])


class UnifiedRolloutBuffer:
    """统一轨迹缓冲区。"""

    def __init__(self):
        self.experiences = []
        self.episode_rewards = []

    def add(self, exp):
        self.experiences.append(exp)

    def add_episode_reward(self, reward):
        self.episode_rewards.append(reward)

    def clear(self):
        self.experiences = []
        self.episode_rewards = []

    def __len__(self):
        return len(self.experiences)


class UnifiedPPOTrainer:
    """
    统一 PPO 训练器。

    Args:
        model: TopologyPolicy
        lr: 学习率
        gamma: 折扣因子
        gae_lambda: GAE lambda
        clip_eps: PPO clip
        value_coef: value loss 系数
        entropy_coef: 熵正则系数
        max_grad_norm: 梯度裁剪
        n_epochs: 每次更新 epoch 数
        batch_size: mini-batch 大小
        topology_lr_ratio: Phase 1 topology head 学习率相对选择头的比例
    """

    def __init__(self, model, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, n_epochs=4, batch_size=64,
                 topology_lr_ratio=0.3):
        self.model = model.to(DEVICE)
        self.lr = lr
        self.topology_lr_ratio = topology_lr_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.buffer = UnifiedRolloutBuffer()
        self.pool_generator = CandidatePoolGenerator(M=model.M_candidates)

        # 默认优化全部参数
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.phase2_only = False

    def set_phase2_warmup(self, enabled=True):
        """启用/关闭 Phase 2 预热：冻结 topology head。"""
        self.phase2_only = enabled
        if enabled:
            # 冻结 Phase 1 相关参数
            for p in self.model.edge_scorer.parameters():
                p.requires_grad = False
            for p in self.model.topo_op_head.parameters():
                p.requires_grad = False
            # Phase 2 与共享编码器保持可训练
            for p in self.model.controller_head.parameters():
                p.requires_grad = True
            for p in self.model.critic.parameters():
                p.requires_grad = True
            # 共享编码器可训练
            for p in self.model.input_proj.parameters():
                p.requires_grad = True
            for p in self.model.gat_layers.parameters():
                p.requires_grad = True
            for p in self.model.jump_proj.parameters():
                p.requires_grad = True
        else:
            for p in self.model.parameters():
                p.requires_grad = True

    def _build_optimizer(self):
        """根据阶段构建优化器（可分别设置学习率）。"""
        if self.phase2_only:
            # 只优化共享编码器 + Phase 2 head + critic
            params = [
                {'params': list(self.model.input_proj.parameters())
                 + list(self.model.gat_layers.parameters())
                 + list(self.model.layer_norms.parameters())
                 + list(self.model.gat_projs.parameters())
                 + list(self.model.residual_projs.parameters())
                 + list(self.model.jump_proj.parameters())
                 + list(self.model.phase_embed.parameters())
                 + list(self.model.global_attention.parameters())
                 + list(self.model.controller_head.parameters())
                 + list(self.model.critic.parameters())
                 + [self.model.temperature, self.model.selected_embedding], 'lr': self.lr},
            ]
        else:
            # 联合训练：共享参数使用 lr，topology head 使用较低 lr
            shared_params = list(self.model.input_proj.parameters()) \
                + list(self.model.gat_layers.parameters()) \
                + list(self.model.layer_norms.parameters()) \
                + list(self.model.gat_projs.parameters()) \
                + list(self.model.residual_projs.parameters()) \
                + list(self.model.jump_proj.parameters()) \
                + list(self.model.phase_embed.parameters()) \
                + list(self.model.global_attention.parameters()) \
                + list(self.model.controller_head.parameters()) \
                + list(self.model.critic.parameters()) \
                + [self.model.temperature, self.model.selected_embedding]
            topo_params = list(self.model.edge_scorer.parameters()) \
                + list(self.model.topo_op_head.parameters())
            params = [
                {'params': shared_params, 'lr': self.lr},
                {'params': topo_params, 'lr': self.lr * self.topology_lr_ratio},
            ]
        return torch.optim.Adam(params)

    def collect_unified_trajectory(self, G, B_budget, K_budget, reward_fn=None,
                                   embed_dim=256, dre_dim=0, seed=42,
                                   walk_length=20, num_walks=100, use_cache=True,
                                   epoch=None, total_epochs=None):
        """
        收集单张图的两阶段统一轨迹。

        Args:
            G: 原始 nx.Graph
            B_budget: 拓扑微调预算
            K_budget: 控制器选择数量
            reward_fn: 统一奖励函数，None 时使用默认 calculate_unified_reward
            ... 特征提取参数

        Returns:
            controllers: 最终控制器列表
            final_reward: 最终奖励
        """
        self.model.eval()

        if reward_fn is None:
            reward_fn = calculate_unified_reward

        graph_state = GraphState(G, device=DEVICE)
        node_list = graph_state.node_list
        N = graph_state.N

        # Phase 1 开始时计算一次 Node2Vec 特征（后续复用）
        obs = build_observation(
            graph_state, node_features=None, selected_mask=None, phase=1,
            use_global_feat=False, embed_dim=embed_dim, dre_dim=dre_dim,
            seed=seed, walk_length=walk_length, num_walks=num_walks, use_cache=use_cache
        )
        node_features = obs['x'].detach()

        experiences = []
        selected_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        controllers = []

        with torch.no_grad():
            # ========== Phase 1: 拓扑微调 ==========
            for step in range(B_budget):
                obs = build_observation(
                    graph_state, node_features=node_features, selected_mask=selected_mask,
                    phase=1, use_global_feat=False
                )
                edge_index = obs['edge_index']

                candidate_pool, _ = self.pool_generator.generate(graph_state, node_features[:, :embed_dim])
                action_mask = build_topology_action_mask(graph_state, candidate_pool)

                action, log_prob, value, probs = self.model.get_action(
                    node_features, edge_index, phase=1,
                    candidate_pool=candidate_pool, action_mask=action_mask, deterministic=False
                )

                action_idx = action.item()
                M = candidate_pool.size(0)

                if M > 0 and 0 <= action_idx < 2 * M:
                    edge_idx = action_idx % M
                    op = action_idx // M  # 0=add, 1=remove
                    u, v = candidate_pool[edge_idx].cpu().numpy()
                    u, v = int(u), int(v)
                    if op == 0:
                        graph_state.add_edge(u, v)
                    else:
                        graph_state.remove_edge(u, v)

                # Phase 1 中间奖励为 0（稀疏），或微小结构奖励
                step_reward = 0.0
                done = False
                exp = UnifiedExperience(
                    x=node_features.clone(),
                    edge_index=edge_index.clone(),
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=step_reward,
                    selected_mask=selected_mask.clone(),
                    done=done,
                    phase=1,
                    candidate_pool=candidate_pool.clone() if candidate_pool.numel() > 0 else None,
                    action_mask=action_mask.clone() if action_mask.numel() > 0 else None,
                )
                experiences.append(exp)

            # ========== Phase 2: 控制器选择 ==========
            for step in range(K_budget):
                obs = build_observation(
                    graph_state, node_features=node_features, selected_mask=selected_mask,
                    phase=2, use_global_feat=False
                )
                edge_index = obs['edge_index']

                action, log_prob, value, probs = self.model.get_action(
                    node_features, edge_index, phase=2,
                    selected_mask=selected_mask, deterministic=False
                )

                action_idx = action.item()
                if 0 <= action_idx < N:
                    center = node_list[action_idx]
                    controllers.append(center)
                    selected_mask = selected_mask.clone()
                    selected_mask[action_idx] = True

                if step == K_budget - 1:
                    # final reward will be computed after Phase 2; use placeholder here
                    step_reward = 0.01
                else:
                    step_reward = 0.01

                done = (step == K_budget - 1)
                exp = UnifiedExperience(
                    x=node_features.clone(),
                    edge_index=edge_index.clone(),
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=step_reward,
                    selected_mask=selected_mask.clone(),
                    done=done,
                    phase=2,
                    candidate_pool=None,
                    action_mask=None,
                )
                experiences.append(exp)

        # 最终奖励并修正最后一步经验
        G_final = graph_state.get_current_nx_graph()
        final_reward = reward_fn(G_final, controllers, G_original=G,
                                 epoch=epoch, total_epochs=total_epochs)

        if experiences:
            last = experiences[-1]
            experiences[-1] = UnifiedExperience(
                x=last.x, edge_index=last.edge_index, action=last.action,
                log_prob=last.log_prob, value=last.value, reward=final_reward,
                selected_mask=last.selected_mask, done=True, phase=last.phase,
                candidate_pool=last.candidate_pool, action_mask=last.action_mask
            )

        for exp in experiences:
            self.buffer.add(exp)
        self.buffer.add_episode_reward(final_reward)

        return controllers, final_reward

    def compute_gae(self, rewards, values, dones):
        """计算 GAE。"""
        advantages = []
        returns = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1] if not dones[t] else 0
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        return advantages, returns

    def update(self):
        """执行 PPO 更新。"""
        if len(self.buffer) == 0:
            return {}

        self.model.train()

        experiences = self.buffer.experiences
        rewards = [exp.reward for exp in experiences]
        values = [exp.value.item() for exp in experiences]
        dones = [exp.done for exp in experiences]

        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_log_probs = torch.stack([exp.log_prob for exp in experiences])

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        indices = list(range(len(experiences)))

        for epoch in range(self.n_epochs):
            random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                end = min(start + self.batch_size, len(indices))
                batch_indices = indices[start:end]

                batch_policy_loss = 0
                batch_value_loss = 0
                batch_entropy = 0

                for idx in batch_indices:
                    exp = experiences[idx]
                    if exp.phase == 1:
                        if exp.candidate_pool is None or exp.candidate_pool.numel() == 0:
                            continue
                        probs, value, _ = self.model(
                            exp.x, exp.edge_index, phase=1,
                            candidate_pool=exp.candidate_pool
                        )
                        if exp.action_mask is not None:
                            probs = probs.masked_fill(~exp.action_mask, 0.0)
                            probs = probs / probs.sum().clamp(min=1e-9)
                    else:
                        probs, value = self.model(
                            exp.x, exp.edge_index, phase=2,
                            selected_mask=exp.selected_mask
                        )

                    if probs.numel() == 0:
                        continue

                    dist = torch.distributions.Categorical(probs)
                    new_log_prob = dist.log_prob(exp.action)
                    entropy = dist.entropy()

                    ratio = torch.exp(new_log_prob - old_log_probs[idx].detach())
                    adv = advantages[idx]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2)

                    value_loss = F.mse_loss(value.squeeze(), returns[idx])

                    batch_policy_loss += policy_loss
                    batch_value_loss += value_loss
                    batch_entropy += entropy

                bs = len(batch_indices)
                if bs == 0:
                    continue
                batch_policy_loss /= bs
                batch_value_loss /= bs
                batch_entropy /= bs

                loss = batch_policy_loss + self.value_coef * batch_value_loss - self.entropy_coef * batch_entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += batch_policy_loss.item()
                total_value_loss += batch_value_loss.item()
                total_entropy += batch_entropy.item()
                num_updates += 1

        avg_reward = np.mean(self.buffer.episode_rewards) if self.buffer.episode_rewards else 0
        self.buffer.clear()

        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
            'avg_reward': avg_reward,
        }
