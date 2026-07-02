# -*- coding: utf-8 -*-
"""
统一 PPO 策略网络：共享 GNN 编码器 + 拓扑微调头 + 控制器选择头。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from src.bit_hyrl import config

DEVICE = config.DEVICE


class TopologyPolicy(nn.Module):
    """
    Phase-Aware 统一策略网络。

    Args:
        in_channels: 输入节点特征维度（Node2Vec + 节点状态 + 可能的全局特征）
        hidden_channels: GNN 隐藏维度
        heads: GAT 头数
        num_layers: GAT 层数
        M_candidates: Phase 1 候选边池大小
        dropout: dropout 比例
    """

    def __init__(self, in_channels=259, hidden_channels=128, heads=4, num_layers=3,
                 M_candidates=50, dropout=0.1):
        super(TopologyPolicy, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers
        self.M_candidates = M_candidates

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU()
        )

        # 共享 GNN 编码器
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.gat_projs = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        for i in range(num_layers):
            if i == num_layers - 1:
                self.gat_layers.append(GATConv(
                    hidden_channels, hidden_channels, heads=1, dropout=dropout, concat=False
                ))
                self.gat_projs.append(nn.Identity())
                self.layer_norms.append(nn.LayerNorm(hidden_channels))
                self.residual_projs.append(nn.Identity())
            else:
                self.gat_layers.append(GATConv(
                    hidden_channels, hidden_channels // heads, heads=heads,
                    dropout=dropout, concat=True
                ))
                actual_out = (hidden_channels // heads) * heads
                self.gat_projs.append(
                    nn.Linear(actual_out, hidden_channels) if actual_out != hidden_channels else nn.Identity()
                )
                self.layer_norms.append(nn.LayerNorm(hidden_channels))
                self.residual_projs.append(nn.Identity())

        # 跳跃连接聚合
        self.jump_proj = nn.Linear(hidden_channels * num_layers, hidden_channels)
        self.jump_norm = nn.LayerNorm(hidden_channels)

        # 阶段嵌入
        self.phase_embed = nn.Embedding(2, hidden_channels)

        # 已选控制器位置编码（Phase 2 使用）
        self.selected_embedding = nn.Parameter(torch.randn(1, hidden_channels))

        # 全局注意力池化
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )

        # ===== Phase 1: 拓扑微调头 =====
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        self.topo_op_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ELU(),
            nn.Linear(hidden_channels // 2, 2)
        )

        # ===== Phase 2: 控制器选择头 =====
        self.controller_head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Linear(hidden_channels // 2, 1)
        )

        # ===== Critic 头 =====
        self.critic = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Linear(hidden_channels // 2, 1)
        )

        # 温度参数
        self.temperature = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)

    def _encode(self, x, edge_index, selected_mask=None, phase=1):
        """共享编码器前向传播，返回节点表示 h 和全局表示 g_global。"""
        h = self.input_proj(x)
        layer_outputs = []

        for i, (gat, ln, gat_proj, res_proj) in enumerate(
            zip(self.gat_layers, self.layer_norms, self.gat_projs, self.residual_projs)
        ):
            residual = res_proj(h)
            h = gat(h, edge_index)
            h = gat_proj(h)
            h = ln(h)
            h = F.elu(h)
            h = h + residual
            layer_outputs.append(h)
            if i < self.num_layers - 1:
                h = self.dropout(h)

        # 跳跃连接
        jump_concat = torch.cat(layer_outputs, dim=-1)
        h = self.jump_proj(jump_concat)
        h = self.jump_norm(h)
        h = F.elu(h)

        # 阶段嵌入
        phase_idx = torch.tensor(0 if phase == 1 else 1, dtype=torch.long, device=h.device)
        phase_emb = self.phase_embed(phase_idx)
        h = h + phase_emb.unsqueeze(0)

        # Phase 2 已选位置编码
        if phase == 2 and selected_mask is not None and selected_mask.any():
            sel_emb = self.selected_embedding.expand(h.size(0), -1)
            h = h + selected_mask.float().unsqueeze(-1) * sel_emb

        # 全局上下文
        attention_weights = self.global_attention(h)
        attention_weights = F.softmax(attention_weights, dim=0)
        g_global = (attention_weights * h).sum(dim=0, keepdim=True)  # (1, hidden)

        return h, g_global

    def forward(self, x, edge_index, phase=1, candidate_pool=None,
                selected_mask=None, global_feat=None):
        """
        前向传播。

        Args:
            x: (N, F)
            edge_index: (2, E)
            phase: 1 or 2
            candidate_pool: (M, 2)，Phase 1 需要
            selected_mask: (N,)，Phase 2 需要
            global_feat: (F_g,)，可选全局特征（暂不使用，保持接口兼容）

        Returns:
            Phase 1: (probs, value, candidate_pool)
            Phase 2: (probs, value)
        """
        h, g_global = self._encode(x, edge_index, selected_mask=selected_mask, phase=phase)

        temp = torch.clamp(self.temperature, min=0.1, max=2.0)

        if phase == 1:
            if candidate_pool is None or candidate_pool.numel() == 0:
                # 无候选边时返回空概率
                return torch.zeros(0, device=x.device), self.critic(g_global).squeeze(-1), candidate_pool

            M = candidate_pool.size(0)
            h_u = h[candidate_pool[:, 0]]
            h_v = h[candidate_pool[:, 1]]
            h_uv = h_u * h_v
            edge_feat = torch.cat([h_u, h_v, h_uv], dim=-1)
            edge_scores = self.edge_scorer(edge_feat).squeeze(-1)  # (M,)

            # 全局操作类型 logits
            op_logits = self.topo_op_head(g_global.squeeze(0))  # (2,)

            # 联合 logits: (M, 2)
            combined_logits = edge_scores.unsqueeze(1) + op_logits.unsqueeze(0)
            combined_logits = combined_logits.view(-1) / temp  # (2M,)
            probs = F.softmax(combined_logits, dim=0)
            value = self.critic(g_global).squeeze(-1)
            return probs, value, candidate_pool

        else:
            # Phase 2: 控制器选择
            g_global_expanded = g_global.expand(h.size(0), -1)
            node_feat = torch.cat([h, g_global_expanded], dim=-1)
            scores = self.controller_head(node_feat).squeeze(-1) / temp

            if selected_mask is not None:
                scores = scores.masked_fill(selected_mask, -1e9)

            probs = F.softmax(scores, dim=0)
            value = self.critic(g_global).squeeze(-1)
            return probs, value

    def get_action(self, x, edge_index, phase=1, candidate_pool=None,
                   selected_mask=None, action_mask=None, deterministic=False):
        """
        采样/贪婪选择一个动作。

        Returns:
            action: 动作索引 scalar tensor
            log_prob: log probability
            value: 状态价值
            probs: 动作概率分布
        """
        if phase == 1:
            probs, value, _ = self.forward(
                x, edge_index, phase=phase, candidate_pool=candidate_pool
            )
        else:
            probs, value = self.forward(
                x, edge_index, phase=phase, selected_mask=selected_mask
            )

        if probs.numel() == 0:
            # 没有合法动作时返回 dummy
            return torch.tensor(-1, device=x.device), torch.tensor(0.0, device=x.device), value, probs

        if action_mask is not None:
            probs = probs.masked_fill(~action_mask, 0.0)
            # 重新归一化
            sum_probs = probs.sum()
            if sum_probs > 1e-9:
                probs = probs / sum_probs
            else:
                # 全部不合法时均匀分布在合法动作上
                probs = action_mask.float() / action_mask.sum().clamp(min=1)

        if deterministic:
            action = torch.argmax(probs)
            log_prob = torch.log(probs[action] + 1e-10)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value, probs

    def evaluate_actions(self, x, edge_index, actions, phases, candidate_pools=None,
                         selected_masks=None, action_masks=None):
        """
        PPO 更新时重新评估动作。

        Args:
            x: list of (N, F) 或 batched 特征（这里按经验逐个处理）
            edge_index: list
            actions: list/tensor of actions
            phases: list of int
            candidate_pools: list
            selected_masks: list
            action_masks: list

        Returns:
            log_probs, values, entropies
        """
        log_probs = []
        values = []
        entropies = []

        for i in range(len(actions)):
            phase = phases[i]
            if phase == 1:
                probs, value, _ = self.forward(
                    x[i], edge_index[i], phase=phase,
                    candidate_pool=candidate_pools[i] if candidate_pools else None
                )
            else:
                probs, value = self.forward(
                    x[i], edge_index[i], phase=phase,
                    selected_mask=selected_masks[i] if selected_masks else None
                )

            if action_masks is not None and action_masks[i] is not None:
                probs = probs.masked_fill(~action_masks[i], 0.0)
                probs = probs / probs.sum().clamp(min=1e-9)

            dist = torch.distributions.Categorical(probs)
            log_probs.append(dist.log_prob(actions[i]))
            values.append(value)
            entropies.append(dist.entropy())

        return torch.stack(log_probs), torch.stack(values), torch.stack(entropies)
