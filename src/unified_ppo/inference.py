# -*- coding: utf-8 -*-
"""
统一 PPO 推理函数：对单图执行拓扑微调 + 控制器选择。
"""
import time
import math
import torch

from src.bit_hyrl import config
from src.unified_ppo.observation import GraphState, build_observation
from src.unified_ppo.action_space import CandidatePoolGenerator, build_topology_action_mask


def unified_solve(G, model, B_budget=3, k_ratio=0.1,
                  embed_dim=256, dre_dim=0, seed=42, deterministic=True):
    """
    使用训练好的 TopologyPolicy 对图 G 进行拓扑微调 + 控制器选择。

    Args:
        G: nx.Graph 原始图
        model: TopologyPolicy
        B_budget: 拓扑微调预算
        k_ratio: 控制器比例
        embed_dim: Node2Vec 维度
        dre_dim: DRE 维度
        seed: 随机种子
        deterministic: 是否确定性策略

    Returns:
        dict: {
            'G_final': 微调后的 nx.Graph,
            'controllers': 控制器节点列表,
            'time': 耗时,
        }
    """
    start = time.time()
    n = G.number_of_nodes()
    K_budget = max(1, math.ceil(n * k_ratio))

    model.eval()
    graph_state = GraphState(G, device=config.DEVICE)
    node_list = graph_state.node_list
    N = graph_state.N

    obs = build_observation(
        graph_state, node_features=None, selected_mask=None, phase=1,
        use_global_feat=False, embed_dim=embed_dim, dre_dim=dre_dim,
        seed=seed, use_cache=False
    )
    node_features = obs['x'].detach()

    pool_generator = CandidatePoolGenerator(M=model.M_candidates)
    selected_mask = torch.zeros(N, dtype=torch.bool, device=config.DEVICE)

    with torch.no_grad():
        # Phase 1
        for _ in range(B_budget):
            obs = build_observation(
                graph_state, node_features=node_features, selected_mask=selected_mask,
                phase=1, use_global_feat=False
            )
            edge_index = obs['edge_index']
            candidate_pool, _ = pool_generator.generate(graph_state, node_features[:, :embed_dim])
            if candidate_pool.numel() == 0:
                break
            action_mask = build_topology_action_mask(graph_state, candidate_pool)
            action, _, _, _ = model.get_action(
                node_features, edge_index, phase=1,
                candidate_pool=candidate_pool, action_mask=action_mask, deterministic=deterministic
            )
            action_idx = action.item()
            M = candidate_pool.size(0)
            if 0 <= action_idx < 2 * M:
                edge_idx = action_idx % M
                op = action_idx // M
                u, v = candidate_pool[edge_idx].cpu().numpy()
                u, v = int(u), int(v)
                if op == 0:
                    graph_state.add_edge(u, v)
                else:
                    graph_state.remove_edge(u, v)

        # Phase 2
        for _ in range(K_budget):
            obs = build_observation(
                graph_state, node_features=node_features, selected_mask=selected_mask,
                phase=2, use_global_feat=False
            )
            edge_index = obs['edge_index']
            action, _, _, _ = model.get_action(
                node_features, edge_index, phase=2,
                selected_mask=selected_mask, deterministic=deterministic
            )
            action_idx = action.item()
            if 0 <= action_idx < N:
                selected_mask = selected_mask.clone()
                selected_mask[action_idx] = True

    centers = [node_list[i] for i in torch.where(selected_mask)[0].cpu().numpy().tolist()]
    G_final = graph_state.get_current_nx_graph()
    elapsed = time.time() - start

    return {
        'G_final': G_final,
        'controllers': centers,
        'time': elapsed,
    }
