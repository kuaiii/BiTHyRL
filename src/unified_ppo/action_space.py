# -*- coding: utf-8 -*-
"""
统一 PPO 拓扑动作空间：候选边池 + 动作掩码。

Phase 1 动作：在候选边池上选择 (edge_idx, operation)，
其中 operation=0 表示添加，operation=1 表示删除。
总动作空间大小为 2 * M_candidates。
"""
import itertools
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F

from src.bit_hyrl import config

DEVICE = config.DEVICE


class CandidatePoolGenerator:
    """
    基于当前图状态和节点嵌入生成 Top-M 候选边池。

    设计原则：
    - 优先 hub-hub 与 leaf-hub 边（增强双峰结构）。
    - 过滤 leaf-leaf 的新增边（避免 leaf 之间直连）。
    - 保留可删除的 leaf-leaf / leaf-hub 原始边。
    - 使用节点嵌入内积 + 结构偏置进行评分。
    - v5 新增：引入 betweenness-aware 候选，帮助降低最大边 betweenness。
    """

    def __init__(self, M=50, target_hub_ratio=0.15, max_leaf_degree=3,
                 use_betweenness_aware=False, top_betweenness_edges=10):
        self.M = M
        self.target_hub_ratio = target_hub_ratio
        self.max_leaf_degree = max_leaf_degree
        self.use_betweenness_aware = use_betweenness_aware
        self.top_betweenness_edges = top_betweenness_edges

    def _get_hub_mask(self, graph_state):
        """基于当前度数返回 hub 节点掩码 (N,)。"""
        deg = graph_state.get_node_degrees().float().cpu().numpy()
        N = len(deg)
        if N == 0:
            return torch.zeros(N, dtype=torch.bool, device=DEVICE)
        hub_count = max(1, int(N * self.target_hub_ratio))
        threshold = sorted(deg, reverse=True)[hub_count - 1] if hub_count <= N else deg.min()
        return torch.tensor(deg >= threshold, dtype=torch.bool, device=DEVICE)

    def _betweenness_aware_candidates(self, graph_state, hub_mask, sim):
        """
        v5：生成能降低边 betweenness 的候选。
        对高 betweenness 边的端点，添加连接其邻居/嵌入相似节点的候选，
        并添加删除该高 betweenness 边的候选。
        """
        candidates = []
        scores = []
        N = graph_state.N
        if N < 3:
            return candidates, scores

        try:
            G_current = graph_state.get_current_nx_graph()
            if G_current.number_of_edges() == 0:
                return candidates, scores
            n = G_current.number_of_nodes()
            # 采样近似，避免大图过慢
            k = min(n, max(10, int(n * 0.05)))
            eb = nx.edge_betweenness_centrality(G_current, k=k)
            sorted_edges = sorted(eb.items(), key=lambda x: x[1], reverse=True)
            top_k = min(self.top_betweenness_edges, len(sorted_edges))
            high_eb_pairs = set()
            for (u_node, v_node), _ in sorted_edges[:top_k]:
                if u_node in graph_state.node_to_idx and v_node in graph_state.node_to_idx:
                    u = graph_state.node_to_idx[u_node]
                    v = graph_state.node_to_idx[v_node]
                    high_eb_pairs.add((min(u, v), max(u, v)))

            # 构建当前邻接表（基于 node index）
            adj = [set() for _ in range(N)]
            for a, b in G_current.edges():
                if a in graph_state.node_to_idx and b in graph_state.node_to_idx:
                    ia = graph_state.node_to_idx[a]
                    ib = graph_state.node_to_idx[b]
                    adj[ia].add(ib)
                    adj[ib].add(ia)

            for u, v in high_eb_pairs:
                # 删除候选（动作掩码会检查桥边）
                candidates.append((u, v, 1))
                scores.append(0.6)  # 较高优先级删除高 betweenness 边

                # 添加候选：为 u, v 各自寻找能分散流量的节点
                for center in (u, v):
                    # 嵌入最相似的 top-5 节点（排除自己和已有邻居）
                    sims = sim[center].cpu().numpy().copy()
                    sims[center] = -1
                    for w in adj[center]:
                        sims[w] = -1
                    top_neighbors = np.argsort(sims)[::-1][:5]
                    for w in top_neighbors:
                        if sims[w] <= 0:
                            continue
                        candidates.append((min(center, int(w)), max(center, int(w)), 0))
                        scores.append(0.4 + 0.4 * sims[w])
        except Exception:
            pass
        return candidates, scores

    def generate(self, graph_state, node_embed):
        """
        生成候选边池。

        Args:
            graph_state: GraphState
            node_embed: (N, F) 节点嵌入张量

        Returns:
            candidate_pool: (M, 2) long 候选边端点索引
            base_scores: (M,) float 候选边基础分数
        """
        N = graph_state.N
        if N < 2:
            return torch.zeros((0, 2), dtype=torch.long, device=DEVICE), torch.zeros(0, device=DEVICE)

        hub_mask = self._get_hub_mask(graph_state)
        hub_nodes = torch.where(hub_mask)[0].cpu().numpy().tolist()
        leaf_nodes = torch.where(~hub_mask)[0].cpu().numpy().tolist()
        deg = graph_state.get_node_degrees().cpu().numpy()

        # 节点嵌入内积
        node_embed = F.normalize(node_embed, p=2, dim=1)
        sim = torch.matmul(node_embed, node_embed.t())  # (N, N)

        candidates = []
        scores = []

        def _add_candidate(u, v, candidate_type):
            u, v = int(u), int(v)
            if u == v:
                return
            exists = graph_state.edge_exists(u, v)
            # 添加候选：不存在的边
            # 删除候选：存在的边
            # 但为了控制池大小，我们分别收集再统一排序
            # 这里为每条无序边生成两个操作候选
            candidates.append((u, v, 0))  # add
            candidates.append((u, v, 1))  # remove
            # 结构偏置
            is_hub_u = hub_mask[u].item()
            is_hub_v = hub_mask[v].item()
            struct_bias = 0.0
            if is_hub_u and is_hub_v:
                struct_bias += 0.5
            elif is_hub_u or is_hub_v:
                struct_bias += 0.2
            else:
                struct_bias -= 0.3  # leaf-leaf 不鼓励

            # 操作偏置：倾向于添加不存在的边，删除 leaf-leaf 边
            op_bias_add = 0.1 if not exists else -0.5
            op_bias_remove = 0.1 if (exists and not is_hub_u and not is_hub_v) else -0.2

            base_score = sim[u, v].item() + struct_bias
            scores.append(base_score + op_bias_add)
            scores.append(base_score + op_bias_remove)

        # 1. hub-hub 全对（通常 hub 数量少）
        for u, v in itertools.combinations(hub_nodes, 2):
            _add_candidate(u, v, 'hub_hub')

        # 2. leaf-hub：每个 leaf 与 top-5 最近 hub
        if node_embed.size(0) > 0 and len(hub_nodes) > 0:
            for leaf in leaf_nodes:
                leaf_emb = node_embed[leaf].unsqueeze(0)
                hub_embed = node_embed[hub_nodes]
                hub_sims = torch.matmul(leaf_emb, hub_embed.t()).squeeze(0)
                topk = min(5, len(hub_nodes))
                top_hubs_idx = torch.topk(hub_sims, topk).indices.cpu().numpy()
                for idx in top_hubs_idx:
                    hub = hub_nodes[idx]
                    _add_candidate(leaf, hub, 'leaf_hub')

        # 3. 原始图中存在的 leaf-leaf 边作为可删除候选
        # 已在上面组合中覆盖；但若 leaf-leaf 对数少，确保保留
        # 补充：原始 leaf-leaf 边
        if graph_state.E_orig.numel() > 0:
            orig_pairs = set()
            orig_t = graph_state.E_orig.t().cpu().numpy()
            for a, b in orig_t:
                key = (min(int(a), int(b)), max(int(a), int(b)))
                orig_pairs.add(key)
            for u, v in orig_pairs:
                if u in leaf_nodes and v in leaf_nodes:
                    if not any(c[0] == u and c[1] == v and c[2] == 1 for c in candidates):
                        _add_candidate(u, v, 'leaf_leaf_existing')

        # v5：加入 betweenness-aware 候选
        if self.use_betweenness_aware:
            btw_candidates, btw_scores = self._betweenness_aware_candidates(graph_state, hub_mask, sim)
            candidates.extend(btw_candidates)
            scores.extend(btw_scores)

        if len(candidates) == 0:
            return torch.zeros((0, 2), dtype=torch.long, device=DEVICE), torch.zeros(0, device=DEVICE)

        # 按分数排序取 Top-M
        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=DEVICE)
        # 去重：同一条边的同一操作只保留一次
        seen = set()
        unique_indices = []
        for i, (u, v, op) in enumerate(candidates):
            key = (min(u, v), max(u, v), op)
            if key not in seen:
                seen.add(key)
                unique_indices.append(i)

        candidates = [candidates[i] for i in unique_indices]
        scores_tensor = scores_tensor[unique_indices]

        M = min(self.M, len(candidates))
        topk = torch.topk(scores_tensor, M)
        selected = [candidates[i] for i in topk.indices.cpu().numpy()]

        # candidate_pool 只保留边端点 (M, 2)
        pool = torch.tensor([[c[0], c[1]] for c in selected], dtype=torch.long, device=DEVICE)
        base_scores = topk.values
        return pool, base_scores


def build_topology_action_mask(graph_state, candidate_pool, max_leaf_degree=3):
    """
    构建 Phase 1 动作掩码。

    动作空间大小 = 2 * M，其中：
    - action 0..M-1：对 candidate_pool[i] 执行 add
    - action M..2M-1：对 candidate_pool[i] 执行 remove

    非法动作：
    - add 已存在的边
    - remove 不存在的边
    - remove 导致图不连通的边（这里只检查是否桥边；实际由 GraphState.remove_edge 二次校验）
    - add 使 leaf 度数超过 max_leaf_degree（hub 除外）

    Args:
        graph_state: GraphState
        candidate_pool: (M, 2)
        max_leaf_degree: leaf 最大度数

    Returns:
        mask: (2*M,) bool，True 表示合法
    """
    if candidate_pool.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=DEVICE)

    M = candidate_pool.size(0)
    mask = torch.ones(2 * M, dtype=torch.bool, device=DEVICE)

    deg = graph_state.get_node_degrees().cpu().numpy()
    hub_count = max(1, int(graph_state.N * 0.15))
    sorted_deg = sorted(deg, reverse=True)
    hub_threshold = sorted_deg[hub_count - 1] if hub_count <= len(deg) else deg.min()

    # 预计算当前图是否为桥边需要 nx 图
    G_current = graph_state.get_current_nx_graph()

    for i in range(M):
        u, v = candidate_pool[i].cpu().numpy()
        u, v = int(u), int(v)
        exists = graph_state.edge_exists(u, v)

        # add: action i
        if exists:
            mask[i] = False
        else:
            # leaf 度数检查
            for node in (u, v):
                if deg[node] < hub_threshold and deg[node] + 1 > max_leaf_degree:
                    mask[i] = False
                    break

        # remove: action i + M
        if not exists:
            mask[i + M] = False
        else:
            # 桥边检查
            u_node = graph_state.node_list[u]
            v_node = graph_state.node_list[v]
            if G_current.has_edge(u_node, v_node):
                G_current.remove_edge(u_node, v_node)
                if not nx.is_connected(G_current):
                    mask[i + M] = False
                G_current.add_edge(u_node, v_node)

    return mask
