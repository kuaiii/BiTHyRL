# -*- coding: utf-8 -*-
"""
统一 PPO 观测空间：动态图状态 + 节点/全局特征。
"""
import numpy as np
import networkx as nx
import torch

from src.bit_hyrl import config
from src.bit_hyrl.gnn_model import get_gnn_node_features, graph_to_pyg_data

DEVICE = config.DEVICE


class GraphState:
    """
    维护动态图状态，避免每一步都重建 nx.Graph。

    通过 edge_mask 标记原始边是否被删除，通过 added_edges 记录新增的边。
    所有内部张量都位于同一 device。
    """

    def __init__(self, G_original, device=None):
        if device is None:
            device = DEVICE
        self.device = device
        self.G_orig = G_original
        self.node_list = list(G_original.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.node_list)}
        self.N = len(self.node_list)

        edges = list(G_original.edges())
        if len(edges) > 0:
            edge_index = []
            for u, v in edges:
                edge_index.append([self.node_to_idx[u], self.node_to_idx[v]])
                edge_index.append([self.node_to_idx[v], self.node_to_idx[u]])
            self.E_orig = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        else:
            self.E_orig = torch.zeros((2, 0), dtype=torch.long, device=device)

        self.edge_mask = torch.ones(self.E_orig.size(1), dtype=torch.bool, device=device)
        self.added_edges = torch.zeros((0, 2), dtype=torch.long, device=device)
        self.removed_edges = set()

    def _edge_key(self, u, v):
        u, v = int(u), int(v)
        return (min(u, v), max(u, v))

    def get_current_nx_graph(self):
        """基于当前状态重建 nx.Graph（仅在需要时使用，尽量复用张量表示）。"""
        G = self.G_orig.__class__()
        G.add_nodes_from(self.node_list)
        # 原始边中未被删除的
        active_orig = self.E_orig[:, self.edge_mask].t().cpu().numpy()
        edges = set()
        for u_idx, v_idx in active_orig:
            u, v = self.node_list[u_idx], self.node_list[v_idx]
            edges.add(self._edge_key(u, v))
        # 新增边
        if self.added_edges.numel() > 0:
            added = self.added_edges.cpu().numpy()
            for u_idx, v_idx in added:
                u, v = self.node_list[u_idx], self.node_list[v_idx]
                edges.add(self._edge_key(u, v))
        G.add_edges_from(edges)
        return G

    def get_current_edge_index(self):
        """返回当前有效边的 PyG edge_index（含双向边）。"""
        active_orig = self.E_orig[:, self.edge_mask]
        if self.added_edges.numel() > 0:
            # 新增边也需要双向
            added = self.added_edges.t().contiguous()
            added_both = torch.cat([added, added.flip(0)], dim=1)
            return torch.cat([active_orig, added_both], dim=1)
        return active_orig

    def edge_exists(self, u_idx, v_idx):
        """检查当前状态下边 (u_idx, v_idx) 是否存在。"""
        key = self._edge_key(u_idx, v_idx)
        # 检查新增边
        if self.added_edges.numel() > 0:
            added = self.added_edges.cpu().numpy()
            for a, b in added:
                if self._edge_key(a, b) == key:
                    return True
        # 检查原始边是否未被删除
        if key in self.removed_edges:
            return False
        # 在原始边中查找
        orig_edges = self.E_orig.t().cpu().numpy()
        for a, b in orig_edges:
            if self._edge_key(a, b) == key:
                return True
        return False

    def add_edge(self, u_idx, v_idx):
        """添加一条边；若已存在则不操作。返回是否成功。"""
        u_idx, v_idx = int(u_idx), int(v_idx)
        if u_idx == v_idx:
            return False
        if self.edge_exists(u_idx, v_idx):
            return False
        new_edge = torch.tensor([[u_idx, v_idx]], dtype=torch.long, device=self.device)
        self.added_edges = torch.cat([self.added_edges, new_edge], dim=0)
        return True

    def remove_edge(self, u_idx, v_idx):
        """
        删除一条边。若删除会导致图不连通，则拒绝。
        返回 (success, is_original_edge)。
        """
        u_idx, v_idx = int(u_idx), int(v_idx)
        if not self.edge_exists(u_idx, v_idx):
            return False, False

        # 先尝试删除并检查连通性
        key = self._edge_key(u_idx, v_idx)
        was_added = False
        if self.added_edges.numel() > 0:
            added_list = self.added_edges.cpu().numpy().tolist()
            for i, (a, b) in enumerate(added_list):
                if self._edge_key(a, b) == key:
                    was_added = True
                    # 从 added_edges 中移除
                    added_list.pop(i)
                    if added_list:
                        self.added_edges = torch.tensor(added_list, dtype=torch.long, device=self.device)
                    else:
                        self.added_edges = torch.zeros((0, 2), dtype=torch.long, device=self.device)
                    break

        if not was_added:
            # 标记原始边为删除
            self.removed_edges.add(key)
            # 更新 edge_mask：找到对应的原始边索引并置 False
            # 注意 E_orig 包含双向边
            keep = []
            orig_t = self.E_orig.t()
            for i in range(orig_t.size(0)):
                a, b = orig_t[i].cpu().numpy()
                keep.append(self._edge_key(a, b) != key)
            self.edge_mask = torch.tensor(keep, dtype=torch.bool, device=self.device)

        # 连通性检查（基于当前有效边）
        G_tmp = self.get_current_nx_graph()
        if not nx.is_connected(G_tmp):
            # 恢复：重新加回这条边
            if was_added:
                new_edge = torch.tensor([[u_idx, v_idx]], dtype=torch.long, device=self.device)
                self.added_edges = torch.cat([self.added_edges, new_edge], dim=0)
            else:
                self.removed_edges.discard(key)
                # 恢复 edge_mask
                keep = []
                orig_t = self.E_orig.t()
                for i in range(orig_t.size(0)):
                    a, b = orig_t[i].cpu().numpy()
                    keep.append(self._edge_key(a, b) != key or key not in self.removed_edges)
                self.edge_mask = torch.tensor(keep, dtype=torch.bool, device=self.device)
            return False, not was_added

        return True, not was_added

    def get_node_degrees(self):
        """返回当前图中每个节点的度数张量。"""
        edge_index = self.get_current_edge_index()
        if edge_index.numel() == 0:
            return torch.zeros(self.N, dtype=torch.long, device=self.device)
        deg = torch.zeros(self.N, dtype=torch.long, device=self.device)
        src = edge_index[0]
        deg.index_add_(0, src, torch.ones(src.size(0), dtype=torch.long, device=self.device))
        return deg

    def clone(self):
        """深拷贝当前状态。"""
        new_state = GraphState(self.G_orig.copy(), device=self.device)
        new_state.edge_mask = self.edge_mask.clone()
        new_state.added_edges = self.added_edges.clone()
        new_state.removed_edges = self.removed_edges.copy()
        return new_state


def _compute_global_features(G):
    """计算全局图特征张量。"""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n <= 1:
        return torch.zeros(4, dtype=torch.float32, device=DEVICE)

    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0
    avg_degree = 2.0 * m / n
    try:
        diameter = nx.diameter(G) if nx.is_connected(G) else n
    except Exception:
        diameter = n

    # 代数连通性（Fiedler 值）
    try:
        if nx.is_connected(G) and n > 2:
            L = nx.laplacian_matrix(G).astype(np.float32)
            eigvals = np.linalg.eigvalsh(L.toarray())
            algebraic_conn = sorted(eigvals)[1] if len(eigvals) > 1 else 0.0
        else:
            algebraic_conn = 0.0
    except Exception:
        algebraic_conn = 0.0

    feats = np.array([n, density, avg_degree, diameter, algebraic_conn], dtype=np.float32)
    return torch.from_numpy(feats).to(DEVICE)


def build_observation(graph_state, node_features=None, selected_mask=None,
                      phase=1, use_global_feat=True,
                      embed_dim=256, dre_dim=0, seed=42,
                      walk_length=20, num_walks=100, use_cache=True):
    """
    构建统一观测。

    Args:
        graph_state: GraphState
        node_features: 预计算节点特征 (N, F)，为 None 时现场计算
        selected_mask: (N,) bool，已选控制器掩码
        phase: 1=拓扑微调，2=控制器选择
        use_global_feat: 是否拼接全局特征
        embed_dim/dre_dim/seed/walk_length/num_walks/use_cache: 传给 get_gnn_node_features

    Returns:
        dict: {
            'x': (N, F_total),
            'edge_index': (2, E_current),
            'node_list': list,
            'selected_mask': (N,) bool,
            'global_feat': (F_g,) or None,
            'phase': int,
        }
    """
    G_current = graph_state.get_current_nx_graph()
    node_list = graph_state.node_list
    N = len(node_list)

    if node_features is None:
        x, _ = get_gnn_node_features(
            G_current, device=DEVICE, embed_dim=embed_dim, dre_dim=dre_dim, seed=seed,
            walk_length=walk_length, num_walks=num_walks, use_cache=use_cache
        )
    else:
        x = node_features.to(DEVICE)

    # 节点状态特征 [is_hub, is_controller, is_modified]
    deg = graph_state.get_node_degrees().float()
    # hub 判定：度数处于 top 15% 或超过平均度 + std
    sorted_deg, _ = torch.sort(deg, descending=True)
    hub_threshold = sorted_deg[max(1, int(N * 0.15)) - 1].item() if N > 0 else 0.0
    is_hub = (deg >= hub_threshold).float().unsqueeze(1)

    if selected_mask is None:
        selected_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    is_controller = selected_mask.float().unsqueeze(1)

    # is_modified: 当前图与原始图相比节点度数发生变化的节点
    orig_deg = torch.tensor([graph_state.G_orig.degree(n) for n in node_list],
                            dtype=torch.float32, device=DEVICE)
    is_modified = ((deg - orig_deg).abs() > 1e-6).float().unsqueeze(1)

    node_state = torch.cat([is_hub, is_controller, is_modified], dim=1)  # (N, 3)
    x = torch.cat([x, node_state], dim=1)

    edge_index = graph_state.get_current_edge_index()

    global_feat = None
    if use_global_feat:
        global_feat = _compute_global_features(G_current)

    return {
        'x': x,
        'edge_index': edge_index,
        'node_list': node_list,
        'selected_mask': selected_mask,
        'global_feat': global_feat,
        'phase': phase,
    }
