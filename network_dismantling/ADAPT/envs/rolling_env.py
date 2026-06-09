"""
Rolling-Window Network Dismantling Environment.

Simulates dismantle_khop_rolling during training:
  - visible_nodes starts as k-hop neighborhood of initial_center.
  - After each removal, k-hop neighbors of removed node are added to visible_nodes.
  - Agent observes only the visible subgraph and picks nodes from it.
  - Episode-level reward incorporates AUC and remove_num.
"""
import math
from typing import Tuple, Dict, Any, Optional, List, Set

import networkx as nx
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.utils import degree, to_undirected

from network_dismantling.ADAPT.envs.dynamics import apply_dynamics


def _bfs_khop_nodes(G: nx.Graph, source: int, k_hop: int) -> Set[int]:
    """BFS k-hop neighborhood (includes source)."""
    visited = {source}
    frontier = {source}
    for _ in range(k_hop):
        next_frontier = set()
        for node in frontier:
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        if not next_frontier:
            break
        frontier = next_frontier
    return visited


class RollingDismantleEnv:
    """
    POMDP environment that mimics dismantle_khop_rolling.
    """

    def __init__(
        self,
        k_hop: int = 2,
        reward_metric: str = "gcc",
        dynamics_mode: str = "static",
        dynamics_kwargs: Optional[Dict[str, Any]] = None,
        max_nodes: int = 1000,
        seed: Optional[int] = None,
        device: torch.device = None,
        reward_weights: Optional[Dict[str, float]] = None,
        terminal_reward_weights: Optional[Dict[str, float]] = None,
        node_feat_dim: int = 10,
    ):
        self.k_hop = k_hop
        self.reward_metric = reward_metric
        self.dynamics_mode = dynamics_mode
        self.dynamics_kwargs = dynamics_kwargs or {}
        self.max_nodes = max_nodes
        self.rng = np.random.default_rng(seed)
        self.device = device if device is not None else torch.device("cpu")
        self.reward_weights = reward_weights or {"gcc": 1.0, "slcc": 0.0, "fragment": 0.0}
        self.terminal_reward_weights = terminal_reward_weights or {
            "auc": 0.5,
            "remove_num": 0.5,
            "penalty_fail": 1.0,
        }
        self.node_feat_dim = node_feat_dim

        # Internal state
        self.G_nx: nx.Graph = None          # kept for GCC checks & dynamics
        self.num_nodes: int = 0
        self.removed_mask: torch.Tensor = None  # (N,) bool
        self.visible_nodes: Set[int] = set()
        self.removed_nodes: List[int] = []
        self.trace_lcc: List[float] = []
        self.agent_node: int = None         # reference node (initial_center)
        self.step_count: int = 0
        self.done: bool = False
        self.last_obs: Optional[Dict[str, Any]] = None

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _gcc_size(self, G: nx.Graph) -> int:
        if G.number_of_nodes() == 0:
            return 0
        return max(len(c) for c in nx.connected_components(G))

    def _slcc_size(self, G: nx.Graph) -> int:
        if G.number_of_nodes() == 0:
            return 0
        comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
        return comps[1] if len(comps) > 1 else 0

    def _compute_reward(self, n_prev: int, gcc_prev: int, slcc_prev: int, G_next: nx.Graph) -> float:
        if n_prev == 0:
            return 0.0
        n_next = G_next.number_of_nodes()
        gcc_next = self._gcc_size(G_next) if n_next > 0 else 0
        r_gcc = float((gcc_prev - gcc_next) / n_prev)

        slcc_next = self._slcc_size(G_next) if n_next > 0 else 0
        r_slcc = float((slcc_prev - slcc_next) / n_prev)

        r_frag = 0.0
        if gcc_next > 0:
            r_frag = 1.0 / gcc_next
        else:
            r_frag = 2.0 / n_prev

        w = self.reward_weights
        return w.get("gcc", 1.0) * r_gcc + w.get("slcc", 0.0) * r_slcc + w.get("fragment", 0.0) * r_frag

    def _compute_terminal_reward(self) -> float:
        """
        Compute episode-level terminal reward based on AUC and remove_num.
        Both are "lower is better", so we negate them.
        """
        n = self.num_nodes
        if n == 0 or len(self.trace_lcc) == 0:
            return 0.0

        stop_10 = max(1, int(0.1 * n))
        q_vals = np.arange(1, len(self.trace_lcc) + 1) / n
        auc = float(np.trapz(self.trace_lcc, q_vals))

        rem_num = n
        for i, lcc_frac in enumerate(self.trace_lcc):
            if lcc_frac <= 0.1:
                rem_num = i + 1
                break

        w = self.terminal_reward_weights
        r_terminal = -(w.get("auc", 0.5) * auc + w.get("remove_num", 0.5) * (rem_num / n))
        if rem_num == n:
            r_terminal -= w.get("penalty_fail", 1.0)

        return r_terminal

    # -----------------------------------------------------------------------
    # Observation building on visible subgraph
    # -----------------------------------------------------------------------
    def _build_obs(self) -> Dict[str, Any]:
        alive_visible = [v for v in self.visible_nodes if not self.removed_mask[v].item()]
        m = len(alive_visible)
        if m == 0:
            return self._empty_obs()

        # Build edge_index on alive_visible subgraph
        # We use the original G_nx (which is never modified) but filter edges
        edge_list = []
        alive_set = set(alive_visible)
        for u, v in self.G_nx.edges():
            if u in alive_set and v in alive_set:
                edge_list.append((u, v))

        # Relabel to 0..m-1
        local_idx_map = {node: i for i, node in enumerate(alive_visible)}
        if len(edge_list) > 0:
            ei = torch.tensor(
                [(local_idx_map[u], local_idx_map[v]) for u, v in edge_list],
                dtype=torch.long, device=self.device
            ).t()
            edge_index_sub = to_undirected(ei)
        else:
            edge_index_sub = torch.zeros((2, 0), dtype=torch.long, device=self.device)

        subset = torch.tensor(alive_visible, dtype=torch.long, device=self.device)
        mapping = torch.tensor([local_idx_map[self.agent_node]], dtype=torch.long, device=self.device) \
            if self.agent_node in local_idx_map else torch.tensor([0], dtype=torch.long, device=self.device)

        x = self._compute_node_features(subset, edge_index_sub, mapping, local_idx_map)

        action_mask = torch.ones(m + 1, dtype=torch.bool, device=self.device)
        action_mask[-1] = True  # terminate always allowed

        return {
            "x": x,
            "edge_index": edge_index_sub,
            "node_map": alive_visible,  # global node IDs
            "agent_local_idx": mapping.item(),
            "global_n": self.num_nodes,
            "action_mask": action_mask,
        }

    def _empty_obs(self) -> Dict[str, Any]:
        return {
            "x": torch.zeros((0, self.node_feat_dim), dtype=torch.float32, device=self.device),
            "edge_index": torch.zeros((2, 0), dtype=torch.long, device=self.device),
            "node_map": [],
            "agent_local_idx": -1,
            "global_n": 0,
            "action_mask": torch.zeros(1, dtype=torch.bool, device=self.device),
        }

    def _compute_node_features(
        self,
        subset: torch.Tensor,
        edge_index_sub: torch.Tensor,
        mapping: torch.Tensor,
        local_idx_map: Dict[int, int],
    ) -> torch.Tensor:
        m = subset.size(0)
        x = torch.zeros(m, self.node_feat_dim, dtype=torch.float32, device=self.device)

        if m == 0:
            return x

        # degree (local in visible subgraph)
        deg = degree(edge_index_sub[0], num_nodes=m).float()
        max_deg = deg.max().item() if m > 0 else 1.0
        if max_deg < 1:
            max_deg = 1.0

        # density
        e_sub = edge_index_sub.size(1) // 2
        max_e = m * (m - 1) / 2 if m > 1 else 1
        ego_density = e_sub / max_e

        # scale estimate
        scale = m / self.max_nodes

        if self.node_feat_dim > 7:
            ego_nx = nx.Graph()
            ego_nx.add_nodes_from(range(m))
            edges = edge_index_sub.t().tolist()
            ego_nx.add_edges_from([(u, v) for u, v in edges if u != v])

            mapping_item = mapping.item()
            if m > 1 and edge_index_sub.size(1) > 0:
                try:
                    dists = nx.single_source_shortest_path_length(ego_nx, mapping_item)
                except Exception:
                    dists = {i: self.k_hop for i in range(m)}
            else:
                dists = {i: 0 if i == mapping_item else self.k_hop for i in range(m)}

            betweenness = {}
            clustering = {}
            if m <= 200 and m > 1:
                try:
                    betweenness = nx.betweenness_centrality(ego_nx, normalized=True)
                except Exception:
                    pass
                try:
                    clustering = nx.clustering(ego_nx)
                except Exception:
                    pass

            for i in range(m):
                x[i, 0] = deg[i].item() / max_deg
                x[i, 1] = betweenness.get(i, 0.0)
                x[i, 2] = clustering.get(i, 0.0)
                x[i, 3] = dists.get(i, self.k_hop) / max(self.k_hop, 1)
                x[i, 4] = 0.0  # removed flag (all visible are alive)
                x[i, 5] = ego_density
                x[i, 6] = scale
                x[i, 7] = (deg[i].item() ** 2) / (max_deg ** 2 + 1e-6)
                if self.node_feat_dim > 8:
                    x[i, 8] = 1.0 if i == mapping_item else 0.0
                if self.node_feat_dim > 9:
                    x[i, 9] = ego_density * deg[i].item() / max_deg
        else:
            mapping_item = mapping.item()
            if m > 1 and edge_index_sub.size(1) > 0:
                edges_np = edge_index_sub.cpu().numpy()
                data = np.ones(edges_np.shape[1], dtype=np.float32)
                csr = csr_matrix((data, (edges_np[0], edges_np[1])), shape=(m, m))
                dist_vals = shortest_path(csr, directed=False, unweighted=True, indices=mapping_item)
                dist_vals = np.where(np.isinf(dist_vals), self.k_hop, dist_vals).astype(np.float32)
            else:
                dist_vals = np.full(m, self.k_hop, dtype=np.float32)
                dist_vals[mapping_item] = 0.0

            x[:, 0] = deg / max_deg
            x[:, 1] = 0.0
            x[:, 2] = 0.0
            x[:, 3] = torch.from_numpy(dist_vals).to(self.device) / max(self.k_hop, 1)
            x[:, 4] = 0.0
            x[:, 5] = ego_density
            x[:, 6] = scale

        return x

    # -----------------------------------------------------------------------
    # Gym-like interface
    # -----------------------------------------------------------------------
    def reset(self, G: nx.Graph, agent_node: Optional[int] = None) -> Dict[str, Any]:
        # Relabel nodes to contiguous 0..n-1
        if G.number_of_nodes() > 0:
            mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
            reverse_mapping = {i: node for node, i in mapping.items()}
            G = nx.relabel_nodes(G, mapping)
            if agent_node is not None and agent_node in mapping:
                agent_node = mapping[agent_node]
        else:
            reverse_mapping = {}

        self.G_nx = G.copy()
        self.num_nodes = G.number_of_nodes()
        self._reverse_mapping = reverse_mapping
        self.step_count = 0
        self.done = False
        self.removed_nodes = []
        self.trace_lcc = []

        nodes = list(G.nodes())
        if len(nodes) == 0:
            self.agent_node = None
            self.visible_nodes = set()
            self.removed_mask = torch.zeros(0, dtype=torch.bool, device=self.device)
            self.last_obs = self._empty_obs()
            return self.last_obs

        if agent_node is None or agent_node not in G:
            self.agent_node = int(self.rng.choice(nodes))
        else:
            self.agent_node = agent_node

        self.removed_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)

        # Initialize visible_nodes as k-hop neighborhood of initial center
        self.visible_nodes = _bfs_khop_nodes(self.G_nx, self.agent_node, self.k_hop)

        # Record initial GCC
        gcc0 = self._gcc_size(self.G_nx)
        self.trace_lcc.append(gcc0 / self.num_nodes)

        obs = self._build_obs()
        self.last_obs = obs
        return obs

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            return self._empty_obs(), 0.0, True, {"reason": "already_done"}

        obs = self.last_obs
        m = len(obs["node_map"])

        if action < 0 or action >= m + 1:
            return obs, -1.0, True, {"reason": "invalid_action"}
        if not obs["action_mask"][action].item():
            return obs, -1.0, True, {"reason": "masked_action"}

        if action == m:
            # Terminate: compute terminal reward
            r_terminal = self._compute_terminal_reward()
            self.done = True
            return obs, r_terminal, True, {"reason": "terminate", "removed_node": None}

        # Cache graph metrics before modifying
        G_alive = self.G_nx.copy()
        alive_nodes = [v for v in self.G_nx.nodes() if not self.removed_mask[v].item()]
        G_alive.remove_nodes_from([v for v in self.G_nx.nodes() if self.removed_mask[v].item()])

        n_prev = G_alive.number_of_nodes()
        gcc_prev = self._gcc_size(G_alive)
        slcc_prev = self._slcc_size(G_alive)

        node_to_remove = obs["node_map"][action]

        # Mark as removed
        self.removed_mask[node_to_remove] = True
        self.removed_nodes.append(node_to_remove)
        self.step_count += 1

        # Expand visibility: add k-hop neighbors of removed node
        if node_to_remove in self.G_nx:
            new_visible = _bfs_khop_nodes(self.G_nx, node_to_remove, self.k_hop)
            self.visible_nodes.update(new_visible)

        # Rebuild alive graph for metrics
        G_next = self.G_nx.copy()
        removed_so_far = [v for v in self.G_nx.nodes() if self.removed_mask[v].item()]
        G_next.remove_nodes_from(removed_so_far)

        # Compute step reward
        reward = self._compute_reward(n_prev, gcc_prev, slcc_prev, G_next)

        # Record GCC for trace
        gcc_next = self._gcc_size(G_next) if G_next.number_of_nodes() > 0 else 0
        self.trace_lcc.append(gcc_next / self.num_nodes)

        # Check termination
        if G_next.number_of_nodes() == 0:
            r_terminal = self._compute_terminal_reward()
            reward += r_terminal
            self.done = True
            info = {"reason": "empty_graph", "removed_node": node_to_remove}
            return self._build_obs(), reward, self.done, info

        if gcc_next <= 1:
            r_terminal = self._compute_terminal_reward()
            reward += r_terminal
            self.done = True
            info = {"reason": "fragmented", "removed_node": node_to_remove}
            return self._build_obs(), reward, self.done, info

        # Apply dynamics
        G_next = apply_dynamics(
            G_next,
            mode=self.dynamics_mode,
            rng=self.rng,
            **self.dynamics_kwargs,
        )

        # Note: dynamics may change node IDs/edges, but we keep original G_nx intact
        # For simplicity with rolling window, we sync G_nx with dynamics result
        # This means we need to rebuild the full graph after dynamics
        # However, this is tricky because removed nodes should stay removed.
        # Simplification: for rolling env, we apply dynamics only on alive nodes,
        # then merge back.
        # Actually, let's follow the original env's approach: G_nx IS the working graph.
        # But in rolling mode, we need G_nx to stay intact for visibility expansion.
        # Compromise: update G_nx edges based on dynamics, but keep all nodes.
        self._sync_graph_after_dynamics(G_next)

        # If agent_node was removed, pick a new reference from visible nodes
        if self.agent_node == node_to_remove:
            alive_visible = [v for v in self.visible_nodes if not self.removed_mask[v].item()]
            if alive_visible:
                # Pick highest-degree node in visible subgraph as new reference
                subG_visible = self.G_nx.subgraph(alive_visible)
                self.agent_node = max(alive_visible, key=lambda v: subG_visible.degree(v))
            else:
                remaining = [v for v in self.G_nx.nodes() if not self.removed_mask[v].item()]
                self.agent_node = int(self.rng.choice(remaining)) if remaining else None

        obs_next = self._build_obs()
        self.last_obs = obs_next
        info = {"removed_node": node_to_remove, "gcc": gcc_next}
        return obs_next, reward, self.done, info

    def _sync_graph_after_dynamics(self, G_alive: nx.Graph):
        """
        Sync G_nx edges after dynamics while preserving removed nodes.
        G_alive contains only alive nodes with updated edges.
        """
        # Remove all edges from G_nx that involve alive nodes
        alive_nodes = set(G_alive.nodes())
        edges_to_remove = []
        for u, v in self.G_nx.edges():
            if u in alive_nodes or v in alive_nodes:
                edges_to_remove.append((u, v))
        self.G_nx.remove_edges_from(edges_to_remove)
        # Add back edges from G_alive
        self.G_nx.add_edges_from(G_alive.edges())
