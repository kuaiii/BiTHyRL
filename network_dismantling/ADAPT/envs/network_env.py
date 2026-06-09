"""
Network Dismantling POMDP Environment (GPU-optimized).
Agent observes only its k-hop ego-network.
"""
import math
from typing import Tuple, Dict, Any, Optional

import networkx as nx
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.utils import k_hop_subgraph, degree, to_undirected

from network_dismantling.ADAPT.envs.dynamics import apply_dynamics


class NetworkDismantleEnv:
    """
    POMDP environment for network dismantling.
    Optimized for GPU: ego-network extraction via PyG, obs tensors on device.
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
        node_feat_dim: int = 10,
    ):
        self.k_hop = k_hop
        self.reward_metric = reward_metric
        self.dynamics_mode = dynamics_mode
        self.dynamics_kwargs = dynamics_kwargs or {}
        self.max_nodes = max_nodes
        self.rng = np.random.default_rng(seed)
        self.device = device if device is not None else torch.device("cpu")
        # Multi-objective reward weights: gcc, slcc, fragment
        self.reward_weights = reward_weights or {"gcc": 1.0, "slcc": 0.0, "fragment": 0.0}
        self.node_feat_dim = node_feat_dim

        # Internal state
        self.G_nx: nx.Graph = None          # kept for GCC checks & dynamics
        self.edge_index: torch.Tensor = None  # (2, E) current graph edges
        self.num_nodes: int = 0
        self.removed_mask: torch.Tensor = None  # (N,) bool
        self.agent_node: int = None
        self.prev_gcc: float = None
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

        # SLCC penalty: discourage creating a large second component
        slcc_next = self._slcc_size(G_next) if n_next > 0 else 0
        r_slcc = float((slcc_prev - slcc_next) / n_prev)

        # Fragmentation bonus: accelerating reward when graph is close to broken
        r_frag = 0.0
        if gcc_next > 0:
            r_frag = 1.0 / gcc_next
        else:
            r_frag = 2.0 / n_prev  # bonus for fully fragmented

        w = self.reward_weights
        return w.get("gcc", 1.0) * r_gcc + w.get("slcc", 0.0) * r_slcc + w.get("fragment", 0.0) * r_frag

    def _build_obs(self) -> Dict[str, Any]:
        if self.agent_node is None or self.agent_node >= self.num_nodes or self.removed_mask[self.agent_node].item():
            return self._empty_obs()

        # Extract k-hop ego-network via PyG (on CPU; subgraph is small)
        subset, edge_index_sub, mapping, _ = k_hop_subgraph(
            self.agent_node,
            self.k_hop,
            self.edge_index,
            num_nodes=self.num_nodes,
            relabel_nodes=True,
        )

        m = subset.size(0)
        if m == 0:
            return self._empty_obs()

        # Filter out any removed nodes that may still appear (isolated)
        alive = ~self.removed_mask[subset]
        if not alive.all():
            # Re-index to keep only alive nodes
            alive_idx = alive.nonzero(as_tuple=False).view(-1)
            subset = subset[alive_idx]
            m = subset.size(0)
            if m == 0:
                return self._empty_obs()
            # Re-extract subgraph on the alive subset
            subset, edge_index_sub, mapping, _ = k_hop_subgraph(
                self.agent_node,
                self.k_hop,
                self.edge_index,
                num_nodes=self.num_nodes,
                relabel_nodes=True,
            )
            alive = ~self.removed_mask[subset]
            alive_idx = alive.nonzero(as_tuple=False).view(-1)
            if alive_idx.numel() == 0:
                return self._empty_obs()
            # Build local alive mask for edges
            local_alive = torch.zeros(subset.size(0), dtype=torch.bool, device=self.device)
            local_alive[alive_idx] = True
            edge_mask = local_alive[edge_index_sub[0]] & local_alive[edge_index_sub[1]]
            edge_index_sub = edge_index_sub[:, edge_mask]
            # Re-index nodes to 0..m-1
            old_to_new = torch.full((subset.size(0),), -1, dtype=torch.long, device=self.device)
            old_to_new[alive_idx] = torch.arange(alive_idx.size(0), device=self.device)
            edge_index_sub = old_to_new[edge_index_sub]
            valid = (edge_index_sub >= 0).all(dim=0)
            edge_index_sub = edge_index_sub[:, valid]
            subset = subset[alive_idx]
            m = subset.size(0)
            mapping = old_to_new[mapping]

        # Node features (compute on CPU, then move to device)
        x = self._compute_node_features(subset, edge_index_sub, mapping)

        # Action mask
        action_mask = torch.ones(m + 1, dtype=torch.bool, device=self.device)
        action_mask[-1] = True  # terminate always allowed

        return {
            "x": x,
            "edge_index": edge_index_sub,
            "node_map": subset.tolist(),
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
    ) -> torch.Tensor:
        m = subset.size(0)
        x = torch.zeros(m, self.node_feat_dim, dtype=torch.float32, device=self.device)

        if m == 0:
            return x

        # degree (local in ego)
        deg = degree(edge_index_sub[0], num_nodes=m).float()
        max_deg = deg.max().item() if m > 0 else 1.0
        if max_deg < 1:
            max_deg = 1.0

        # density
        e_sub = edge_index_sub.size(1) // 2  # undirected unique edges
        max_e = m * (m - 1) / 2 if m > 1 else 1
        ego_density = e_sub / max_e

        # scale estimate
        scale = m / self.max_nodes

        if self.node_feat_dim > 7:
            # Build tiny NetworkX graph for structural features
            ego_nx = nx.Graph()
            ego_nx.add_nodes_from(range(m))
            edges = edge_index_sub.t().tolist()
            ego_nx.add_edges_from([(u, v) for u, v in edges if u != v])

            # distance to agent (BFS on small ego subgraph)
            mapping_item = mapping.item()
            if m > 1 and edge_index_sub.size(1) > 0:
                try:
                    dists = nx.single_source_shortest_path_length(ego_nx, mapping_item)
                except Exception:
                    dists = {i: self.k_hop for i in range(m)}
            else:
                dists = {i: 0 if i == mapping_item else self.k_hop for i in range(m)}

            # local betweenness and clustering (only for small ego to avoid timeout)
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
                x[i, 4] = 0.0  # removed flag (all visible nodes are alive)
                x[i, 5] = ego_density
                x[i, 6] = scale
                x[i, 7] = (deg[i].item() ** 2) / (max_deg ** 2 + 1e-6)  # degree squared (hub indicator)
                if self.node_feat_dim > 8:
                    x[i, 8] = 1.0 if i == mapping_item else 0.0  # is agent node
                if self.node_feat_dim > 9:
                    x[i, 9] = ego_density * deg[i].item() / max_deg  # density-degree interaction
        else:
            # Avoid nx.Graph entirely; use scipy.sparse.csgraph.shortest_path
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
        # Relabel nodes to contiguous 0..n-1 to ensure tensor indices are valid
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
        self._reverse_mapping = reverse_mapping  # for external reference if needed
        self.step_count = 0
        self.done = False

        nodes = list(G.nodes())
        if len(nodes) == 0:
            self.agent_node = None
            self.prev_gcc = 0.0
            self.done = True
            self.edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            self.removed_mask = torch.zeros(0, dtype=torch.bool, device=self.device)
            self.last_obs = self._empty_obs()
            return self.last_obs

        if agent_node is None or agent_node not in G:
            self.agent_node = int(self.rng.choice(nodes))
        else:
            self.agent_node = agent_node

        # Build edge_index tensor
        edges = list(G.edges())
        if len(edges) > 0:
            ei = torch.tensor(edges, dtype=torch.long, device=self.device).t()
            self.edge_index = to_undirected(ei)
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

        self.removed_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        self.prev_gcc = self._gcc_size(self.G_nx)
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
            self.done = True
            return obs, 0.0, True, {"reason": "terminate", "removed_node": None}

        # Cache graph metrics before modifying
        n_prev = self.G_nx.number_of_nodes()
        gcc_prev = self._gcc_size(self.G_nx)
        slcc_prev = self._slcc_size(self.G_nx)
        node_to_remove = obs["node_map"][action]
        neighbors_of_removed = list(self.G_nx.neighbors(node_to_remove))

        # Remove selected node
        self.removed_mask[node_to_remove] = True
        self.G_nx.remove_node(node_to_remove)
        self.step_count += 1

        # Filter edge_index
        mask = ~(self.removed_mask[self.edge_index[0]] | self.removed_mask[self.edge_index[1]])
        self.edge_index = self.edge_index[:, mask]

        # Compute reward
        reward = self._compute_reward(n_prev, gcc_prev, slcc_prev, self.G_nx)

        # Check if agent node was removed -> smart repositioning (F2 fix)
        if self.agent_node == node_to_remove:
            remaining_nodes = list(self.G_nx.nodes())
            if len(remaining_nodes) > 0:
                # Strategy: move to the highest-degree neighbor of the removed node
                alive_neighbors = [n for n in neighbors_of_removed if n in self.G_nx]
                if alive_neighbors:
                    self.agent_node = max(alive_neighbors, key=lambda n: self.G_nx.degree(n))
                else:
                    # Fallback: pick from the largest connected component
                    largest_cc = max(nx.connected_components(self.G_nx), key=len)
                    self.agent_node = int(self.rng.choice(list(largest_cc)))
            else:
                self.agent_node = None

        # Check termination
        if self.G_nx.number_of_nodes() == 0:
            self.done = True
            info = {"reason": "empty_graph", "removed_node": node_to_remove}
            return self._build_obs(), reward, self.done, info

        gcc = self._gcc_size(self.G_nx)
        if gcc <= 1:
            self.done = True
            info = {"reason": "fragmented", "removed_node": node_to_remove}
            return self._build_obs(), reward, self.done, info

        # Apply dynamics
        self.G_nx = apply_dynamics(
            self.G_nx,
            mode=self.dynamics_mode,
            rng=self.rng,
            **self.dynamics_kwargs,
        )

        # Rebuild edge_index after dynamics
        edges = list(self.G_nx.edges())
        if len(edges) > 0:
            ei = torch.tensor(edges, dtype=torch.long, device=self.device).t()
            self.edge_index = to_undirected(ei)
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        # num_nodes stays as the original graph size (node IDs are preserved)
        # removed_mask stays the same size; dynamics uses original node IDs

        # Ensure agent still exists
        if self.agent_node not in self.G_nx:
            remaining = list(self.G_nx.nodes())
            self.agent_node = int(self.rng.choice(remaining)) if remaining else None

        obs_next = self._build_obs()
        self.last_obs = obs_next
        info = {"removed_node": node_to_remove, "gcc": gcc}
        return obs_next, reward, self.done, info
