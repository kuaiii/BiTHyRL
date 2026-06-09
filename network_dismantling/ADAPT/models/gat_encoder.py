"""
GAT-based local encoder with attentional graph pooling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax as tg_softmax

from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder for ego-networks.
    Supports fallback to MLP for very dense / complete graphs.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        complete_graph_threshold: int = 500,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.complete_graph_threshold = complete_graph_threshold

        # GAT layers
        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()  # residual linear projections
        dims = [in_channels] + [hidden_dim] * num_layers
        for i in range(num_layers):
            in_ch = dims[i]
            # If not first layer and concat is True, input dim is hidden_dim * heads
            if i > 0:
                in_ch = hidden_dim * heads
            self.convs.append(
                GATConv(
                    in_channels=in_ch,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=True,
                    negative_slope=0.2,
                    dropout=dropout,
                    add_self_loops=True,
                    bias=True,
                )
            )
            self.lins.append(
                nn.Linear(in_ch, hidden_dim * heads, bias=False)
            )

        # Projection to out_dim after GAT layers
        self.node_proj = nn.Linear(hidden_dim * heads, out_dim)

        # Attentional graph pooling
        self.pool_attn = nn.Linear(out_dim, 1)

        # MLP fallback for complete graphs
        self.mlp_fallback = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _is_nearly_complete(self, num_nodes: int, num_edges: int) -> bool:
        if num_nodes < self.complete_graph_threshold:
            return False
        max_edges = num_nodes * (num_nodes - 1) // 2
        return num_edges >= 0.95 * max_edges

    def forward(self, x, edge_index, num_nodes=None, num_edges=None, batch=None):
        """
        Parameters
        ----------
        x : Tensor, shape (N, in_channels) or (total_nodes, in_channels) when batching
        edge_index : Tensor, shape (2, E)
        num_nodes : int, optional (single-graph mode only)
        num_edges : int, optional (single-graph mode only)
        batch : LongTensor, shape (N,), optional. If provided, runs in batch mode.

        Returns
        -------
        node_emb : Tensor, shape (N, out_dim)
        graph_emb : Tensor, shape (out_dim,) in single mode, (B, out_dim) in batch mode
        """
        is_batch = batch is not None

        if not is_batch:
            if num_nodes is None:
                num_nodes = x.size(0)
            if num_edges is None:
                num_edges = edge_index.size(1)

            # Fallback for nearly complete large graphs
            if self._is_nearly_complete(num_nodes, num_edges):
                node_emb = self.mlp_fallback(x)
                graph_emb = node_emb.mean(dim=0)
                return node_emb, graph_emb

        # GAT layers with residual (same for single / batch)
        for conv, lin in zip(self.convs, self.lins):
            out = F.elu(conv(x, edge_index) + lin(x))
            x = out

        # Project to out_dim
        node_emb = self.node_proj(x)

        # Attentional graph pooling
        attn_scores = self.pool_attn(node_emb).squeeze(-1)  # (N,)
        if is_batch:
            attn_weights = tg_softmax(attn_scores, batch)  # (N,)
            graph_emb = global_add_pool(node_emb * attn_weights.unsqueeze(-1), batch)  # (B, out_dim)
        else:
            attn_weights = F.softmax(attn_scores, dim=0)
            graph_emb = (node_emb * attn_weights.unsqueeze(-1)).sum(dim=0)  # (out_dim,)

        return node_emb, graph_emb
