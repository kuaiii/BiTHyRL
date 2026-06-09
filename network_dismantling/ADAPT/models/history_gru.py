"""
GRU-based history encoder for belief state approximation.
"""
import torch
import torch.nn as nn


class HistoryGRU(nn.Module):
    """
    GRU that maintains hidden state h_t based on previous graph embedding,
    previous action embedding, and previous reward.
    """

    def __init__(
        self,
        graph_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        input_dim = graph_dim + action_dim + 1  # +1 for reward scalar
        self.gru = nn.GRUCell(input_dim, hidden_dim)

    def forward(
        self,
        z_graph: torch.Tensor,
        action_emb: torch.Tensor,
        reward: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_graph : Tensor, shape (graph_dim,) or (B, graph_dim)
        action_emb : Tensor, shape (action_dim,) or (B, action_dim)
        reward : Tensor, scalar or (1,) or (B,)
        h_prev : Tensor, shape (hidden_dim,) or (B, hidden_dim)

        Returns
        -------
        h_t : Tensor, shape (hidden_dim,) or (B, hidden_dim)
        """
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)
        if reward.dim() == 1 and z_graph.dim() == 2:
            reward = reward.unsqueeze(-1)
        inp = torch.cat([z_graph, action_emb, reward], dim=-1)
        if h_prev.dim() == 1:
            h_t = self.gru(inp.unsqueeze(0), h_prev.unsqueeze(0)).squeeze(0)
        else:
            h_t = self.gru(inp, h_prev)
        return h_t
