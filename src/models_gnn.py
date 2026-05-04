from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyBipartiteGNN(nn.Module):
    """Two-step cross-type propagation (lnc <-> disease) without external GNN libs."""

    def __init__(self, n_lnc: int, n_dis: int, dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.l0 = nn.Embedding(n_lnc, dim)
        self.d0 = nn.Embedding(n_dis, dim)
        self.msg_ld = nn.Linear(dim, dim, bias=False)
        self.msg_dl = nn.Linear(dim, dim, bias=False)
        self.lin_l = nn.Linear(dim * 2, dim)
        self.lin_d = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(dim * 2, 1, bias=False)
        nn.init.xavier_uniform_(self.l0.weight)
        nn.init.xavier_uniform_(self.d0.weight)

    def propagate(self, adj_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_l = self.l0.weight
        h_d = self.d0.weight
        m_d = adj_norm.T @ h_l
        h_d1 = torch.tanh(self.msg_ld(m_d))
        m_l = adj_norm @ h_d1
        h_l1 = torch.tanh(self.msg_dl(m_l))
        h_l_out = torch.tanh(self.lin_l(torch.cat([h_l, h_l1], dim=-1)))
        h_d_out = torch.tanh(self.lin_d(torch.cat([h_d, h_d1], dim=-1)))
        h_l_out = self.dropout(h_l_out)
        h_d_out = self.dropout(h_d_out)
        return h_l_out, h_d_out

    def forward(
        self,
        li: torch.Tensor,
        dj: torch.Tensor,
        h_l: torch.Tensor,
        h_d: torch.Tensor,
    ) -> torch.Tensor:
        z = torch.cat([h_l[li], h_d[dj]], dim=-1)
        return self.score(z).squeeze(-1)

    @torch.no_grad()
    def logits_matrix(self, adj_norm: torch.Tensor) -> torch.Tensor:
        h_l, h_d = self.propagate(adj_norm)
        return h_l @ h_d.T
