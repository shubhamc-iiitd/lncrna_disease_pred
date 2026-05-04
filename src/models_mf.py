from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BipartiteMF(nn.Module):
    """Logistic matrix factorization on a bipartite incidence graph."""

    def __init__(self, n_lnc: int, n_dis: int, rank: int = 32):
        super().__init__()
        self.l_emb = nn.Embedding(n_lnc, rank)
        self.d_emb = nn.Embedding(n_dis, rank)
        nn.init.xavier_uniform_(self.l_emb.weight)
        nn.init.xavier_uniform_(self.d_emb.weight)

    def forward(self, li: torch.Tensor, dj: torch.Tensor) -> torch.Tensor:
        return (self.l_emb(li) * self.d_emb(dj)).sum(dim=-1)

    @torch.no_grad()
    def score_all_pairs(self) -> torch.Tensor:
        """Return logits matrix (n_lnc, n_disease)."""
        return self.l_emb.weight @ self.d_emb.weight.T
