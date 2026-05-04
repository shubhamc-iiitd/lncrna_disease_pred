"""Train joint embeddings on the training subgraph: logistic MF or tiny bipartite GNN.

Both optimize link-prediction on observed training edges with negative sampling,
then expose a dense logit / score matrix for held-out evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from src.models_gnn import TinyBipartiteGNN
from src.models_mf import BipartiteMF


@dataclass
class MatrixScorer:
    """Wraps a dense score or logit matrix S with shape (n_lnc, n_disease)."""

    S: np.ndarray

    def scores_for_disease(self, j: int) -> np.ndarray:
        return self.S[:, j]


def _sample_neg_j(
    Y: sp.csr_matrix,
    i: int,
    n_d: int,
    rng: np.random.Generator,
    forbid: set[int],
) -> int:
    for _ in range(50):
        j = int(rng.integers(0, n_d))
        if j in forbid:
            continue
        if Y[i, j] == 0:
            return j
    return int(rng.integers(0, n_d))


def train_logistic_mf(
    Y_train: sp.csr_matrix,
    *,
    rank: int = 32,
    epochs: int = 120,
    batch_edges: int = 4096,
    lr: float = 0.08,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Return logit matrix (n_l, n_d) on CPU float32."""
    torch.manual_seed(seed)
    n_l, n_d = Y_train.shape
    pos = list(zip(*Y_train.nonzero()))
    if not pos:
        raise ValueError("Y_train has no edges")
    dev = torch.device(device)
    model = BipartiteMF(n_l, n_d, rank).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    Y_csr = Y_train.tocsr()

    for _ in range(epochs):
        model.train()
        idx = rng.permutation(len(pos))
        for start in range(0, len(pos), batch_edges):
            batch = idx[start : start + batch_edges]
            li = torch.tensor([pos[k][0] for k in batch], device=dev, dtype=torch.long)
            dj = torch.tensor([pos[k][1] for k in batch], device=dev, dtype=torch.long)
            logits_pos = model(li, dj)
            jn_list = [_sample_neg_j(Y_csr, int(li[b]), n_d, rng, {int(dj[b])}) for b in range(len(batch))]
            jn = torch.tensor(jn_list, device=dev, dtype=torch.long)
            logits_neg = model(li, jn)
            loss = F.binary_cross_entropy_with_logits(logits_pos, torch.ones_like(logits_pos))
            loss = loss + F.binary_cross_entropy_with_logits(logits_neg, torch.zeros_like(logits_neg))
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model.score_all_pairs().cpu().numpy().astype(np.float32)
    return logits


def train_tiny_gnn(
    Y_train: sp.csr_matrix,
    *,
    dim: int = 32,
    epochs: int = 60,
    batch_edges: int = 4096,
    lr: float = 0.05,
    seed: int = 0,
    device: str | torch.device = "cpu",
    dropout: float = 0.0,
) -> np.ndarray:
    """Two-layer bipartite message passing; scores = dot(h_l, h_d) after propagation.

    Returns logit matrix (n_l, n_d) on CPU float32.
    """
    torch.manual_seed(seed)
    n_l, n_d = Y_train.shape
    pos = list(zip(*Y_train.nonzero()))
    if not pos:
        raise ValueError("Y_train has no edges")
    dev = torch.device(device)
    Y_den = Y_train.toarray().astype(np.float32)
    Y_t = torch.tensor(Y_den, device=dev)
    row_sum = Y_t.sum(dim=1, keepdim=True).clamp(min=1.0)
    adj_norm = Y_t / row_sum

    net = TinyBipartiteGNN(n_l, n_d, dim=dim, dropout=dropout).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    Y_csr = Y_train.tocsr()

    for _ in range(epochs):
        net.train()
        idx = rng.permutation(len(pos))
        for start in range(0, len(pos), batch_edges):
            batch = idx[start : start + batch_edges]
            li = torch.tensor([pos[k][0] for k in batch], device=dev, dtype=torch.long)
            dj = torch.tensor([pos[k][1] for k in batch], device=dev, dtype=torch.long)
            h_l, h_d = net.propagate(adj_norm)
            logits_pos = (h_l[li] * h_d[dj]).sum(dim=-1)
            jn_list = [_sample_neg_j(Y_csr, int(li[b]), n_d, rng, {int(dj[b])}) for b in range(len(batch))]
            jn = torch.tensor(jn_list, device=dev, dtype=torch.long)
            logits_neg = (h_l[li] * h_d[jn]).sum(dim=-1)
            loss = F.binary_cross_entropy_with_logits(logits_pos, torch.ones_like(logits_pos))
            loss = loss + F.binary_cross_entropy_with_logits(logits_neg, torch.zeros_like(logits_neg))
            opt.zero_grad()
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        h_l, h_d = net.propagate(adj_norm)
        logits = (h_l @ h_d.T).cpu().numpy().astype(np.float32)
    return logits
