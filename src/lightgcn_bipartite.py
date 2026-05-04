"""Bipartite LightGCN-style embeddings (no feature transform; linear propagation only).

Reference idea: He et al., LightGCN (SIGIR 2020), adapted to an lncRNA–disease
incidence matrix R with symmetric normalization 1/sqrt(deg_l * deg_d).

Only the **layer-0** embeddings are learned; higher layers are fixed graph propagation.
Scores are dot products in the mean-pooled multi-layer embedding space.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_normalized_adjacency(
    Y: sp.csr_matrix,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Symmetric-normalized bipartite adjacency R (n_l × n_d) as sparse COO."""
    Y = Y.tocoo()
    n_l, n_d = Y.shape
    rows = Y.row.astype(np.int64)
    cols = Y.col.astype(np.int64)
    deg_l = np.asarray(Y.sum(axis=1)).ravel().astype(np.float64)
    deg_d = np.asarray(Y.sum(axis=0)).ravel().astype(np.float64)
    deg_l = np.maximum(deg_l, 1.0)
    deg_d = np.maximum(deg_d, 1.0)
    vals = 1.0 / np.sqrt(deg_l[rows] * deg_d[cols])
    idx = torch.tensor(np.stack([rows, cols]), dtype=torch.long, device=device)
    v = torch.tensor(vals, dtype=torch.float32, device=device)
    R = torch.sparse_coo_tensor(idx, v, (n_l, n_d), device=device).coalesce()
    return R, R.transpose(0, 1).coalesce(), (n_l, n_d)


class BipartiteLightGCN(nn.Module):
    def __init__(self, n_l: int, n_d: int, dim: int, n_layers: int = 3):
        super().__init__()
        self.n_l = n_l
        self.n_d = n_d
        self.dim = dim
        self.n_layers = n_layers
        self.embedding_l = nn.Embedding(n_l, dim)
        self.embedding_d = nn.Embedding(n_d, dim)
        nn.init.normal_(self.embedding_l.weight, std=0.1)
        nn.init.normal_(self.embedding_d.weight, std=0.1)

    def propagate(self, R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e_l = [self.embedding_l.weight]
        e_d = [self.embedding_d.weight]
        Rt = R.transpose(0, 1).coalesce()
        for _ in range(self.n_layers):
            prev_l, prev_d = e_l[-1], e_d[-1]
            # disease layer k+1 from lnc layer k; lnc layer k+1 from disease layer k
            e_d.append(torch.sparse.mm(Rt, prev_l))
            e_l.append(torch.sparse.mm(R, prev_d))
        E_l = torch.stack(e_l, dim=0).mean(dim=0)
        E_d = torch.stack(e_d, dim=0).mean(dim=0)
        return E_l, E_d

    def forward_logits_pairs(self, R: torch.Tensor, li: torch.Tensor, dj: torch.Tensor) -> torch.Tensor:
        E_l, E_d = self.propagate(R)
        return (E_l[li] * E_d[dj]).sum(dim=-1)

    @torch.no_grad()
    def score_matrix(self, R: torch.Tensor) -> torch.Tensor:
        E_l, E_d = self.propagate(R)
        return E_l @ E_d.T


def train_lightgcn(
    Y_train: sp.csr_matrix,
    *,
    dim: int = 64,
    n_layers: int = 3,
    epochs: int = 200,
    batch_edges: int = 4096,
    lr: float = 0.001,
    reg: float = 1e-4,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train on training edges + negative sampling. Returns (logit_matrix_cpu, meta)."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    R, _, shape = build_normalized_adjacency(Y_train, dev)
    n_l, n_d = shape
    pos = list(zip(*Y_train.nonzero()))
    if not pos:
        raise ValueError("empty training graph")
    model = BipartiteLightGCN(n_l, n_d, dim, n_layers=n_layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    Y_csr = Y_train.tocsr()

    def neg_j(i: int, j_pos: int) -> int:
        for _ in range(64):
            j = int(rng.integers(0, n_d))
            if j != j_pos and Y_csr[i, j] == 0:
                return j
        return int(rng.integers(0, n_d))

    for _ in range(epochs):
        model.train()
        perm = rng.permutation(len(pos))
        for start in range(0, len(pos), batch_edges):
            batch = perm[start : start + batch_edges]
            li = torch.tensor([pos[k][0] for k in batch], device=dev, dtype=torch.long)
            dj = torch.tensor([pos[k][1] for k in batch], device=dev, dtype=torch.long)
            jn = torch.tensor([neg_j(int(li[b]), int(dj[b])) for b in range(len(batch))], device=dev, dtype=torch.long)
            logits_pos = model.forward_logits_pairs(R, li, dj)
            logits_neg = model.forward_logits_pairs(R, li, jn)
            loss = F.binary_cross_entropy_with_logits(logits_pos, torch.ones_like(logits_pos))
            loss = loss + F.binary_cross_entropy_with_logits(logits_neg, torch.zeros_like(logits_neg))
            loss = loss + reg * (model.embedding_l.weight.pow(2).mean() + model.embedding_d.weight.pow(2).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model.score_matrix(R).cpu().numpy().astype(np.float32)
    meta = {
        "dim": dim,
        "n_layers": n_layers,
        "n_l": n_l,
        "n_d": n_d,
        "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
    }
    return logits, meta


def save_lightgcn_checkpoint(meta: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(meta, path)


def load_lightgcn_for_inference(
    Y: sp.csr_matrix,
    path: Path,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, BipartiteLightGCN, torch.Tensor]:
    """Load weights and score full incidence graph Y (same shape as training)."""
    dev = torch.device(device)
    try:
        blob = torch.load(path, map_location=dev, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=dev)
    n_l, n_d = int(blob["n_l"]), int(blob["n_d"])
    dim = int(blob["dim"])
    n_layers = int(blob["n_layers"])
    if Y.shape != (n_l, n_d):
        raise ValueError(f"checkpoint expects Y {n_l}x{n_d}, got {Y.shape}")
    R, _, _ = build_normalized_adjacency(Y, dev)
    model = BipartiteLightGCN(n_l, n_d, dim, n_layers=n_layers).to(dev)
    model.load_state_dict(blob["model_state"])
    model.eval()
    with torch.no_grad():
        S = model.score_matrix(R).cpu().numpy().astype(np.float32)
    return S, model, R
