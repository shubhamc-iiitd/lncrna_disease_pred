#!/usr/bin/env python3
"""Hub / popularity bias audit for top-scoring novel pairs.

Compares the **product of training degrees** (lncRNA × disease) for the highest-
scoring absent edges to a **null** of random absent pairs. If top predictions
only connect **hub lncRNAs** to **hub diseases**, scores may track **popularity**
(over-studied entities) rather than a balanced biological prior.

Uses the same hybrid scorer as ``category_bias_audit.py`` by default; set
``--checkpoint`` to audit LightGCN scores instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask_tool.ranker import HybridLinkScorer  # noqa: E402
from src.dataio import load_bipartite  # noqa: E402
from src.learned_edge_models import MatrixScorer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--top-k", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=Path, default=None, help="Optional LightGCN .pt from train_lightgcn_full.py")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    bp = load_bipartite(args.data_dir)
    Y = bp.Y.tocsr()
    n_l, n_d = Y.shape
    deg_l = np.asarray(Y.sum(axis=1)).ravel().astype(np.float64)
    deg_d = np.asarray(Y.sum(axis=0)).ravel().astype(np.float64)

    if args.checkpoint and args.checkpoint.is_file():
        from src.lightgcn_bipartite import load_lightgcn_for_inference

        S, _, _ = load_lightgcn_for_inference(Y, args.checkpoint)
        ranker = MatrixScorer(S)
        label = "LightGCN checkpoint"
    else:
        ranker = HybridLinkScorer.fit(Y, n_components=32, random_state=args.seed)
        label = "Hybrid (SVD+CN)"

    Y_dense = Y.toarray() > 0.5
    S = np.column_stack([ranker.scores_for_disease(j) for j in range(n_d)])
    novel_mask = ~Y_dense
    scores_flat = S[novel_mask]
    idx_flat = np.flatnonzero(novel_mask.ravel(order="C"))
    k = min(args.top_k, len(scores_flat))
    part = np.argpartition(-scores_flat, kth=k - 1)[:k]
    top_flat = idx_flat[part]
    top_scores = scores_flat[part]
    order = np.argsort(-top_scores)
    top_flat = top_flat[order]
    js = (top_flat % n_d).astype(np.int64)
    is_ = (top_flat // n_d).astype(np.int64)

    prod_top = deg_l[is_] * deg_d[js]
    log_top = np.log1p(prod_top)

    # null: random absent edges
    null_prod = []
    for _ in range(k):
        for __ in range(100):
            i = int(rng.integers(0, n_l))
            j = int(rng.integers(0, n_d))
            if Y[i, j] == 0:
                null_prod.append(deg_l[i] * deg_d[j])
                break
    null_prod = np.array(null_prod, dtype=np.float64)
    log_null = np.log1p(null_prod)

    print(f"Scorer: {label}")
    print(f"Top-{k} novel pairs: mean log(1+deg_l*deg_d) = {log_top.mean():.3f}  (median {np.median(log_top):.3f})")
    print(f"Random null:       mean log(1+deg_l*deg_d) = {log_null.mean():.3f}  (median {np.median(log_null):.3f})")
    ratio = log_top.mean() / (log_null.mean() + 1e-9)
    print(f"Ratio (top/null mean log-product) = {ratio:.2f}x  — values >>1 suggest hub-heavy top predictions.")


if __name__ == "__main__":
    main()
