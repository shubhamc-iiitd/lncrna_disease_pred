#!/usr/bin/env python3
"""Annotation-bias audit: do top-scoring *novel* (non-)edges cluster by disease category?

Fits the same hybrid scorer as the portal on the full graph, ranks all absent
edges, takes the top ``--top-k`` by score, and compares the **disease category**
mixture of those pairs to:

1. **Positive edges** — empirical mixture among all curated associations (often
   dominated by well-studied classes such as neoplasms).
2. **Disease pool** — uniform over disease nodes (each disease counts once).

Large **fold change** vs positives for a category suggests top predictions track
**where the literature is dense** (annotation / reporting bias) rather than a
flat biological prior. Interpreting this needs care: real biology can also
concentrate in cancer; the comparison is a **sanity check**, not a definitive test.

Writes ``figures/category_novel_enrichment.png`` by default.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask_tool.ranker import HybridLinkScorer  # noqa: E402
from src.dataio import load_bipartite  # noqa: E402


def _score_matrix(ranker: HybridLinkScorer) -> np.ndarray:
    _, n_d = ranker.cn_log.shape
    S = np.empty_like(ranker.cn_log, dtype=np.float32)
    for j in range(n_d):
        S[:, j] = ranker.scores_for_disease(j)
    return S


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--top-k", type=int, default=3000)
    p.add_argument("--n-components", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-figure", type=Path, default=ROOT / "figures" / "category_novel_enrichment.png")
    args = p.parse_args()

    if not (args.data_dir / "associations.csv").is_file():
        raise SystemExit(f"Missing data under {args.data_dir} (ingest v3.0 first).")

    bp = load_bipartite(args.data_dir)
    Y = bp.Y.tocsr()
    n_l, n_d = Y.shape

    ranker = HybridLinkScorer.fit(Y, n_components=args.n_components, random_state=args.seed)
    S = _score_matrix(ranker)
    Y_dense = Y.toarray() > 0.5
    novel_mask = ~Y_dense
    n_novel = int(novel_mask.sum())
    k = min(args.top_k, n_novel)
    if k < 100:
        raise SystemExit("Too few novel pairs for a stable category audit.")

    scores_flat = S[novel_mask]
    idx_flat = np.flatnonzero(novel_mask.ravel(order="C"))
    part = np.argpartition(-scores_flat, kth=k - 1)[:k]
    top_flat = idx_flat[part]
    top_scores = scores_flat[part]
    order = np.argsort(-top_scores)
    top_flat = top_flat[order]

    # C-order ravel: flat = i * n_d + j
    js = (top_flat % n_d).astype(np.int64)

    def cat_for_col(j: int) -> str:
        did = bp.disease_ids[j]
        return bp.disease_category.get(did, "Unknown")

    top_cats = [cat_for_col(int(j)) for j in js]
    top_counts = Counter(top_cats)

    pos_cats: list[str] = []
    for ii, jj in zip(*Y.nonzero()):
        did = bp.disease_ids[int(jj)]
        pos_cats.append(bp.disease_category.get(did, "Unknown"))
    pos_counts = Counter(pos_cats)

    pool_cats = [bp.disease_category.get(d, "Unknown") for d in bp.disease_ids]
    pool_counts = Counter(pool_cats)

    categories = sorted(set(top_counts) | set(pos_counts) | set(pool_counts))

    def frac(counter: Counter, cat: str, denom: int) -> float:
        return counter.get(cat, 0) / max(denom, 1)

    n_pos = Y.nnz
    n_pool = len(bp.disease_ids)

    folds = []
    for c in categories:
        r = frac(top_counts, c, k)
        p = frac(pos_counts, c, n_pos)
        folds.append(r / (p + 1e-12))

    args.out_figure.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(categories))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 0.45), 4.8))
    ax.bar(x - w, [frac(top_counts, c, k) for c in categories], width=w, label=f"Top-{k} novel pairs", color="#2a6f97")
    ax.bar(x, [frac(pos_counts, c, n_pos) for c in categories], width=w, label="Positive edges (curated)", color="#e76f51")
    ax.bar(x + w, [frac(pool_counts, c, n_pool) for c in categories], width=w, label="Disease pool (1×/disease)", color="#8d99ae")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Fraction")
    ax.set_title("Disease category mix: top novel predictions vs baselines")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, min(1.05, max(ax.get_ylim()[1], 0.05) * 1.15))
    fig.tight_layout()
    fig.savefig(args.out_figure, dpi=150)
    plt.close(fig)

    # Fold vs positives (second small plot)
    fig2, ax2 = plt.subplots(figsize=(max(8, len(categories) * 0.45), 3.8))
    colors = ["#c1121f" if f > 1.25 or f < 0.75 else "#457b9d" for f in folds]
    ax2.axhline(1.0, color="gray", ls="--", lw=1)
    ax2.bar(x, folds, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Fold vs positives")
    ax2.set_title("Enrichment: (fraction in top-novel) / (fraction among positives)")
    fig2.tight_layout()
    fold_path = args.out_figure.with_name(args.out_figure.stem + "_fold.png")
    fig2.savefig(fold_path, dpi=150)
    plt.close(fig2)

    print(f"Top-{k} novel pairs (of {n_novel} absent edges). Wrote:")
    print(f"  {args.out_figure}")
    print(f"  {fold_path}")
    ranked = sorted(
        ((c, top_counts[c] / k, pos_counts[c] / n_pos, (top_counts[c] / k) / (pos_counts[c] / n_pos + 1e-12)) for c in categories),
        key=lambda t: -t[3],
    )
    print("\nCategory | frac top-novel | frac positives | fold vs positives")
    for c, a, b, f in ranked:
        if top_counts[c] > 0 or pos_counts[c] / n_pos > 0.02:
            print(f"  {c:28} {a:6.3f}  {b:6.3f}  {f:5.2f}x")


if __name__ == "__main__":
    main()
