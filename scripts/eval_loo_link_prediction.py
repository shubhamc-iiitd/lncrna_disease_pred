#!/usr/bin/env python3
"""Leave-one-out link prediction: mask each positive edge, refit SVD, score vs negatives.

Produces ROC and precision–recall curves (AUROC, AUPR) for the same latent model
used in the Flask portal (truncated SVD on the bipartite adjacency).

Full leave-one-out on very large graphs is slow (one SVD fit per positive). By
default, if there are more than ``--loo-threshold`` positives, a fixed-size
random subset is evaluated (stratified LOO-style audit).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask_tool.ranker import LatentRanker  # noqa: E402
from src.dataio import load_bipartite  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "examples" / "minimal_data")
    p.add_argument("--n-components", type=int, default=32)
    p.add_argument("--n-neg", type=int, default=25, help="Random negatives per held-out positive (same lncRNA row)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loo-threshold", type=int, default=400, help="If more positives exist, subsample to this many")
    p.add_argument("--all-loo", action="store_true", help="Evaluate every positive (can be very slow)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    bp = load_bipartite(args.data_dir)
    Y = bp.Y.tocsr()
    n_l, n_d = Y.shape
    pairs = list(zip(*Y.nonzero()))
    n_pos_total = len(pairs)
    if args.all_loo or n_pos_total <= args.loo_threshold:
        eval_pairs = pairs
        subsampled = False
    else:
        idx = rng.choice(n_pos_total, size=args.loo_threshold, replace=False)
        eval_pairs = [pairs[i] for i in idx]
        subsampled = True

    y_true: list[int] = []
    y_score: list[float] = []

    for t, (i, j) in enumerate(eval_pairs):
        Ym = Y.copy()
        if Ym[i, j] == 0:
            continue
        Ym[i, j] = 0
        Ym.eliminate_zeros()
        try:
            ranker = LatentRanker.fit(Ym, n_components=args.n_components, random_state=args.seed)
        except Exception:
            continue
        s_pos = float(ranker.U[i] @ ranker.Vt[:, j])
        y_true.append(1)
        y_score.append(s_pos)

        row = Ym.getrow(i)
        present = set(row.indices.tolist())
        candidates = [c for c in range(n_d) if c not in present]
        if not candidates:
            continue
        take = min(args.n_neg, len(candidates))
        neg_js = rng.choice(candidates, size=take, replace=False)
        for jn in neg_js:
            y_true.append(0)
            y_score.append(float(ranker.U[i] @ ranker.Vt[:, int(jn)]))

    y_true_arr = np.array(y_true, dtype=np.int32)
    y_score_arr = np.array(y_score, dtype=np.float64)
    if y_true_arr.sum() == 0 or (y_true_arr == 0).sum() == 0:
        raise SystemExit("Not enough mixed labels for ROC/PR (check data and negatives).")

    auroc = roc_auc_score(y_true_arr, y_score_arr)
    aupr = average_precision_score(y_true_arr, y_score_arr)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Leave-one-out ROC (masked SVD per held-out edge)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.text(
        0.02,
        0.02,
        f"data={args.data_dir.name}  positives={len(eval_pairs)}/{n_pos_total}  n_neg/pos={args.n_neg}"
        + ("  [subsampled]" if subsampled else ""),
        fontsize=7,
        color="#444",
    )
    fig.tight_layout()
    roc_path = args.out_dir / "loo_roc.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_true_arr, y_score_arr)
    fig2, ax2 = plt.subplots(figsize=(5, 4.5))
    ax2.plot(rec, prec, lw=2, label=f"AUPR = {aupr:.3f}")
    baseline = y_true_arr.mean()
    ax2.axhline(baseline, color="gray", ls="--", lw=1, label=f"baseline = {baseline:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Leave-one-out precision–recall")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    fig2.text(
        0.02,
        0.02,
        f"data={args.data_dir.name}  positives={len(eval_pairs)}/{n_pos_total}",
        fontsize=7,
        color="#444",
    )
    fig2.tight_layout()
    pr_path = args.out_dir / "loo_pr.png"
    fig2.savefig(pr_path, dpi=150)
    plt.close(fig2)

    print(f"AUROC={auroc:.4f}  AUPR={aupr:.4f}")
    print(f"Wrote {roc_path} and {pr_path}")
    if subsampled:
        print(f"Note: subsampled {len(eval_pairs)} of {n_pos_total} positives (use --all-loo for full LOO).")


if __name__ == "__main__":
    main()
