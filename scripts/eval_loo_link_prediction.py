#!/usr/bin/env python3
"""Link-prediction metrics (ROC / PR) on the bipartite graph.

**Default (recommended on the full LncRNADisease graph):** ``--protocol holdout`` —
random **edge-level** train/test split on the **complete** incidence matrix, **one**
model fit on the training edges, then score held-out positives vs random
same-row negatives. Uses every edge exactly once across train+test (default
85% / 15% split).

**Optional:** ``--protocol loo`` — leave-one-out (one refit per positive). Use
``--all-loo`` for **every** positive on the full graph (tractable for hybrid/svd);
``--loo-threshold`` subsamples LOO for speed (**not** equivalent to full hold-out).

Figures (default names): ``figures/holdout_roc.png``, ``figures/holdout_pr.png``.

Models:

- ``hybrid`` / ``svd`` — closed-form truncated SVD (+ co-occurrence for hybrid), same spirit as the Flask UI.
- ``mf`` — **learned** logistic matrix factorization (PyTorch), dot-product embeddings on training edges + negatives.
- ``gnn`` — **learned** tiny 2-hop bipartite message-passing net (PyTorch), dot scores after propagation.
- ``lightgcn`` — **bipartite LightGCN** (PyTorch, no PyG): linear propagation on symmetric-normalized **R**, learn only layer-0 embeddings; dot scores (recommended for joint lncRNA–disease space).

For ``mf`` / ``gnn`` / ``lightgcn``, **hold-out** is the intended protocol; LOO refits per edge — use small
``--loo-epochs-*`` or stay with hold-out.

**Phase-3 ranking:** with ``--ranking-report`` (hold-out only), reports **MRR** and **HR@10 / HR@50** for each
held-out positive among **all diseases with no training edge** to that lncRNA (how high the masked 1 ranks).
On sparse graphs, **PR–AUC** is often more informative than ROC; both are plotted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask_tool.ranker import HybridLinkScorer, LatentRanker  # noqa: E402
from src.dataio import load_bipartite  # noqa: E402


def _dense_scores(ranker, n_l: int, n_d: int) -> np.ndarray:
    if hasattr(ranker, "S"):
        return ranker.S
    return np.column_stack([np.asarray(ranker.scores_for_disease(j), dtype=np.float64) for j in range(n_d)])


def _ranking_metrics(S: np.ndarray, test_pairs: list, Y_train: sp.csr_matrix) -> dict | None:
    """Rank each held-out (i,j) among diseases with no edge in Y_train to i."""
    n_d = S.shape[1]
    Yt = Y_train.tocsr()
    mrrs: list[float] = []
    hr10: list[float] = []
    hr50: list[float] = []
    for i, j in test_pairs:
        present = set(Yt.getrow(i).indices.tolist())
        cands = [c for c in range(n_d) if c not in present]
        if j not in cands:
            continue
        sub = S[i, np.array(cands, dtype=np.int64)]
        sj = float(S[i, j])
        rank = int(np.sum(sub > sj)) + 1
        mrrs.append(1.0 / rank)
        hr10.append(1.0 if rank <= 10 else 0.0)
        hr50.append(1.0 if rank <= 50 else 0.0)
    if not mrrs:
        return None
    return {
        "mrr": float(np.mean(mrrs)),
        "hr10": float(np.mean(hr10)),
        "hr50": float(np.mean(hr50)),
        "n_ranked": len(mrrs),
    }


def _fit_scorer(
    Y_train: sp.csr_matrix,
    model: str,
    n_components: int,
    seed: int,
    *,
    epochs_mf: int,
    epochs_gnn: int,
    epochs_lightgcn: int,
    lightgcn_layers: int,
    lr_lightgcn: float,
    device: str,
):
    if model == "hybrid":
        return HybridLinkScorer.fit(Y_train, n_components=n_components, random_state=seed)
    if model == "svd":
        return LatentRanker.fit(Y_train, n_components=n_components, random_state=seed)
    if model in ("mf", "gnn", "lightgcn"):
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise SystemExit("Install PyTorch for --model mf, gnn, or lightgcn: pip install torch") from e
    if model == "mf":
        from src.learned_edge_models import MatrixScorer, train_logistic_mf

        S = train_logistic_mf(
            Y_train,
            rank=n_components,
            epochs=epochs_mf,
            seed=seed,
            device=device,
        )
        return MatrixScorer(S)
    if model == "gnn":
        from src.learned_edge_models import MatrixScorer, train_tiny_gnn

        S = train_tiny_gnn(
            Y_train,
            dim=n_components,
            epochs=epochs_gnn,
            seed=seed,
            device=device,
            dropout=0.0,
        )
        return MatrixScorer(S)
    if model == "lightgcn":
        from src.learned_edge_models import MatrixScorer
        from src.lightgcn_bipartite import train_lightgcn

        S, _ = train_lightgcn(
            Y_train,
            dim=n_components,
            n_layers=lightgcn_layers,
            epochs=epochs_lightgcn,
            lr=lr_lightgcn,
            seed=seed,
            device=device,
        )
        return MatrixScorer(S)
    raise ValueError(f"unknown model {model}")


def _resolve_data_dir(p: Path) -> Path:
    if not (p / "associations.csv").is_file():
        raise SystemExit(
            f"No associations.csv under {p.resolve()}\n"
            "Run: python scripts/fetch_lncrnadisease_v30.py && python scripts/ingest_lncrnadisease_v30.py\n"
            "Or pass --data-dir examples/minimal_data for a toy graph."
        )
    return p


def run_holdout(
    Y: sp.csr_matrix,
    *,
    n_components: int,
    model: str,
    seed: int,
    train_fraction: float,
    n_neg: int,
    rng: np.random.Generator,
    epochs_mf: int,
    epochs_gnn: int,
    epochs_lightgcn: int,
    lightgcn_layers: int,
    lr_lightgcn: float,
    device: str,
    ranking_report: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n_l, n_d = Y.shape
    pairs = list(zip(*Y.nonzero()))
    if len(pairs) < 2:
        raise SystemExit("Holdout evaluation needs at least two positive edges in the graph.")
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_train = max(1, int(len(pairs) * train_fraction))
    n_train = min(n_train, len(pairs) - 1)  # need ≥1 test edge
    tr_idx, te_idx = idx[:n_train], idx[n_train:]
    train_pairs = [pairs[i] for i in tr_idx]
    test_pairs = [pairs[i] for i in te_idx]

    rows, cols = zip(*train_pairs)
    data = np.ones(len(rows), dtype=np.float32)
    Y_train = sp.csr_matrix((data, (rows, cols)), shape=(n_l, n_d))
    Y_train.eliminate_zeros()

    ranker = _fit_scorer(
        Y_train,
        model,
        n_components,
        seed,
        epochs_mf=epochs_mf,
        epochs_gnn=epochs_gnn,
        epochs_lightgcn=epochs_lightgcn,
        lightgcn_layers=lightgcn_layers,
        lr_lightgcn=lr_lightgcn,
        device=device,
    )

    y_true: list[int] = []
    y_score: list[float] = []

    for i, j in test_pairs:
        s_pos = float(ranker.scores_for_disease(j)[i])
        y_true.append(1)
        y_score.append(s_pos)

        present = set(Y_train.getrow(i).indices.tolist())
        candidates = [c for c in range(n_d) if c not in present and c != j]
        if not candidates:
            continue
        take = min(n_neg, len(candidates))
        neg_js = rng.choice(candidates, size=take, replace=False)
        for jn in neg_js:
            y_true.append(0)
            y_score.append(float(ranker.scores_for_disease(int(jn))[i]))

    meta = {
        "n_train_edges": len(train_pairs),
        "n_test_edges": len(test_pairs),
        "train_fraction": train_fraction,
    }
    if ranking_report:
        S = _dense_scores(ranker, n_l, n_d)
        rm = _ranking_metrics(S, test_pairs, Y_train)
        if rm:
            meta.update(rm)
    return np.array(y_true, dtype=np.int32), np.array(y_score, dtype=np.float64), meta


def run_loo(
    Y: sp.csr_matrix,
    *,
    n_components: int,
    model: str,
    seed: int,
    n_neg: int,
    loo_threshold: int | None,
    all_loo: bool,
    rng: np.random.Generator,
    epochs_mf: int,
    epochs_gnn: int,
    epochs_lightgcn: int,
    lightgcn_layers: int,
    lr_lightgcn: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n_l, n_d = Y.shape
    pairs = list(zip(*Y.nonzero()))
    n_pos_total = len(pairs)
    if all_loo or loo_threshold is None or n_pos_total <= loo_threshold:
        eval_pairs = pairs
        subsampled = False
    else:
        idx = rng.choice(n_pos_total, size=loo_threshold, replace=False)
        eval_pairs = [pairs[i] for i in idx]
        subsampled = True

    y_true: list[int] = []
    y_score: list[float] = []

    for i, j in eval_pairs:
        Ym = Y.copy()
        if Ym[i, j] == 0:
            continue
        Ym[i, j] = 0
        Ym.eliminate_zeros()
        try:
            ranker = _fit_scorer(
                Ym,
                model,
                n_components,
                seed,
                epochs_mf=epochs_mf,
                epochs_gnn=epochs_gnn,
                epochs_lightgcn=epochs_lightgcn,
                lightgcn_layers=lightgcn_layers,
                lr_lightgcn=lr_lightgcn,
                device=device,
            )
        except Exception:
            continue
        s_pos = float(ranker.scores_for_disease(j)[i])
        y_true.append(1)
        y_score.append(s_pos)

        row = Ym.getrow(i)
        present = set(row.indices.tolist())
        candidates = [c for c in range(n_d) if c not in present]
        if not candidates:
            continue
        take = min(n_neg, len(candidates))
        neg_js = rng.choice(candidates, size=take, replace=False)
        for jn in neg_js:
            y_true.append(0)
            y_score.append(float(ranker.scores_for_disease(int(jn))[i]))

    meta = {
        "eval_positives": len(eval_pairs),
        "total_positives": n_pos_total,
        "subsampled": subsampled,
    }
    return np.array(y_true, dtype=np.int32), np.array(y_score, dtype=np.float64), meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Folder with associations.csv, diseases.csv, lncrnas.csv (default: ./data full v3 ingest)",
    )
    p.add_argument(
        "--protocol",
        choices=("holdout", "loo"),
        default="holdout",
        help="holdout = one train/test edge split on the full graph (default); loo = leave-one-out",
    )
    p.add_argument("--train-fraction", type=float, default=0.85, help="Training edge fraction (holdout only)")
    p.add_argument("--n-components", type=int, default=32)
    p.add_argument("--n-neg", type=int, default=25, help="Random negatives per test positive (same lncRNA row)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--loo-threshold",
        type=int,
        default=500,
        help="loo only: if more positives than this, subsample (use --all-loo for every edge)",
    )
    p.add_argument("--all-loo", action="store_true", help="loo only: evaluate every positive (very slow)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    p.add_argument(
        "--model",
        choices=("hybrid", "svd", "mf", "gnn", "lightgcn"),
        default="hybrid",
        help="hybrid/svd=truncated SVD (+ co-oc); mf/gnn/lightgcn=PyTorch joint embeddings on train edges",
    )
    p.add_argument("--epochs-mf", type=int, default=120, help="MF training epochs (hold-out)")
    p.add_argument("--epochs-gnn", type=int, default=60, help="GNN training epochs (hold-out)")
    p.add_argument("--epochs-lightgcn", type=int, default=200, help="LightGCN training epochs (hold-out)")
    p.add_argument("--lightgcn-layers", type=int, default=3, help="LightGCN propagation layers")
    p.add_argument("--lr-lightgcn", type=float, default=0.001, help="LightGCN Adam LR")
    p.add_argument(
        "--loo-epochs-mf",
        type=int,
        default=12,
        help="MF epochs per LOO refit (keep small; LOO is expensive for learned models)",
    )
    p.add_argument("--loo-epochs-gnn", type=int, default=8, help="GNN epochs per LOO refit")
    p.add_argument("--loo-epochs-lightgcn", type=int, default=8, help="LightGCN epochs per LOO refit")
    p.add_argument("--device", type=str, default="cpu", help="PyTorch device, e.g. cpu or cuda")
    p.add_argument(
        "--ranking-report",
        action="store_true",
        help="Hold-out only: print MRR, HR@10, HR@50 vs all train-unlinked diseases per test lncRNA",
    )
    p.add_argument(
        "--roc-name",
        type=str,
        default=None,
        help="Output ROC PNG filename (default: holdout_roc.png or loo_roc.png)",
    )
    p.add_argument("--pr-name", type=str, default=None, help="Output PR PNG filename")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    data_dir = _resolve_data_dir(args.data_dir)
    bp = load_bipartite(data_dir)
    Y = bp.Y.tocsr()

    if args.protocol == "holdout":
        y_true_arr, y_score_arr, meta = run_holdout(
            Y,
            n_components=args.n_components,
            model=args.model,
            seed=args.seed,
            train_fraction=args.train_fraction,
            n_neg=args.n_neg,
            rng=rng,
            epochs_mf=args.epochs_mf,
            epochs_gnn=args.epochs_gnn,
            epochs_lightgcn=args.epochs_lightgcn,
            lightgcn_layers=args.lightgcn_layers,
            lr_lightgcn=args.lr_lightgcn,
            device=args.device,
            ranking_report=args.ranking_report,
        )
        title_roc = f"Hold-out ROC ({args.model}, {meta['n_train_edges']} train / {meta['n_test_edges']} test edges)"
        title_pr = f"Hold-out precision–recall ({args.model})"
        footer = (
            f"protocol=holdout  model={args.model}  data={data_dir.name}  "
            f"train_edges={meta['n_train_edges']}  test_edges={meta['n_test_edges']}  n_neg/pos={args.n_neg}"
        )
        roc_name = args.roc_name or "holdout_roc.png"
        pr_name = args.pr_name or "holdout_pr.png"
    else:
        y_true_arr, y_score_arr, meta = run_loo(
            Y,
            n_components=args.n_components,
            model=args.model,
            seed=args.seed,
            n_neg=args.n_neg,
            loo_threshold=None if args.all_loo else args.loo_threshold,
            all_loo=args.all_loo,
            rng=rng,
            epochs_mf=args.loo_epochs_mf,
            epochs_gnn=args.loo_epochs_gnn,
            epochs_lightgcn=args.loo_epochs_lightgcn,
            lightgcn_layers=args.lightgcn_layers,
            lr_lightgcn=args.lr_lightgcn,
            device=args.device,
        )
        title_roc = f"Leave-one-out ROC ({args.model})"
        title_pr = f"Leave-one-out precision–recall ({args.model})"
        footer = (
            f"protocol=loo  model={args.model}  data={data_dir.name}  "
            f"eval_pos={meta['eval_positives']}/{meta['total_positives']}  n_neg/pos={args.n_neg}"
            + ("  [subsampled]" if meta.get("subsampled") else "")
        )
        roc_name = args.roc_name or "loo_roc.png"
        pr_name = args.pr_name or "loo_pr.png"

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
    ax.set_title(title_roc)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.text(0.02, 0.02, footer, fontsize=7, color="#444")
    fig.tight_layout()
    roc_path = args.out_dir / roc_name
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_true_arr, y_score_arr)
    fig2, ax2 = plt.subplots(figsize=(5, 4.5))
    ax2.plot(rec, prec, lw=2, label=f"AUPR = {aupr:.3f}")
    baseline = y_true_arr.mean()
    ax2.axhline(baseline, color="gray", ls="--", lw=1, label=f"baseline = {baseline:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(title_pr)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    fig2.text(0.02, 0.02, footer, fontsize=7, color="#444")
    fig2.tight_layout()
    pr_path = args.out_dir / pr_name
    fig2.savefig(pr_path, dpi=150)
    plt.close(fig2)

    print(f"AUROC={auroc:.4f}  AUPR={aupr:.4f}")
    print(f"Wrote {roc_path} and {pr_path}")
    if args.protocol == "holdout" and args.ranking_report and "mrr" in meta:
        print(
            f"Ranking (held-out positives vs train-unlinked diseases): "
            f"MRR={meta['mrr']:.4f}  HR@10={meta['hr10']:.4f}  HR@50={meta['hr50']:.4f}  (n={meta['n_ranked']})"
        )


if __name__ == "__main__":
    main()
