#!/usr/bin/env python3
"""Train bipartite LightGCN on the **full** ingested graph and save a Flask checkpoint.

Writes ``checkpoints/lightgcn_full.pt`` (by default). The web app loads this file
when present so rankings use the same joint embeddings on the whole dataset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataio import load_bipartite  # noqa: E402
from src.lightgcn_bipartite import save_lightgcn_checkpoint, train_lightgcn  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "lightgcn_full.pt")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not (args.data_dir / "associations.csv").is_file():
        raise SystemExit(f"Missing {args.data_dir}/associations.csv — run ingest first.")

    bp = load_bipartite(args.data_dir)
    print(f"Training LightGCN on full graph {bp.Y.shape[0]} × {bp.Y.shape[1]}, nnz={bp.Y.nnz} …")
    _, meta = train_lightgcn(
        bp.Y.tocsr(),
        dim=args.dim,
        n_layers=args.layers,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    save_payload = {
        "n_l": meta["n_l"],
        "n_d": meta["n_d"],
        "dim": meta["dim"],
        "n_layers": meta["n_layers"],
        "model_state": meta["model_state"],
    }
    save_lightgcn_checkpoint(save_payload, args.out)
    print(f"Saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
