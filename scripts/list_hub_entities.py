#!/usr/bin/env python3
"""Print highest-degree lncRNAs and diseases in the bipartite graph (hub audit).

Reads the same CSV layout as ``src.dataio.load_bipartite``. Use this to reproduce
the hub tables in the README after fetch + ingest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataio import load_bipartite  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--top", type=int, default=25, help="How many rows per side")
    args = p.parse_args()

    if not (args.data_dir / "associations.csv").is_file():
        raise SystemExit(f"Missing ingest under {args.data_dir} (associations.csv).")

    bp = load_bipartite(args.data_dir)
    Y = bp.Y.tocsr()
    deg_l = np.asarray(Y.sum(axis=1)).ravel().astype(int)
    deg_d = np.asarray(Y.sum(axis=0)).ravel().astype(int)
    n_l, n_d = Y.shape
    top = min(args.top, n_l, n_d)

    print("=== Graph summary ===")
    print(f"lncRNAs: {n_l}, diseases: {n_d}, edges: {int(Y.nnz)}")
    print(f"lncRNA degree: max={deg_l.max()}, median={np.median(deg_l):.0f}, mean={deg_l.mean():.2f}")
    print(f"disease degree: max={deg_d.max()}, median={np.median(deg_d):.0f}, mean={deg_d.mean():.2f}")

    print("\n=== Top lncRNAs by # of disease associations ===\n")
    for rank, i in enumerate(np.argsort(-deg_l)[:top], 1):
        lid = bp.lnc_ids[i]
        name = bp.lnc_names.get(lid, lid)
        print(f"{rank:3d}. {deg_l[i]:4d}  {lid}  |  {name}")

    print("\n=== Top diseases by # of lncRNA associations ===\n")
    for rank, j in enumerate(np.argsort(-deg_d)[:top], 1):
        did = bp.disease_ids[j]
        dname = bp.disease_names.get(did, did)
        cat = bp.disease_category.get(did, "?")
        print(f"{rank:3d}. {deg_d[j]:4d}  {did}  |  {dname}  [{cat}]")


if __name__ == "__main__":
    main()
