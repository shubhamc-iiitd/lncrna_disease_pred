#!/usr/bin/env python3
"""Generate synthetic demo graph (not from LncRNADisease).

For real data use:
  python scripts/fetch_lncrnadisease_v30.py
  python scripts/ingest_lncrnadisease_v30.py

Original purpose: small bipartite graph with latent block structure.

Edges are noisy superpositions of category–category affinities so matrix
factorization / message passing can outperform random baselines, while
category labels let us audit top predictions for annotation-style clustering.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    categories = ["Cancer", "Neurological", "Cardiovascular", "Metabolic", "Immune", "Other"]
    n_dis = 48
    n_lnc = 140

    disease_rows = []
    for i in range(n_dis):
        cat = categories[i % len(categories)]
        disease_rows.append(
            {
                "disease_id": f"D{i:04d}",
                "disease_name": f"Demo disease {i} ({cat})",
                "category": cat,
            }
        )
    diseases = pd.DataFrame(disease_rows)

    lnc_rows = [{"lncrna_id": f"LNC{i:04d}", "lncrna_name": f"Demo lncRNA {i}"} for i in range(n_lnc)]
    lncs = pd.DataFrame(lnc_rows)

    # Latent "topics" for diseases (soft block by index band)
    K = 8
    theta_d = rng.normal(size=(n_dis, K))
    # lncRNAs drawn near mixture of a few disease-topic vectors
    topic_pref = rng.integers(0, K, size=(n_lnc, 3))
    theta_l = np.zeros((n_lnc, K))
    for i in range(n_lnc):
        for t in topic_pref[i]:
            theta_l[i, t] += 1.0
    theta_l += rng.normal(scale=0.15, size=(n_lnc, K))

    logits = theta_l @ theta_d.T
    probs = 1 / (1 + np.exp(-logits))
    adj = rng.random(size=probs.shape) < probs
    # ensure minimum degree
    for i in range(n_lnc):
        if adj[i].sum() == 0:
            j = int(rng.integers(0, n_dis))
            adj[i, j] = True
    for j in range(n_dis):
        if adj[:, j].sum() == 0:
            i = int(rng.integers(0, n_lnc))
            adj[i, j] = True

    li, dj = np.where(adj)
    assoc = pd.DataFrame(
        {
            "lncrna_id": [f"LNC{i:04d}" for i in li],
            "disease_id": [f"D{j:04d}" for j in dj],
        }
    )
    assoc = assoc.merge(lncs, on="lncrna_id", how="left").merge(diseases, on="disease_id", how="left")

    diseases.to_csv(out / "diseases.csv", index=False)
    lncs.to_csv(out / "lncrnas.csv", index=False)
    assoc.to_csv(out / "associations.csv", index=False)
    print(f"Wrote {len(assoc)} edges, {n_lnc} lncRNAs, {n_dis} diseases -> {out}")


if __name__ == "__main__":
    main()
