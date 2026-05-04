#!/usr/bin/env python3
"""Build bipartite tables (associations.csv, diseases.csv, lncrnas.csv) from LncRNADisease v3.0.

Expects `data/raw/website_simple_data.csv` from `fetch_lncrnadisease_v30.py`.

Filters (defaults): human (Homo sapiens) and ncRNA_Category == LncRNA so the
graph matches lncRNA–disease experimental support in v3.0.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def disease_bucket(name: str) -> str:
    """Coarse disease class from free-text name (no MeSH tree in simple CSV)."""
    n = name.lower()
    onco = (
        "neoplasm",
        "carcinoma",
        "cancer",
        "tumor",
        "tumour",
        "sarcoma",
        "lymphoma",
        "leukemia",
        "leukaemia",
        "melanoma",
        "glioma",
        "glioblastoma",
        "adenocarcinoma",
        "myeloma",
        "blastoma",
        "cholangiocarcinoma",
    )
    if any(k in n for k in onco):
        return "Neoplasm"
    neuro = (
        "alzheimer",
        "parkinson",
        "epilepsy",
        "neuro",
        "schizophrenia",
        "amyotrophic",
        "huntington",
        "dementia",
    )
    if any(k in n for k in neuro):
        return "Neurological_or_vascular_brain"
    cardio = (
        "cardio",
        "heart",
        "coronary",
        "atherosclerosis",
        "hypertension",
        "myocardial",
        "atrial",
        "heart failure",
    )
    if any(k in n for k in cardio):
        return "Cardiovascular"
    meta = ("diabetes", "obesity", "metabolic", "insulin", "glucose", "lipid")
    if any(k in n for k in meta):
        return "Metabolic"
    imm = ("arthritis", "lupus", "immune", "inflammation", "inflammatory", "autoimmune", "psoriasis", "asthma")
    if any(k in n for k in imm):
        return "Immune_inflammatory"
    if "hepatitis" in n and "alcoholic" in n:
        return "Other"
    inf = ("infection", "tuberculosis", "sepsis", "covid", "influenza", "viral hepatitis")
    if "hiv" in n or any(k in n for k in inf) or ("hepatitis" in n and "viral" in n):
        return "Infectious"
    return "Other"


def stable_disease_id(name: str) -> str:
    h = hashlib.sha1(name.strip().encode("utf-8")).hexdigest()[:14]
    return f"D{h}"


def lnc_id(species: str, symbol: str) -> str:
    sp = "HS" if species.strip().lower() == "homo sapiens" else species.replace(" ", "_")[:8]
    return f"{sp}:{symbol.strip()}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--simple-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw" / "website_simple_data.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    p.add_argument("--species", default="Homo sapiens", help="Species filter (exact match in file)")
    p.add_argument("--category", default="LncRNA", help="ncRNA_Category filter (exact match)")
    args = p.parse_args()
    if not args.simple_csv.is_file():
        raise SystemExit(f"Missing input file: {args.simple_csv}\nRun: python scripts/fetch_lncrnadisease_v30.py")

    pairs: set[tuple[str, str]] = set()
    disease_meta: dict[str, tuple[str, str]] = {}
    lnc_meta: dict[str, str] = {}

    with open(args.simple_csv, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        expected = {"Database_ID", "ncRNA_Symbol", "ncRNA_Category", "Species", "Disease_Name"}
        if not r.fieldnames or not expected.issubset(set(r.fieldnames)):
            raise SystemExit(f"Unexpected CSV columns: {r.fieldnames}")
        for row in r:
            if row["Species"] != args.species or row["ncRNA_Category"] != args.category:
                continue
            dname = row["Disease_Name"].strip().strip('"')
            if not dname:
                continue
            sym = row["ncRNA_Symbol"].strip()
            if not sym:
                continue
            did = stable_disease_id(dname)
            lid = lnc_id(row["Species"], sym)
            disease_meta[did] = (dname, disease_bucket(dname))
            lnc_meta[lid] = sym
            pairs.add((lid, did))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    assoc_path = args.out_dir / "associations.csv"
    with open(assoc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lncrna_id", "disease_id", "lncrna_name", "disease_name", "category"])
        for lid, did in sorted(pairs):
            dname, cat = disease_meta[did]
            w.writerow([lid, did, lnc_meta[lid], dname, cat])

    with open(args.out_dir / "diseases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["disease_id", "disease_name", "category"])
        for did in sorted(disease_meta):
            dname, cat = disease_meta[did]
            w.writerow([did, dname, cat])

    with open(args.out_dir / "lncrnas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lncrna_id", "lncrna_name"])
        for lid in sorted(lnc_meta):
            w.writerow([lid, lnc_meta[lid]])

    print(
        f"Wrote {len(pairs)} unique edges, {len(lnc_meta)} lncRNAs, {len(disease_meta)} diseases -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()