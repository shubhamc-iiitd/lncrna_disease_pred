# lncRNA–disease prediction (`lncrna_disease_pred`)

Predict and explore **lncRNA–disease** links by treating curated databases as a **bipartite graph**. This repository ingests **[LncRNADisease v3.0](http://www.rnanut.net/lncrnadisease/)**, learns **joint embeddings** (closed-form baselines, matrix factorization, a small GNN, or **bipartite LightGCN** in pure PyTorch), evaluates **held-out** and **leave-one-out** link prediction with **ROC / PR** and **ranking metrics**, and ships a **Flask** portal for browsing ranked candidates on the **full** graph.

**License:** [GNU General Public License v3.0](LICENSE).

---

## Table of contents

1. [Quick start](#quick-start)
2. [Data: LncRNADisease v3.0](#data-lncrnadisease-v30)
3. [Models (joint embeddings)](#models-joint-embeddings)
4. [Evaluation & figures](#evaluation--figures)
5. [Biology vs annotation bias](#biology-vs-annotation-bias)
6. [Web portal (Flask)](#web-portal-flask)
7. [Repository layout](#repository-layout)
8. [Citation](#citation)

---

## Quick start

```bash
git clone https://github.com/shubhamc-iiitd/lncrna_disease_pred.git
cd lncrna_disease_pred

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Build the human lncRNA graph from v3.0 (writes ./data/)
python scripts/fetch_lncrnadisease_v30.py
python scripts/ingest_lncrnadisease_v30.py

# Optional: train LightGCN on the full graph for the Flask UI (~minutes on CPU)
python scripts/train_lightgcn_full.py --data-dir data --epochs 250 --dim 64 --layers 3

python run_flask.py
```

Open **http://127.0.0.1:5000**. If `checkpoints/lightgcn_full.pt` exists, rankings use **LightGCN**; otherwise the app uses a **hybrid SVD + co-occurrence** baseline.

**Minimal demo without ingest:** the app falls back to `examples/minimal_data/` when `./data/associations.csv` is missing.

---

## Data: LncRNADisease v3.0

| Script | Purpose |
|--------|---------|
| [`scripts/fetch_lncrnadisease_v30.py`](scripts/fetch_lncrnadisease_v30.py) | Downloads official bulk files (default: `website_simple_data.csv`) into `data/raw/`. |
| [`scripts/ingest_lncrnadisease_v30.py`](scripts/ingest_lncrnadisease_v30.py) | Builds **`data/associations.csv`**, **`diseases.csv`**, **`lncrnas.csv`**: **Homo sapiens**, **LncRNA** rows only, unique edges, stable disease IDs, and **keyword disease categories** (for bias summaries; not MeSH). |

**Loader:** [`src/dataio.py`](src/dataio.py) → sparse binary matrix `Y` (lncRNAs × diseases).

Large raw downloads and optional full `data/*.csv` copies may be omitted from git (see [`.gitignore`](.gitignore)); regenerate locally with the scripts above.

---

## Models (joint embeddings)

All scoring uses **dot products** (or equivalent logits) between **lncRNA** and **disease** vectors in a shared latent space, except where noted.

| Model | Flag / location | Description |
|--------|-----------------|-------------|
| **Hybrid** | `flask_tool/ranker.py`, `--model hybrid` | Truncated SVD + deterministic **3-path** co-occurrence `Y @ Yᵀ @ Y`, column-wise z-score blend. Fast, no PyTorch training. |
| **SVD** | `--model svd` | Truncated SVD only. |
| **Logistic MF** | `--model mf`, `src/models_mf.py`, `src/learned_edge_models.py` | Learned embeddings, BCE on train edges + negatives. |
| **Tiny bipartite GNN** | `--model gnn`, `src/models_gnn.py` | Two-hop tanh message passing; dot scores. |
| **Bipartite LightGCN** | `--model lightgcn`, `src/lightgcn_bipartite.py` | LightGCN-style **linear** propagation on symmetric-normalized `R`; only **layer-0** embeddings trained; **mean** of layer embeddings; **no PyG/DGL**. Full-graph training: [`scripts/train_lightgcn_full.py`](scripts/train_lightgcn_full.py) → `checkpoints/lightgcn_full.pt` (gitignored). |

**Evaluation driver:** [`scripts/eval_loo_link_prediction.py`](scripts/eval_loo_link_prediction.py) — `--protocol holdout` (default) or `loo`; `--model …`; optional `--ranking-report` (MRR, HR@10, HR@50).

---

## Evaluation & figures

### Hold-out link prediction (recommended)

Random **edge-level** split (default **85% train / 15% test**, seed **42**). Train **only** on `Y_train`; score **test positives** vs random **same-row negatives** (diseases with no train edge to that lncRNA). **ROC** and **precision–recall** are saved under [`figures/`](figures/).

**Reference numbers** on the full v3 ingest (same split, seed 42):

| Model | AUROC | AUPR | MRR | HR@10 | HR@50 |
|--------|-------|------|-----|-------|-------|
| Hybrid | 0.427 | 0.034 | 0.007 | 0.3% | 1.7% |
| LightGCN (dim 64, 3 layers, 120 epochs) | **0.734** | **0.282** | **0.201** | **41%** | **62%** |

`--ranking-report` ranks each held-out disease **j** among all diseases **not** linked to **i** in training (hundreds of candidates per row). Under extreme sparsity, **PR-AUC** and **ranking** are often more informative than ROC alone.

<p align="center">
  <img src="figures/holdout_roc.png" alt="Hold-out ROC hybrid" width="45%" />
  <img src="figures/holdout_lightgcn_roc.png" alt="Hold-out ROC LightGCN" width="45%" /><br/>
  <img src="figures/holdout_pr.png" alt="Hold-out PR hybrid" width="45%" />
  <img src="figures/holdout_lightgcn_pr.png" alt="Hold-out PR LightGCN" width="45%" />
</p>

**Regenerate:**

```bash
python scripts/eval_loo_link_prediction.py --data-dir data --protocol holdout --model hybrid --seed 42 \
  --roc-name holdout_roc.png --pr-name holdout_pr.png --out-dir figures

python scripts/eval_loo_link_prediction.py --data-dir data --protocol holdout --model lightgcn --seed 42 \
  --n-components 64 --lightgcn-layers 3 --epochs-lightgcn 120 --ranking-report \
  --roc-name holdout_lightgcn_roc.png --pr-name holdout_lightgcn_pr.png --out-dir figures
```

### Leave-one-out (subsampled)

One refit per held-out positive is expensive. Example: **`--loo-threshold 350`** (figures in `figures/loo_*.png`). Use `--all-loo` only on small graphs.

```bash
python scripts/eval_loo_link_prediction.py --data-dir data --protocol loo --loo-threshold 350 --seed 42 --out-dir figures
```

---

## Biology vs annotation bias

| Script | Question |
|--------|----------|
| [`scripts/category_bias_audit.py`](scripts/category_bias_audit.py) | Do **top-scoring novel (absent) edges** cluster in the same **keyword disease categories** as curated positives (literature depth)? |
| [`scripts/hub_degree_audit.py`](scripts/hub_degree_audit.py) | Are top novel pairs **hub–hub** (high lncRNA × disease degree)? Compare to a random absent-edge null; optional `--checkpoint` for LightGCN scores. |

<p align="center">
  <img src="figures/category_novel_enrichment.png" alt="Category mix" width="48%" />
  <img src="figures/category_novel_enrichment_fold.png" alt="Fold vs positives" width="48%" />
</p>

```bash
python scripts/category_bias_audit.py --data-dir data --top-k 3000 --out-figure figures/category_novel_enrichment.png
python scripts/hub_degree_audit.py --data-dir data --top-k 3000
```

---

## Web portal (Flask)

| Entry | Command |
|-------|---------|
| Dev server | `python run_flask.py` |
| Flask CLI | `export PYTHONPATH="$PWD" && flask --app wsgi run --debug` |

**Data directory:** uses `./data/` when `data/associations.csv` exists; else `examples/minimal_data/`. Override with **`LNC_DATA_DIR`**.

**LightGCN checkpoint:** if **`checkpoints/lightgcn_full.pt`** exists, it is loaded by default. Set **`LNC_USE_LIGHTGCN=0`** to force the hybrid ranker. **`LNC_LIGHTGCN_CKPT`**: custom path. **`LNC_TORCH_DEVICE`**: e.g. `cuda` or `cpu`.

---

## Repository layout

| Path | Role |
|------|------|
| `run_flask.py`, `wsgi.py` | Application entrypoints |
| `flask_tool/` | Flask app, templates, static assets |
| `src/dataio.py` | CSV → sparse `Y` |
| `src/lightgcn_bipartite.py` | LightGCN train / checkpoint I/O |
| `src/learned_edge_models.py`, `src/models_mf.py`, `src/models_gnn.py` | MF / tiny GNN training |
| `flask_tool/ranker.py` | Hybrid + SVD baselines for the portal |
| `scripts/fetch_*.py`, `scripts/ingest_*.py` | v3.0 pipeline |
| `scripts/eval_loo_link_prediction.py` | Hold-out / LOO / ranking |
| `scripts/train_lightgcn_full.py` | Full-graph LightGCN → checkpoint |
| `scripts/category_bias_audit.py`, `scripts/hub_degree_audit.py` | Bias audits |
| `examples/minimal_data/` | Tiny bundled graph |
| `figures/` | ROC/PR and category plots |
| `checkpoints/` | Trained `.pt` (default gitignored; keep `checkpoints/.gitkeep`) |
| `requirements.txt`, `environment.yml` | Dependencies |
| `LICENSE` | GPL-3.0 |

---

## Citation

If you use **LncRNADisease v3.0** data, cite the database and its publication (e.g. Zhang *et al.*, *Nucleic Acids Research*, [PMC10767967](https://pmc.ncbi.nlm.nih.gov/articles/PMC10767967/)), per the [official site](http://www.rnanut.net/lncrnadisease/index.php/home/info/download).

---

## Contributing / upstream

**Remote:** `https://github.com/shubhamc-iiitd/lncrna_disease_pred.git`

```bash
git add .
git commit -m "Your message"
git push origin main
```
