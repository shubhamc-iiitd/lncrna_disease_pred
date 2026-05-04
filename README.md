# lncRNA–disease prediction (`lncrna_disease_pred`)

This project turns **[LncRNADisease v3.0](http://www.rnanut.net/lncrnadisease/)** into a simple **network**: lncRNAs on one side, diseases on the other, and lines where the database says they are linked. It then **learns numeric profiles** for each node (several methods, including **LightGCN**), **tests** how well missing links can be guessed when you hide some known links, and offers a small **Flask** website to browse suggested pairs on the **full** graph.

**License:** [GNU General Public License v3.0](LICENSE).

---

## Table of contents

1. [Quick start](#quick-start)
2. [Data: LncRNADisease v3.0](#data-lncrnadisease-v30)
3. [Models (joint embeddings)](#models-joint-embeddings)
4. [The biological question](#the-biological-question-graph-structure-vs-annotation-bias)
5. [Evaluation & figures](#evaluation--figures)
6. [Biology vs annotation bias (scripts)](#biology-vs-annotation-bias-scripts)
7. [Web portal (Flask)](#web-portal-flask)
8. [Repository layout](#repository-layout)
9. [Citation](#citation)

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

Each lncRNA and each disease gets a **vector of numbers**. A high **dot product** (or related score) means “more likely linked.”

| Model | Flag / location | In plain terms |
|--------|-----------------|----------------|
| **Hybrid** | `flask_tool/ranker.py`, `--model hybrid` | Fast baseline: matrix factorization–style SVD plus a **shared-neighbour** count; no neural training. |
| **SVD** | `--model svd` | SVD-only version of the above idea. |
| **Logistic MF** | `--model mf`, `src/models_mf.py`, `src/learned_edge_models.py` | **Neural** two-sided embeddings trained with yes/no loss on real vs fake edges. |
| **Tiny GNN** | `--model gnn`, `src/models_gnn.py` | **Neural**; passes messages along the graph for two steps, then scores with a dot product. |
| **LightGCN** | `--model lightgcn`, `src/lightgcn_bipartite.py` | **Neural**, LightGCN-style: smooth embeddings along links, no heavy MLP. Implemented in **plain PyTorch** (no PyG). Train on the full graph with [`scripts/train_lightgcn_full.py`](scripts/train_lightgcn_full.py) → `checkpoints/lightgcn_full.pt` (not in git by default). |

**One script runs the tests:** [`scripts/eval_loo_link_prediction.py`](scripts/eval_loo_link_prediction.py) — use `--protocol holdout` (default) or `loo`, pick `--model`, and add `--ranking-report` for rank-based metrics.

---

## The biological question (graph structure vs annotation bias)

We are not chasing one magic accuracy score. We want to know two simpler things:

**1. Does the *shape* of the database actually help you guess missing links?**  
Imagine you **hide one known link**, **retrain** your scorer on what is left, and then ask: “Does the model rank that hidden link **above** random non-links?” **Leave-one-out (LOO)** does exactly that, one link at a time. The **ROC** and **PR** curves summarize how often that works. If the curves look strong, the **connections already in the graph** carry useful signal. If they look flat, the model is **not** finding an easy pattern in this network alone—that can mean biology is messy, the data is noisy, or the model is too simple. It does **not** by itself prove there is “no biology.”  

Doing LOO on every link is slow for a big graph. This repo includes **LOO on a random subset** in `figures/loo_roc.png` and `figures/loo_pr.png`. For day-to-day use we also use a **hold-out split** (hide a chunk of links at once), which is faster and tells a similar story—see [Evaluation & figures](#evaluation--figures).

<p align="center">
  <img src="figures/loo_roc.png" alt="Leave-one-out ROC" width="45%" />
  <img src="figures/loo_pr.png" alt="Leave-one-out PR" width="45%" />
</p>

**2. Are our top “new” guesses just chasing what is already over-studied?**  
Some lncRNAs and diseases appear everywhere in the literature (**hubs**). Cancer-style entries also dominate many databases. Take the **highest-scoring pairs that are *not* in the database**. If those pairs **pile into the same disease buckets** (or the same super-connected nodes) as the **training links**, the model may mostly be learning **“what gets studied a lot”**—**annotation bias** and **popularity**—not a new biological rule. If the pattern **looks different**, that is a hint the model is **not only** copying the obvious skew (but **be careful**: real biology also clusters, and our “categories” are **rough keyword labels**, not a full disease ontology).  

Scripts and plots for this second check are in [Biology vs annotation bias (scripts)](#biology-vs-annotation-bias-scripts).

**In short:** the **LOO / hold-out curves** ask whether **the graph supports prediction**. The **category and hub checks** ask whether **top guesses mirror how lopsided the database is**. You want both: neither answer is final on its own, but together they separate **“structure in the data”** from **“bias in how the data was collected.”**

---

## Evaluation & figures

### Hold-out link prediction (recommended)

We randomly put **85%** of known links in **training** and **15%** in a **test** set (default seed **42**). The model never sees the test links during training. We then score each **test link** against random **decoy** diseases for the same lncRNA and plot **ROC** and **precision–recall** in [`figures/`](figures/).

**Reference numbers** on the full v3 ingest (same split, seed 42):

| Model | AUROC | AUPR | MRR | HR@10 | HR@50 |
|--------|-------|------|-----|-------|-------|
| Hybrid | 0.427 | 0.034 | 0.007 | 0.3% | 1.7% |
| LightGCN (dim 64, 3 layers, 120 epochs) | **0.734** | **0.282** | **0.201** | **41%** | **62%** |

With `--ranking-report` we also ask: for each hidden link, **what rank** is the true disease among all diseases **not** already linked to that lncRNA in training? When almost everything is a “no link,” **PR** and these **rank** numbers are often easier to read than ROC alone.

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

## Biology vs annotation bias (scripts)

These scripts support **question 2** above: “Are we just rediscovering busy topics and hub nodes?”

| Script | What it does |
|--------|----------------|
| [`scripts/category_bias_audit.py`](scripts/category_bias_audit.py) | Looks at the **disease type mix** among the **top predicted new links** and compares it to (a) real links in the data and (b) all diseases. |
| [`scripts/hub_degree_audit.py`](scripts/hub_degree_audit.py) | Compares **how “connected”** those top new links are (lncRNA degree × disease degree) to **random** non-links. High scores may mean the model likes **celebrity** nodes. Optional `--checkpoint` uses **LightGCN** scores instead of the hybrid model. |

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
