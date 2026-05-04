# lncRNA–disease link exploration

Small Flask tool to browse **LncRNADisease-style** bipartite graphs: pick a disease and inspect **ranked lncRNA candidates** from a low-rank factorization (truncated SVD) of the association matrix. Scripts download **LncRNADisease v3.0**, ingest it to CSVs, and evaluate **leave-one-out** link prediction with ROC and PR curves.

---

## What we built (step by step)

1. **Problem framing** — Treat an lncRNA–disease resource as a **bipartite graph** (lncRNAs × diseases, edges = reported associations).

2. **LncRNADisease v3.0 integration**
   - **`scripts/fetch_lncrnadisease_v30.py`** — Downloads official bulk files from [LncRNADisease3](http://www.rnanut.net/lncrnadisease/index.php/home/info/download) (default: `website_simple_data.csv`).
   - **`scripts/ingest_lncrnadisease_v30.py`** — Builds `associations.csv`, `diseases.csv`, and `lncrnas.csv`: human **Homo sapiens** + **LncRNA** rows, unique edges, stable disease IDs, and coarse **keyword disease categories** (for bias-style summaries; not MeSH).

3. **Data loader** — **`src/dataio.py`** loads the three CSVs into a sparse **0/1 incidence matrix** `Y` plus ID/name/category maps.

4. **Ranking model (portal)** — **`flask_tool/ranker.py`** fits **truncated SVD** on `Y` (same idea as logistic MF / LSA on the bipartite adjacency). **`flask_tool/app.py`** serves a **Flask** UI: choose a disease, call `/api/rank`, see ranked lncRNAs. **Important:** the live portal refits on the **full** loaded graph for exploration (not a held-out split).

5. **Synthetic demo generator** — **`scripts/generate_demo_data.py`** can emit a toy bipartite graph (optional); real runs use v3.0 or **`examples/minimal_data/`**.

6. **Runnable portal** — **`run_flask.py`** and **`wsgi.py`**, templates under **`flask_tool/templates/`**, styles in **`flask_tool/static/`**.

7. **Leave-one-out evaluation** — **`scripts/eval_loo_link_prediction.py`** (see below): for each evaluated positive edge, remove it from `Y`, **refit SVD** on the masked matrix, score the held-out pair vs **random same-row negatives** (other diseases with no edge to that lncRNA under the mask), pool labels, compute **AUROC** and **AUPR**, and save **`figures/loo_roc.png`** and **`figures/loo_pr.png`**.

8. **Repository hygiene** — **`requirements.txt`**, **`environment.yml`**, **`.gitignore`** (large raw downloads and optional full `data/*.csv` graphs), and this **README** for GitHub.

Cite the **LncRNADisease v3.0** database and its paper when using their downloads (e.g. Zhang *et al.*, *NAR*, [PMC10767967](https://pmc.ncbi.nlm.nih.gov/articles/PMC10767967/)).

---

## Leave-one-out AUROC and AUPR (masked SVD)

**Protocol.** For each positive edge \((i, j)\) in the evaluation set: set \(Y_{ij} \leftarrow 0\), fit truncated SVD on the masked \(Y\), take the latent **score** for \((i,j)\) as one **positive** score, draw **`--n-neg`** **negative** scores from the same lncRNA row \(i\) at columns \(j'\) with \(Y_{ij'}=0\) under the mask. Concatenate all positives and negatives, compute **AUROC** (`sklearn.metrics.roc_auc_score`) and **AUPR** (`average_precision_score`), and plot pooled ROC and precision–recall curves.

**Figures in this repo** (generated on the bundled **`examples/minimal_data`** graph; small graphs give **noisy** curves and are for **pipeline illustration** only):

| Metric (minimal demo) | Value |
|----------------------|-------|
| AUROC | 0.556 |
| AUPR | 0.279 |

![Leave-one-out ROC](figures/loo_roc.png)

![Leave-one-out precision–recall](figures/loo_pr.png)

**Regenerate** (after `pip install -r requirements.txt`):

```bash
python scripts/eval_loo_link_prediction.py --data-dir examples/minimal_data --out-dir figures
```

**Full v3.0 graph:** leave-one-out with **one SVD refit per positive** is expensive (many thousands of fits). The script **subsamples** to **400** positives by default when the graph has more than **400** edges (override with **`--all-loo`** if you accept a long run, or raise **`--loo-threshold`**):

```bash
python scripts/eval_loo_link_prediction.py --data-dir data --out-dir figures --n-components 32 --n-neg 25
```

---

## Quick start (clone from GitHub)

```bash
git clone <your-repo-url>
cd huang_assignment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_flask.py
```

Open **http://127.0.0.1:5000** in a browser.

Alternative (Flask CLI):

```bash
export PYTHONPATH="$PWD"
flask --app wsgi run --debug
```

By default the app loads the tiny bundled graph under `examples/minimal_data/`. To use your own tables, set:

```bash
export LNC_DATA_DIR=/absolute/path/to/folder_with_three_csvs
python run_flask.py
```

The folder must contain `associations.csv`, `diseases.csv`, and `lncrnas.csv` (see `src/dataio.py` for the expected columns).

## LncRNADisease v3.0 data

```bash
pip install -r requirements.txt
python scripts/fetch_lncrnadisease_v30.py
python scripts/ingest_lncrnadisease_v30.py
export LNC_DATA_DIR="$PWD/data"
python run_flask.py
```

## Repository layout

| Path | Role |
|------|------|
| `run_flask.py` | Start the Flask server |
| `wsgi.py` | `flask --app wsgi run` |
| `flask_tool/` | App, templates, static assets, SVD ranker |
| `scripts/` | Fetch + ingest v3.0, LOO eval, optional synthetic data |
| `src/dataio.py` | Load CSVs into a sparse bipartite matrix |
| `examples/minimal_data/` | Demo graph for out-of-the-box runs and LOO figures |
| `figures/loo_roc.png`, `figures/loo_pr.png` | Leave-one-out ROC / PR plots (regenerate with `scripts/eval_loo_link_prediction.py`) |

## Push to GitHub

```bash
git init
git add .
git commit -m "Add Flask lncRNA–disease portal and LncRNADisease v3 ingest"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

## License

Add your own `LICENSE` file before publishing on GitHub if needed.
