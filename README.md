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

**One script runs the tests:** [`scripts/eval_loo_link_prediction.py`](scripts/eval_loo_link_prediction.py) — default **`--protocol holdout`** gives **complete-dataset** train/test coverage of all edges; optional **`loo`** for stricter per-edge checks (see [Evaluation & figures](#evaluation--figures)). Add **`--ranking-report`** for MRR / HR@K on hold-out.

---

## The biological question (graph structure vs annotation bias)

This report develops **two lines of evidence**, both on the **full** ingested graph where possible: **(A)** **quantitative recovery** of **masked** edges from the remaining structure (hold-out link prediction), and **(B)** **alignment** of top-scoring **absent** pairs with **hub popularity** and **literature-heavy keyword buckets** compared to random absent pairs and to the positive-edge distribution.

### A. Link prediction protocol (hold-out on the complete edge list)

The **primary** evaluation **shuffles all known edges**, assigns default **85%** to **training** and **15%** to **test**, fits **one** model on the training edges, and scores **every** test edge against random decoys for the same lncRNA. **ROC**, **precision–recall**, and (with `--ranking-report`) **MRR / HR@K** are computed over the **entire test split**. Every positive in `data/associations.csv` sits in **exactly one** of train or test for that run (default seed **42**). Curves and the reference table are in [Evaluation & figures](#evaluation--figures).

**Leave-one-out (LOO)** removes **one** positive, refits, scores, and repeats. Full coverage requires **`--all-loo`** over **~10.5k** positives; with neural refits per edge that is usually **impractical**, so subsampled LOO is only an auxiliary check. **Hold-out** remains the headline protocol for complete-dataset numbers.

### Inference from link-prediction results

On the **full** v3 human ingest (**10,518** edges) with the default **85/15** split and seed **42**, the **hybrid** baseline shows **weak** discrimination (AUROC **~0.43**, AUPR **~0.03**) and **negligible ranking** of the true disease among held-out candidates (MRR **~0.007**, HR@10 **~0.3%**, HR@50 **~1.7%**). **Bipartite LightGCN** under the same protocol reaches AUROC **~0.73**, AUPR **~0.28**, MRR **~0.20**, HR@10 **~41%**, and HR@50 **~62%** (64-dim, 3 layers, 120 epochs as in the table below).

**Takeaway:** the **network alone** carries **substantial** edge signal for a suitable **learned** joint embedding, but **not** for the fast **SVD + shared-neighbour** hybrid under this strict masking. Strong **PR** and **ranking** gains relative to ROC reflect extreme class imbalance (almost all pairs are absent); they still indicate that **LightGCN** places many held-out positives in the **top few percent** of row-wise candidates. This supports **inference (i):** recoverable **graph regularities** exist in LncRNADisease v3 for human lncRNAs under edge supervision; **inference (ii):** capturing them **requires** a model that propagates structure **non-trivially**, not the hybrid baseline alone.

### B. Hubs, categories, and what top “novel” scores may reflect

The matrix is **sparse for most nodes** but **heavy-tailed**: a handful of lncRNAs (e.g. MALAT1, NEAT1, H19, …) and diseases (many **Neoplasm** rows, plus a few large non-cancer hubs) absorb a large share of edges. **Highest-scoring absent pairs** can therefore **enrich** for the same **keyword disease classes** or **degree product** as the positives simply because the training graph is **lopsided**, even when held-out **prediction** is good.

### Inference about annotation pressure and popularity bias

**Keyword categories** (especially **`Neoplasm`**) are **coarse**; overlap between top novel pairs and positive-edge categories is **consistent with both** concentrated cancer biology **and** **reporting bias**. The **category** and **hub-degree** scripts do not separate those causes; they **measure** how much top absent pairs resemble positives in category mix and in **lncRNA degree × disease degree** versus random absent pairs.

**Takeaway:** **LightGCN’s** strong held-out metrics **do not** imply that **Flask** top-k lists are free of **hub** or **annotation** effects on the **full** graph. **Inference (iii):** treat **global** ranking on the complete matrix as **complementary** to hold-out metrics—use the audits to **bound** how much top scores **track celebrity endpoints** and **Neoplasm-heavy** keyword buckets. **Inference (iv):** the degree tables identify **which symbols and disease names** mechanically dominate edge counts, so **expected** pressure points for any graph-based scorer are **explicit**, not hidden.

### What the ingested graph actually looks like

The numbers below are for the **default full human ingest** in [`data/`](data/) (LncRNADisease v3.0 after [`scripts/fetch_lncrnadisease_v30.py`](scripts/fetch_lncrnadisease_v30.py) + [`scripts/ingest_lncrnadisease_v30.py`](scripts/ingest_lncrnadisease_v30.py)). They explain **where hub bias comes from** in this assignment: the graph is **sparse for most nodes** but **heavy-tailed** for a few.

| | Count |
|--|--:|
| lncRNA nodes | 5,842 |
| Disease nodes | 440 |
| Known positive edges | 10,518 |
| lncRNA degree (links to diseases): **median** / **max** | 1 / 97 |
| Disease degree (links to lncRNAs): **median** / **max** | 2 / 1,264 |

**lncRNAs with the most disease associations** (these are the usual heavily studied transcripts; any co-occurrence or graph model “sees” them constantly):

| Rank | # diseases | Name |
|-----:|-----------:|------|
| 1 | 97 | MALAT1 |
| 2 | 92 | NEAT1 |
| 3 | 92 | H19 |
| 4 | 64 | PVT1 |
| 5 | 64 | GAS5 |
| 6 | 57 | MEG3 |
| 7 | 54 | HOTAIR |
| 8 | 53 | TUG1 |
| 9 | 48 | CDKN2B-AS1 |
| 10 | 48 | XIST |
| 11 | 44 | UCA1 |
| 12 | 42 | ZFAS1 |
| 13 | 41 | SNHG1 |
| 14 | 35 | SNHG16 |
| 15 | 33 | MIAT |

**Diseases with the most lncRNA associations** (a few rows absorb a large share of edges; **Neoplasm** dominates this tail):

| Rank | # lncRNAs | Disease | Category (keyword) |
|-----:|----------:|---------|---------------------|
| 1 | 1,264 | Esophageal Squamous Cell Carcinoma | Neoplasm |
| 2 | 859 | Atrial Fibrillation | Cardiovascular |
| 3 | 523 | Stomach Neoplasms | Neoplasm |
| 4 | 441 | Carcinoma, Hepatocellular | Neoplasm |
| 5 | 416 | Colorectal Neoplasms | Neoplasm |
| 6 | 390 | Breast Neoplasms | Neoplasm |
| 7 | 389 | Depression | Other |
| 8 | 351 | Osteoarthritis | Immune_inflammatory |
| 9 | 346 | Carcinoma, Non-Small-Cell Lung | Neoplasm |
| 10 | 277 | Pterygium | Other |
| 11 | 241 | Uterine Cervical Neoplasms | Neoplasm |
| 12 | 234 | Glioma | Neoplasm |
| 13 | 222 | Squamous Cell Carcinoma of Head and Neck | Neoplasm |
| 14 | 219 | Osteosarcoma | Neoplasm |
| 15 | 199 | Adenocarcinoma of Lung | Neoplasm |

Stable internal IDs for these diseases live in [`data/diseases.csv`](data/diseases.csv). To recompute degree-ranked lists from your own `data/`, run:

```bash
python scripts/list_hub_entities.py --data-dir data --top 25
```

Scripts and plots for this second line of evidence are in [Biology vs annotation bias (scripts)](#biology-vs-annotation-bias-scripts).

**Synthesis:** **Hold-out** results support **usable link signal** under masking when using **LightGCN**, not the hybrid baseline. **Bias audits** bound how much **top novel scores** align with **hub degree** and **keyword category skew** inherited from the database. Together, they separate **predictive structure in the observed graph** from **structural lopsidedness of the curation itself**; neither line alone replaces external validation.

---

## Evaluation & figures

### Hold-out link prediction (primary: complete dataset)

We randomly put **85%** of known links in **training** and **15%** in a **test** set (default seed **42**). The split is over **all** positive edges in the loaded matrix—**no** subsampling of positives for the split itself. The model never sees the test links during training. We then score **each test positive** against random **decoy** diseases for the same lncRNA and plot **ROC** and **precision–recall** in [`figures/`](figures/). With `--ranking-report`, ranking metrics use **all** held-out positives that pass the candidate filter (see script docstring).

**Reference numbers** on the **full** v3 human ingest in `data/` (same split, seed 42; complete train/test partition of all **10,518** edges):

| Model | AUROC | AUPR | MRR | HR@10 | HR@50 |
|--------|-------|------|-----|-------|-------|
| Hybrid | 0.427 | 0.034 | 0.007 | 0.3% | 1.7% |
| LightGCN (dim 64, 3 layers, 120 epochs) | **0.734** | **0.282** | **0.201** | **41%** | **62%** |

With `--ranking-report`, each held-out positive gets a **rank** among diseases with **no** training edge to that lncRNA (full row-wise candidate list where applicable). Because almost all pairs are absent, **PR** and **MRR / HR@K** summarize **precision in the head of the ranked list** more directly than ROC alone; the table above shows **LightGCN** achieves **meaningful** top-10 and top-50 hit rates under that definition, whereas the hybrid does **not**.

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

### Leave-one-out (optional; not the primary table)

LOO refits the scorer after **removing each evaluated positive** in turn. On this graph, **full** LOO means **~10,518** refits per model—feasible for **hybrid / SVD** (no long neural training), but usually **too slow** for **MF / GNN / LightGCN** unless you use very few epochs per refit.

- **Complete-dataset LOO:** pass **`--all-loo`** (and omit a restrictive `--loo-threshold`) so **every** positive is evaluated; use **`--model hybrid`** or **`--model svd`** for tractable runs, or tiny `examples/minimal_data/`.
- **Subsampled LOO** (e.g. **`--loo-threshold 350`**): only for quick illustrations; **do not** treat it as equivalent to the full hold-out numbers above.

```bash
# Example: subsampled LOO (fast illustration only)
python scripts/eval_loo_link_prediction.py --data-dir data --protocol loo --loo-threshold 350 --seed 42 --out-dir figures

# Example: complete-dataset LOO on every positive (use hybrid/svd unless you accept very long runs)
python scripts/eval_loo_link_prediction.py --data-dir data --protocol loo --all-loo --model hybrid --seed 42 --out-dir figures
```

<p align="center">
  <img src="figures/loo_roc.png" alt="Leave-one-out ROC (optional subsample)" width="45%" />
  <img src="figures/loo_pr.png" alt="Leave-one-out PR (optional subsample)" width="45%" />
</p>

---

## Biology vs annotation bias (scripts)

These scripts implement **line of evidence B** above: they **quantify overlap** between **top-scoring absent pairs** and **positive edges** in **keyword category mix** and in **endpoint degree**, with **random** absent pairs and the **disease pool** as references.

### What ails this approach (honest limits)

Our methods only see **curated positive links** in LncRNADisease plus **synthetic negatives** at evaluation time. They do **not** see full transcriptomics, pathways, or unpublished results. That already mixes **real co-regulation** with **what journals and curators chose to report**. On top of that, **link prediction and graph embeddings are not fair**: they propagate signal along edges, so **high-degree nodes** get richer updates and higher scores almost by construction. The **hybrid** ranker is especially sensitive to **shared-neighbour / co-occurrence** structure, which hubs distort. Even **LightGCN**, which generalizes much better on held-out links, can still **rank popular pairs highly** on the full graph—so we run the audits below. None of this is a bug in the code; it is the **cost of learning from a biased bipartite snapshot**.

### Which entities and “terms” drive the bias here

Cross-check the **degree tables** in [The biological question](#the-biological-question-graph-structure-vs-annotation-bias). In our default ingest, the risky pattern is:

- **lncRNA hubs (symbols):** **MALAT1**, **NEAT1**, **H19**, **PVT1**, **GAS5**, **MEG3**, **HOTAIR**, **TUG1**, **XIST**, **UCA1**, and **SNHG**-family lncRNAs (**SNHG1**, **SNHG5**, **SNHG6**, **SNHG7**, **SNHG16**, …)—a small set of famous transcripts accounts for a disproportionate share of edges, so any scorer that uses graph structure will keep “meeting” them.
- **Disease hubs (names):** **Esophageal Squamous Cell Carcinoma**, **Atrial Fibrillation**, **Stomach Neoplasms**, **Hepatocellular Carcinoma**, **Colorectal Neoplasms**, **Breast Neoplasms**, **NSCLC**, **Glioma**, and other **high-degree cancer rows**—one disease node can link to **hundreds or thousands** of lncRNAs in the matrix, so predictions and training loss are pulled toward those rows.
- **Category keyword that dominates:** **`Neoplasm`** (plus, to a lesser extent, **`Cardiovascular`**, **`Immune_inflammatory`**, **`Other`**) in our ingest’s **keyword** field—**not** a formal ontology. **Enrichment** of top novel pairs in the same bucket as positives is **ambiguous** from graph data alone (**biology** vs **annotation density**); the scripts **surface the overlap** without disentangling the source.

The **hub degree audit** **compares** the **lncRNA degree × disease degree** of top-scoring **absent** edges to **random** absent edges—a proxy for **celebrity-endpoint** concentration. The **category audit** **compares** the **keyword-class histogram** of those top pairs to **labeled edges**, with **Neoplasm** as the usual dominant class in positives.

| Script | What it does |
|--------|----------------|
| [`scripts/category_bias_audit.py`](scripts/category_bias_audit.py) | Looks at the **disease type mix** among the **top predicted new links** and compares it to (a) real links in the data and (b) all diseases. |
| [`scripts/hub_degree_audit.py`](scripts/hub_degree_audit.py) | Compares **how “connected”** those top new links are (lncRNA degree × disease degree) to **random** non-links. **High** degree products on top pairs indicate **concentration** on **hub-like** endpoints. Optional `--checkpoint` uses **LightGCN** scores instead of the hybrid model. |

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
| `scripts/list_hub_entities.py` | Print top-degree lncRNAs / diseases (reproduce README hub tables) |
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
