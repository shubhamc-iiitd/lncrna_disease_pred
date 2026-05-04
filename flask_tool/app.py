"""Flask portal: pick a disease, view ranked lncRNA candidates (latent SVD scores)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

from .ranker import HybridLinkScorer

# Repo root (parent of flask_tool/)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIGHTGCN_CKPT = ROOT / "checkpoints" / "lightgcn_full.pt"


def _default_data_dir() -> Path:
    """Prefer ingested LncRNADisease v3 tables under ./data; fallback to bundled demo."""
    full = ROOT / "data"
    if (full / "associations.csv").is_file():
        return full
    return ROOT / "examples" / "minimal_data"


def create_app(data_dir: Path | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )

    env_raw = os.environ.get("LNC_DATA_DIR")
    env_dir = Path(env_raw) if env_raw else None
    data_path = Path(data_dir or env_dir or _default_data_dir())
    app.config["DATA_DIR"] = data_path
    app.config["RANKER"] = None
    app.config["BIPARTITE"] = None
    app.config["MODEL_LABEL"] = ""

    def _ensure_loaded() -> tuple[bool, str | None]:
        if app.config["BIPARTITE"] is not None:
            return True, None
        assoc = data_path / "associations.csv"
        if not assoc.is_file():
            return False, str(data_path)
        from src.dataio import load_bipartite

        bp = load_bipartite(data_path)
        app.config["BIPARTITE"] = bp

        ckpt_env = os.environ.get("LNC_LIGHTGCN_CKPT")
        ckpt = Path(ckpt_env) if ckpt_env else DEFAULT_LIGHTGCN_CKPT
        use_lg = os.environ.get("LNC_USE_LIGHTGCN", "1").lower() not in ("0", "false", "no")

        if use_lg and ckpt.is_file():
            try:
                from src.learned_edge_models import MatrixScorer
                from src.lightgcn_bipartite import load_lightgcn_for_inference

                dev = os.environ.get("LNC_TORCH_DEVICE", "cpu")
                S, _, _ = load_lightgcn_for_inference(bp.Y.tocsr(), ckpt, device=dev)
                app.config["RANKER"] = MatrixScorer(S)
                app.config["MODEL_LABEL"] = f"Bipartite LightGCN (checkpoint: {ckpt.name})"
            except Exception:
                app.config["RANKER"] = HybridLinkScorer.fit(bp.Y, n_components=32)
                app.config["MODEL_LABEL"] = "Hybrid SVD + co-occurrence (LightGCN load failed)"
        else:
            app.config["RANKER"] = HybridLinkScorer.fit(bp.Y, n_components=32)
            app.config["MODEL_LABEL"] = "Hybrid SVD + co-occurrence"
        return True, None

    @app.route("/")
    def index():
        ok, missing_dir = _ensure_loaded()
        if not ok:
            return render_template(
                "setup.html",
                missing_dir=missing_dir,
                root=str(ROOT),
            ), 200
        bp = app.config["BIPARTITE"]
        diseases = [
            {
                "id": did,
                "name": bp.disease_names.get(did, did),
                "category": bp.disease_category.get(did, "Unknown"),
            }
            for did in bp.disease_ids
        ]
        diseases.sort(key=lambda x: x["name"].lower())
        return render_template(
            "index.html",
            n_lnc=len(bp.lnc_ids),
            n_dis=len(bp.disease_ids),
            n_edges=int(bp.Y.nnz),
            diseases=diseases,
            data_dir=str(data_path),
            model_label=app.config.get("MODEL_LABEL", ""),
        )

    @app.route("/api/diseases")
    def api_diseases():
        ok, missing_dir = _ensure_loaded()
        if not ok:
            return jsonify({"error": "no_data", "data_dir": missing_dir}), 404
        bp = app.config["BIPARTITE"]
        out = [
            {"id": did, "name": bp.disease_names.get(did, did), "category": bp.disease_category.get(did, "")}
            for did in bp.disease_ids
        ]
        out.sort(key=lambda x: x["name"].lower())
        return jsonify(out)

    @app.route("/api/rank")
    def api_rank():
        ok, _ = _ensure_loaded()
        if not ok:
            return jsonify({"error": "no_data"}), 404
        did = request.args.get("disease_id", type=str)
        top = request.args.get("top", default=50, type=int) or 50
        top = min(max(top, 5), 500)
        if not did:
            return jsonify({"error": "missing disease_id"}), 400
        bp = app.config["BIPARTITE"]
        ranker = app.config["RANKER"]
        if did not in bp.disease_ids:
            return jsonify({"error": "unknown disease_id"}), 404
        j = bp.disease_ids.index(did)
        scores = ranker.scores_for_disease(j)
        known = set(bp.Y.getcol(j).nonzero()[0])
        order = np.argsort(-scores)
        rows = []
        for i in order:
            if len(rows) >= top:
                break
            lid = bp.lnc_ids[i]
            rows.append(
                {
                    "lncrna_id": lid,
                    "name": bp.lnc_names.get(lid, lid),
                    "score": float(scores[i]),
                    "known": i in known,
                }
            )
        return jsonify(
            {
                "disease_id": did,
                "disease_name": bp.disease_names.get(did, did),
                "category": bp.disease_category.get(did, ""),
                "results": rows,
            }
        )

    return app
