"""Flask portal: pick a disease, view ranked lncRNA candidates (latent SVD scores)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

from .ranker import LatentRanker

# Repo root (parent of flask_tool/)
ROOT = Path(__file__).resolve().parents[1]


def create_app(data_dir: Path | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )

    data_path = Path(data_dir or os.environ.get("LNC_DATA_DIR", ROOT / "examples" / "minimal_data"))
    app.config["DATA_DIR"] = data_path
    app.config["RANKER"] = None
    app.config["BIPARTITE"] = None

    def _ensure_loaded() -> tuple[bool, str | None]:
        if app.config["BIPARTITE"] is not None:
            return True, None
        assoc = data_path / "associations.csv"
        if not assoc.is_file():
            return False, str(data_path)
        from src.dataio import load_bipartite

        bp = load_bipartite(data_path)
        app.config["BIPARTITE"] = bp
        app.config["RANKER"] = LatentRanker.fit(bp.Y, n_components=32)
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
