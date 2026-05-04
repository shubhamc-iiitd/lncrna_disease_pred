#!/usr/bin/env python3
"""Run the lncRNA–disease ranking portal (Flask).

Usage:
  pip install -r requirements.txt
  python run_flask.py

Easy path (download + ingest + train LightGCN if anything is missing):
  python run_flask.py --auto-setup

Same via environment:
  export LNC_AUTO_SETUP=1
  python run_flask.py

Use full LncRNADisease v3.0 tables after ingest:
  export LNC_DATA_DIR=/path/to/huang_assignment/data
  python run_flask.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask_tool.app import create_app  # noqa: E402
from src.setup_pipeline import auto_setup_enabled_from_env, run_auto_setup_if_enabled  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="lncRNA–disease Flask portal")
    p.add_argument(
        "--auto-setup",
        action="store_true",
        help="If data or LightGCN checkpoint is missing: fetch v3.0, ingest, then train (needs torch).",
    )
    p.add_argument("--train-epochs", type=int, default=None, help="LightGCN epochs when auto-training (default: 250 or LNC_AUTO_TRAIN_EPOCHS)")
    p.add_argument("--skip-train", action="store_true", help="With --auto-setup: only fetch+ingest, no LightGCN training")
    args, _unknown = p.parse_known_args()

    do_setup = args.auto_setup or auto_setup_enabled_from_env()
    run_auto_setup_if_enabled(
        enabled=do_setup,
        train_epochs=args.train_epochs,
        skip_train=args.skip_train,
    )

    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
