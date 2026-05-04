#!/usr/bin/env python3
"""Run the lncRNA–disease ranking portal (Flask).

Usage:
  pip install -r requirements.txt
  python run_flask.py

Use full LncRNADisease v3.0 tables after ingest:
  export LNC_DATA_DIR=/path/to/huang_assignment/data
  python run_flask.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask_tool.app import create_app  # noqa: E402


def main() -> None:
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
