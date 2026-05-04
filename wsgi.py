"""WSGI entry for `flask --app wsgi run` or production servers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask_tool.app import create_app
from src.setup_pipeline import auto_setup_enabled_from_env, run_auto_setup_if_enabled

if auto_setup_enabled_from_env():
    run_auto_setup_if_enabled(enabled=True)

app = create_app()
