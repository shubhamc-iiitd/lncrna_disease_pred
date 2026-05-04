"""One-shot fetch → ingest → optional LightGCN training when artifacts are missing.

Used by ``run_flask.py --auto-setup`` and by ``wsgi.py`` when ``LNC_AUTO_SETUP=1``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _resolve_data_dir() -> Path:
    raw = os.environ.get("LNC_DATA_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return (ROOT / "data").resolve()


def _resolve_ckpt() -> Path:
    raw = os.environ.get("LNC_LIGHTGCN_CKPT")
    if raw:
        return Path(raw).expanduser().resolve()
    return (ROOT / "checkpoints" / "lightgcn_full.pt").resolve()


def ensure_ingested_tables(data_dir: Path) -> None:
    """Download v3.0 simple CSV and build associations if ``associations.csv`` is absent."""
    assoc = data_dir / "associations.csv"
    if assoc.is_file():
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    simple = raw_dir / "website_simple_data.csv"

    print("[setup] No associations.csv — downloading LncRNADisease v3.0 (simple CSV) …")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fetch_lncrnadisease_v30.py"),
            "--out-dir",
            str(raw_dir),
            "--which",
            "simple",
        ],
        check=True,
        cwd=str(ROOT),
    )
    if not simple.is_file():
        raise RuntimeError(f"Fetch did not produce {simple}")

    print("[setup] Building lncRNA–disease CSV tables …")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ingest_lncrnadisease_v30.py"),
            "--simple-csv",
            str(simple),
            "--out-dir",
            str(data_dir),
        ],
        check=True,
        cwd=str(ROOT),
    )
    if not assoc.is_file():
        raise RuntimeError(f"Ingest did not write {assoc}")


def ensure_lightgcn_checkpoint(
    data_dir: Path,
    ckpt: Path,
    *,
    epochs: int,
    dim: int,
    layers: int,
    device: str,
) -> None:
    """Train and save LightGCN if checkpoint missing and PyTorch is available."""
    if ckpt.is_file():
        return

    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "[setup] PyTorch not installed — skipping LightGCN training. "
            "Install torch and re-run, or the app will use the hybrid ranker."
        )
        return

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    print(f"[setup] No checkpoint at {ckpt} — training LightGCN ({epochs} epochs, may take several minutes) …")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_lightgcn_full.py"),
            "--data-dir",
            str(data_dir),
            "--out",
            str(ckpt),
            "--epochs",
            str(epochs),
            "--dim",
            str(dim),
            "--layers",
            str(layers),
            "--device",
            device,
        ],
        check=True,
        cwd=str(ROOT),
    )
    if not ckpt.is_file():
        raise RuntimeError(f"Training did not write {ckpt}")


def run_auto_setup_if_enabled(
    *,
    enabled: bool,
    train_epochs: int | None = None,
    dim: int | None = None,
    layers: int | None = None,
    device: str | None = None,
    skip_train: bool = False,
) -> None:
    """If ``enabled``, ensure CSVs exist; then train LightGCN unless skipped or disabled via env."""
    if not enabled:
        return

    data_dir = _resolve_data_dir()
    ensure_ingested_tables(data_dir)

    use_lg = os.environ.get("LNC_USE_LIGHTGCN", "1").strip().lower() not in ("0", "false", "no")
    if skip_train or os.environ.get("LNC_AUTO_SETUP_SKIP_TRAIN", "").strip().lower() in ("1", "true", "yes"):
        print("[setup] Skipping LightGCN training (skip flag or LNC_AUTO_SETUP_SKIP_TRAIN=1).")
        return
    if not use_lg:
        print("[setup] LNC_USE_LIGHTGCN=0 — skipping LightGCN training.")
        return

    ckpt = _resolve_ckpt()
    epochs = train_epochs if train_epochs is not None else int(os.environ.get("LNC_AUTO_TRAIN_EPOCHS", "250"))
    d = dim if dim is not None else int(os.environ.get("LNC_AUTO_DIM", "64"))
    L = layers if layers is not None else int(os.environ.get("LNC_AUTO_LAYERS", "3"))
    if device is not None:
        dev = device
    else:
        dev = os.environ.get("LNC_AUTO_DEVICE") or os.environ.get("LNC_TORCH_DEVICE", "cpu")

    ensure_lightgcn_checkpoint(data_dir, ckpt, epochs=epochs, dim=d, layers=L, device=dev)


def auto_setup_enabled_from_env() -> bool:
    return os.environ.get("LNC_AUTO_SETUP", "").strip().lower() in ("1", "true", "yes")
