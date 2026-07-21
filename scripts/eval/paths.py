"""Пути к датасетам и артефактам eval."""

from __future__ import annotations

import os
from pathlib import Path

# .../scripts/eval/paths.py -> <repo root>
REPO_DIR = Path(__file__).resolve().parents[2]
AGENT_RUNTIME_DIR = REPO_DIR


def datasets_dir() -> Path:
    """Каталог золотого датасета (переопределяется через EVAL_DATASETS_DIR)."""
    override = os.environ.get("EVAL_DATASETS_DIR")
    if override:
        return Path(override).resolve()
    return REPO_DIR / "datasets"


def reports_dir() -> Path:
    """Каталог для отчётов валидации (переопределяется через EVAL_REPORTS_DIR)."""
    override = os.environ.get("EVAL_REPORTS_DIR")
    base = Path(override).resolve() if override else (REPO_DIR / "eval_reports")
    base.mkdir(parents=True, exist_ok=True)
    return base


def checkpoints_dir() -> Path:
    """Каталог чекпоинтов покейсного прогона (для дозапуска после обрыва)."""
    base = reports_dir() / "checkpoints"
    base.mkdir(parents=True, exist_ok=True)
    return base


def manifest_path() -> Path:
    return datasets_dir() / "manifest.yaml"
