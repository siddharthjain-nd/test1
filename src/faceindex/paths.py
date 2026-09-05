"""Canonical filesystem locations.

Every path used by the project resolves through here so that the Mac and the Linux
box can be pointed at different directories without touching any other code.

Overridable with the ``FACEINDEX_ROOT`` and ``FACEINDEX_DATA`` environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path


def _default_root() -> Path:
    """Repository root, i.e. the directory containing ``src/``."""
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    env = os.environ.get("FACEINDEX_ROOT")
    return Path(env).expanduser().resolve() if env else _default_root()


def models_dir() -> Path:
    """Downloaded model weights. Gitignored; non-redistributable."""
    return project_root() / "models"


def data_dir() -> Path:
    """Derived artifacts: crops, embeddings, labels. Gitignored; biometric data."""
    env = os.environ.get("FACEINDEX_DATA")
    return Path(env).expanduser().resolve() if env else project_root() / "data"


def crops_dir() -> Path:
    """Aligned 112x112 crops consumed by the embedding models."""
    return data_dir() / "crops"


def context_crops_dir() -> Path:
    """Wider crops shown to humans during labelling; never fed to a model."""
    return data_dir() / "context"


def gold_dir() -> Path:
    """Gold-set labels (evaluation ground truth only)."""
    return data_dir() / "gold"


def test_assets_dir() -> Path:
    """Fixtures that cannot be generated, such as a real photograph containing faces."""
    return data_dir() / "test_assets"


def sample_faces_photo() -> Path:
    """Multi-face photograph used by detection and alignment tests."""
    return test_assets_dir() / "faces_sample.jpg"


def results_dir() -> Path:
    """One row per experiment run."""
    return data_dir() / "results"


def index_db_path() -> Path:
    """SQLite store for photos, faces and attributes (Phases 1-5)."""
    return data_dir() / "index.db"


def configs_dir() -> Path:
    return project_root() / "configs"


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for path in (
        models_dir(),
        data_dir(),
        crops_dir(),
        context_crops_dir(),
        gold_dir(),
        results_dir(),
        test_assets_dir(),
    ):
        path.mkdir(parents=True, exist_ok=True)
