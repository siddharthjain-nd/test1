"""Phase 0 sanity checks: environment, paths, and model integrity.

The model tests skip when weights are absent so the suite still runs on a fresh clone
before ``scripts/download_models.py`` has been executed.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from faceindex import paths

MODEL_FILES = {
    "buffalo_sc/det_500m.onnx": "detector",
    "buffalo_sc/w600k_mbf.onnx": "embedder",
    "buffalo_l/det_10g.onnx": "detector",
    "buffalo_l/w600k_r50.onnx": "embedder",
}


def test_paths_are_absolute_and_nested_under_root() -> None:
    root = paths.project_root()
    for path in (paths.models_dir(), paths.data_dir(), paths.crops_dir(), paths.gold_dir()):
        assert path.is_absolute()
        assert root in path.parents or path == root


def test_data_and_models_are_gitignored() -> None:
    """Committing weights or crops would leak biometric data or non-redistributable files.

    Asserts real ``git check-ignore`` behaviour rather than the text of ``.gitignore``,
    because the lock-file exception makes the rules non-obvious.
    """
    import subprocess

    root = paths.project_root()
    if not (root / ".git").exists() or shutil.which("git") is None:
        pytest.skip("not a git checkout")

    def is_ignored(relative: str) -> bool:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", relative],
            cwd=root,
            capture_output=True,
        )
        if result.returncode not in (0, 1):
            pytest.skip(f"git check-ignore unavailable: {result.stderr.decode()}")
        return result.returncode == 0

    assert is_ignored("data/crops/face.jpg"), "face crops must never be committable"
    assert is_ignored("data/index.db"), "the face database must never be committable"
    assert is_ignored("models/buffalo_sc/w600k_mbf.onnx"), "weights are not redistributable"

    # The one deliberate exception: without it, cross-machine model verification breaks.
    assert not is_ignored("models/models.lock.json"), "the checksum lock MUST be committed"


@pytest.mark.parametrize("relative_path", sorted(MODEL_FILES))
def test_model_matches_lock_file(relative_path: str) -> None:
    """Guards against silently running different weights on the two machines."""
    import hashlib

    lock_path = paths.models_dir() / "models.lock.json"
    if not lock_path.exists():
        pytest.skip("models.lock.json absent; run scripts/download_models.py")

    model_path = paths.models_dir() / relative_path
    if not model_path.exists():
        pytest.skip(f"{relative_path} not downloaded")

    recorded = json.loads(lock_path.read_text(encoding="utf-8"))["sha256"]
    assert relative_path in recorded, f"{relative_path} missing from lock file"

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert digest == recorded[relative_path]


@pytest.mark.parametrize(
    "relative_path", [p for p, role in MODEL_FILES.items() if role == "embedder"]
)
def test_embedder_produces_512d_vector(relative_path: str) -> None:
    import onnxruntime as ort

    model_path = paths.models_dir() / relative_path
    if not model_path.exists():
        pytest.skip(f"{relative_path} not downloaded")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, 3, 112, 112), dtype=np.float32)
    output = session.run(None, {session.get_inputs()[0].name: dummy})[0]

    assert output.shape == (1, 512)
    assert output.dtype == np.float32


@pytest.mark.parametrize(
    "relative_path", [p for p, role in MODEL_FILES.items() if role == "detector"]
)
def test_detector_has_nine_outputs(relative_path: str) -> None:
    """SCRFD emits score/bbox/keypoint tensors for each of three strides."""
    import onnxruntime as ort

    model_path = paths.models_dir() / relative_path
    if not model_path.exists():
        pytest.skip(f"{relative_path} not downloaded")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    outputs = session.get_outputs()

    assert len(outputs) == 9, "expected 3 strides x (score, bbox, keypoints)"
    assert [o.shape[-1] for o in outputs[6:9]] == [10, 10, 10], "keypoint heads must be 5 xy pairs"


def test_cpu_execution_provider_available() -> None:
    """Stored embeddings must be produced on CPU so both machines agree (PLAN.md section 3)."""
    import onnxruntime as ort

    assert "CPUExecutionProvider" in ort.get_available_providers()
