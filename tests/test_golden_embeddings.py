"""Golden-value regression test for embeddings (PLAN.md section 3).

Reference parity proves the preprocessing is right *today*. This proves it stays right
across refactors, which is a different guarantee: a future change to decoding, alignment
or normalisation would otherwise shift every embedding silently, and the only symptom
would be clustering results that quietly stopped being comparable to earlier experiments.

The baseline lives in gitignored ``data/``, not in ``tests/``, for two reasons. It holds
embeddings of a real person's face -- biometric templates, not "just floats" -- and
committing them to a repository intended to be public would be an irreversible leak. It is
also platform-specific: arm64 and x86_64 do not agree bit-for-bit, so each machine writes
and checks its own baseline.

Generate it with::

    python scripts/verify_embedding_parity.py --write-golden

Run that only after the reference-parity check passes. Pinning unverified values would
lock in whatever bug was present at the time.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from faceindex import embed, paths

GOLDEN_PATH = paths.data_dir() / "golden_embeddings.json"

# PLAN.md section 3 specifies 1e-4 across machines. The same machine reproduces its own
# values far more tightly than that; the allowance exists so a baseline stays usable if the
# platform is ever regenerated, not to excuse drift.
TOLERANCE = 1e-4


def _load() -> dict[str, object]:
    return json.loads(GOLDEN_PATH.read_text())


requires_golden = pytest.mark.skipif(
    not GOLDEN_PATH.exists(),
    reason=(
        "no golden baseline on this machine; run "
        "`python scripts/verify_embedding_parity.py --write-golden` after parity passes"
    ),
)


@requires_golden
def test_golden_baseline_records_its_provenance() -> None:
    """A baseline without provenance cannot be interpreted when it fails."""
    golden = _load()
    for field in ("model", "mean", "std", "embed_version", "platform", "onnxruntime"):
        assert golden.get(field), f"golden baseline is missing {field!r}"


@requires_golden
def test_golden_baseline_matches_this_machine() -> None:
    """A stale baseline from another machine would report a false failure."""
    golden = _load()
    platform, ort_version = embed.runtime_fingerprint()
    if golden["platform"] != platform:
        pytest.skip(
            f"baseline was written on {golden['platform']}, this is {platform}; "
            f"regenerate with --write-golden"
        )
    assert golden["onnxruntime"] == ort_version, (
        f"baseline used onnxruntime {golden['onnxruntime']}, this machine has {ort_version}. "
        f"Pinned versions must match or embeddings drift (PLAN.md section 3)."
    )


@requires_golden
def test_golden_vectors_are_l2_normalised() -> None:
    vectors = np.array(_load()["vectors"], dtype=np.float32)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


@requires_golden
def test_embeddings_have_not_drifted() -> None:
    """The actual regression check: re-embed the fixed crops and compare.

    A failure here means something upstream of the model changed -- decode, alignment,
    normalisation or the model file itself. It does not say which; it says stop and find out
    before running anything that stores embeddings.
    """
    golden = _load()
    platform, _ = embed.runtime_fingerprint()
    if golden["platform"] != platform:
        pytest.skip(f"baseline is from {golden['platform']}, not {platform}")

    model_path = paths.models_dir() / str(golden["model"])
    if not model_path.exists():
        pytest.skip(f"model not downloaded: {golden['model']}")
    if not paths.sample_faces_photo().exists():
        pytest.skip("sample photograph not downloaded")

    # Imported here so the rest of the file still collects without the detector present.
    from faceindex import align, facepool
    from faceindex.detect import ScrfdDetector

    detector_path = paths.models_dir() / "buffalo_l" / "det_10g.onnx"
    if not detector_path.exists():
        pytest.skip("detector not downloaded")

    image, _ = facepool.decode(paths.sample_faces_photo(), 2048)
    faces = sorted(
        ScrfdDetector(detector_path, input_size=(640, 640)).detect(image),
        key=lambda f: -f.interocular_px,
    )
    expected = np.array(golden["vectors"], dtype=np.float32)
    crops = np.stack([align.align_face(image, f.landmarks) for f in faces[: len(expected)]])

    embedder = embed.ArcFaceEmbedder(
        embed.EmbedConfig(
            model_path=model_path, mean=float(golden["mean"]), std=float(golden["std"])
        )
    )
    actual = embedder.embed(crops)

    similarities = np.sum(actual * expected, axis=1)
    worst = float(1.0 - similarities.min())
    assert worst < TOLERANCE, (
        f"embeddings drifted: worst cosine distance {worst:.2e} exceeds {TOLERANCE:.0e}. "
        f"Something upstream of the model changed (decode, alignment, normalisation, or the "
        f"model file). Do not run a pass that stores embeddings until this is understood."
    )
