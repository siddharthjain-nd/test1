"""Embedding tests.

The preprocessing assertions here are the important ones. They do not prove the pipeline
matches InsightFace -- only scripts/verify_embedding_parity.py can do that, against the
reference package -- but they pin the arithmetic so a refactor cannot quietly change it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from faceindex import embed, store


def test_l2_normalise_makes_unit_rows() -> None:
    vectors = np.array([[3.0, 4.0], [1.0, 0.0], [-5.0, 12.0]], dtype=np.float32)
    normalised = embed.l2_normalise(vectors)
    np.testing.assert_allclose(np.linalg.norm(normalised, axis=1), 1.0, rtol=1e-6)


def test_l2_normalise_leaves_zero_rows_finite() -> None:
    """A zero vector must not become NaN and poison every distance it appears in."""
    normalised = embed.l2_normalise(np.zeros((1, 4), dtype=np.float32))
    assert np.isfinite(normalised).all()


def test_blob_round_trip_is_exact() -> None:
    vector = np.random.default_rng(0).normal(size=512).astype(np.float32)
    restored = embed.from_blob(embed.to_blob(vector))
    np.testing.assert_array_equal(vector, restored)


def test_blob_is_little_endian_float32() -> None:
    """The on-disk format is a contract: the Mac and the Linux box must agree byte-for-byte."""
    assert len(embed.to_blob(np.ones(512, dtype=np.float32))) == 512 * 4
    assert embed.to_blob(np.array([1.0], dtype=np.float32)) == b"\x00\x00\x80\x3f"


def test_load_matrix_handles_empty() -> None:
    matrix = embed.load_matrix([])
    assert matrix.shape == (0, embed.EMBED_DIM)


class _FakeEmbedder:
    """Exercises preprocess() without loading an ONNX model."""

    def __init__(self, mean: float = 127.5, std: float = 127.5) -> None:
        self.config = embed.EmbedConfig(model_path=Path("unused.onnx"), mean=mean, std=std)

    preprocess = embed.ArcFaceEmbedder.preprocess


def test_preprocess_uses_arcface_normalisation_not_x_over_255() -> None:
    """The classic silent bug. A mid-grey pixel must map to ~0, not ~0.5."""
    crops = np.full((1, 112, 112, 3), 127, dtype=np.uint8)
    blob = _FakeEmbedder().preprocess(crops)
    assert abs(float(blob.mean())) < 0.01

    white = _FakeEmbedder().preprocess(np.full((1, 112, 112, 3), 255, dtype=np.uint8))
    assert float(white.mean()) == pytest.approx(1.0, abs=0.01)

    black = _FakeEmbedder().preprocess(np.zeros((1, 112, 112, 3), dtype=np.uint8))
    assert float(black.mean()) == pytest.approx(-1.0, abs=0.01)


def test_preprocess_emits_nchw_not_nhwc() -> None:
    crops = np.zeros((2, 112, 112, 3), dtype=np.uint8)
    assert _FakeEmbedder().preprocess(crops).shape == (2, 3, 112, 112)


def test_preprocess_preserves_channel_order() -> None:
    """Channel 0 must stay red. A silent RGB/BGR swap is indistinguishable downstream."""
    crops = np.zeros((1, 112, 112, 3), dtype=np.uint8)
    crops[..., 0] = 255
    blob = _FakeEmbedder().preprocess(crops)
    assert float(blob[0, 0].mean()) == pytest.approx(1.0, abs=0.01)
    assert float(blob[0, 1].mean()) == pytest.approx(-1.0, abs=0.01)


def test_preprocess_rejects_wrong_crop_size() -> None:
    with pytest.raises(ValueError, match="112x112"):
        _FakeEmbedder().preprocess(np.zeros((1, 64, 64, 3), dtype=np.uint8))


def test_preprocess_rejects_greyscale() -> None:
    with pytest.raises(ValueError, match="RGB"):
        _FakeEmbedder().preprocess(np.zeros((1, 112, 112, 1), dtype=np.uint8))


def test_normalisation_constants_differ_measurably() -> None:
    """Document the /127.5 vs /128.0 divergence rather than assuming it is nil."""
    crops = np.full((1, 112, 112, 3), 200, dtype=np.uint8)
    a = _FakeEmbedder(std=127.5).preprocess(crops)
    b = _FakeEmbedder(std=128.0).preprocess(crops)
    relative = float(np.abs(a - b).max() / np.abs(a).max())
    assert relative < 0.01, "constants should differ by well under 1% of signal"


def _seeded_db(tmp_path: Path) -> sqlite3.Connection:
    conn = store.connect(tmp_path / "index.db")
    conn.execute(
        "INSERT INTO photos (id, path, rel_path, size_bytes, mtime, kind, scanned_at, "
        "scan_version) VALUES (1,'/p/a.jpg','a.jpg',10,0,'photo','now','2')"
    )
    for face_id in (1, 2):
        conn.execute(
            "INSERT INTO faces (id, photo_id, face_index, bbox_x1, bbox_y1, bbox_x2, bbox_y2,"
            " landmarks, det_score, decode_scale, crop_path, detector, pool_version, created_at)"
            " VALUES (?,1,?,0,0,10,10,'[]',0.9,1.0,?, 'det','1','now')",
            (face_id, face_id - 1, f"/crops/{face_id}.jpg"),
        )
    conn.commit()
    return conn


def test_pending_faces_lists_unembedded(tmp_path: Path) -> None:
    conn = _seeded_db(tmp_path)
    assert [face_id for face_id, _ in embed.pending_faces(conn, "m.onnx")] == [1, 2]


def test_pending_faces_is_resumable(tmp_path: Path) -> None:
    """Interrupting a 64k-face pass must not restart it."""
    conn = _seeded_db(tmp_path)
    conn.execute(
        "INSERT INTO face_embeddings (face_id, model, embedding, dim, embed_version, "
        "platform, onnxruntime_version, created_at) "
        "VALUES (1, 'm.onnx', ?, 512, ?, 'p', 'v', 'now')",
        (embed.to_blob(np.ones(512, dtype=np.float32)), embed.EMBED_VERSION),
    )
    conn.commit()
    assert [face_id for face_id, _ in embed.pending_faces(conn, "m.onnx")] == [2]


def test_switching_model_makes_every_face_pending_again(tmp_path: Path) -> None:
    """The bug this schema change fixes.

    Keyed on embed_version alone, every face already had *an* embedding, so asking for a
    different model reported "nothing to do" and silently refused to run.
    """
    conn = _seeded_db(tmp_path)
    conn.execute(
        "INSERT INTO face_embeddings (face_id, model, embedding, dim, embed_version, "
        "platform, onnxruntime_version, created_at) "
        "VALUES (1, 'w600k_mbf.onnx', ?, 512, ?, 'p', 'v', 'now')",
        (embed.to_blob(np.ones(512, dtype=np.float32)), embed.EMBED_VERSION),
    )
    conn.commit()

    assert [f for f, _ in embed.pending_faces(conn, "w600k_mbf.onnx")] == [2]
    assert [f for f, _ in embed.pending_faces(conn, "w600k_r50.onnx")] == [1, 2]


def test_two_models_coexist_for_one_face(tmp_path: Path) -> None:
    """Both must be retrievable, so an A/B needs no re-embedding."""
    conn = _seeded_db(tmp_path)
    for model, value in (("w600k_mbf.onnx", 1.0), ("w600k_r50.onnx", 2.0)):
        conn.execute(
            "INSERT INTO face_embeddings (face_id, model, embedding, dim, embed_version, "
            "platform, onnxruntime_version, created_at) VALUES (1, ?, ?, 512, ?, 'p', 'v', 'now')",
            (model, embed.to_blob(np.full(512, value, dtype=np.float32)), embed.EMBED_VERSION),
        )
    conn.commit()

    from faceindex import cluster

    assert dict(cluster.available_models(conn)) == {"w600k_mbf.onnx": 1, "w600k_r50.onnx": 1}
    _, small = cluster.load_embeddings(conn, model="w600k_mbf.onnx")
    _, large = cluster.load_embeddings(conn, model="w600k_r50.onnx")
    assert small[0][0] == 1.0
    assert large[0][0] == 2.0


def test_load_crops_skips_unreadable_without_raising(tmp_path: Path) -> None:
    """One bad file must never abort a multi-hour run (CONTEXT.md rule 7)."""
    missing = str(tmp_path / "nope.jpg")
    batch, ok = embed.load_crops([missing, missing])
    assert batch.shape[0] == 0
    assert ok == []
