"""End-to-end face pool construction.

Uses a real photograph containing several faces, so decoding, detection, alignment,
attribute extraction, storage and resumption are all exercised together rather than
mocked. Skips cleanly when the asset or models are unavailable.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from faceindex import facepool, ingest, paths, store

DETECTOR = paths.models_dir() / "buffalo_sc" / "det_500m.onnx"
SAMPLE = paths.sample_faces_photo()

pytestmark = pytest.mark.skipif(
    not DETECTOR.exists() or not SAMPLE.exists(),
    reason="detector or sample photo unavailable",
)


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny library built from a real multi-face photograph."""
    root = tmp_path / "Photos Timeline"
    (root / "2021 Trip").mkdir(parents=True)
    (root / "portrait").mkdir(parents=True)

    shutil.copy(SAMPLE, root / "2021 Trip" / "group.jpg")

    # A portrait-orientation copy, to prove EXIF rotation happens before detection.
    with Image.open(SAMPLE) as image:
        rotated = image.rotate(90, expand=True)
        exif = Image.Exif()
        exif[274] = 1
        rotated.save(root / "portrait" / "rotated.jpg", "JPEG", exif=exif, quality=90)

    db_path = tmp_path / "index.db"
    with store.open_index(db_path) as conn:
        ingest.scan(conn, root)
        ingest.resolve_duplicates(conn)

    return root, db_path


def _config(tmp_path: Path) -> facepool.PoolConfig:
    return facepool.PoolConfig(
        detector_path=DETECTOR,
        crops_dir=tmp_path / "crops",
        context_dir=tmp_path / "context",
        threads_per_worker=2,
    )


def test_pool_detects_aligns_and_stores(corpus: tuple[Path, Path], tmp_path: Path) -> None:
    _, db_path = corpus
    config = _config(tmp_path)

    with store.open_index(db_path) as conn:
        tasks = facepool.pending_photos(conn)
        assert len(tasks) == 2

        results = list(facepool.run(conn, tasks, config, workers=1))
        assert all(r.error is None for r in results), [r.error for r in results]

        faces = conn.execute("SELECT * FROM faces ORDER BY photo_id, face_index").fetchall()

    assert len(faces) >= 6, "the sample photograph contains several faces"

    for face in faces:
        assert face["bbox_x2"] > face["bbox_x1"]
        assert face["bbox_y2"] > face["bbox_y1"]
        assert 0.0 < face["det_score"] <= 1.0
        assert face["interocular_px"] > 0
        assert -91.0 <= face["yaw_deg"] <= 91.0
        assert face["blur"] >= 0.0
        assert face["decode_scale"] >= 1.0

        landmarks = np.array(json.loads(face["landmarks"]), dtype=np.float32)
        assert landmarks.shape == (5, 2)

        crop = Path(str(face["crop_path"]))
        assert crop.exists()
        with Image.open(crop) as image:
            assert image.size == (112, 112)

        context = Path(str(face["context_path"]))
        assert context.exists()
        with Image.open(context) as image:
            assert image.size == (256, 256)


def test_face_index_is_ordered_by_size(corpus: tuple[Path, Path], tmp_path: Path) -> None:
    """face_index 0 must be the largest face, so "the main subject" is addressable."""
    _, db_path = corpus
    with store.open_index(db_path) as conn:
        list(facepool.run(conn, facepool.pending_photos(conn), _config(tmp_path), workers=1))
        rows = conn.execute(
            "SELECT photo_id, face_index, (bbox_x2-bbox_x1)*(bbox_y2-bbox_y1) AS area "
            "FROM faces ORDER BY photo_id, face_index"
        ).fetchall()

    by_photo: dict[int, list[float]] = {}
    for row in rows:
        by_photo.setdefault(int(row["photo_id"]), []).append(float(row["area"]))

    for areas in by_photo.values():
        assert areas == sorted(areas, reverse=True)


def test_pool_is_resumable(corpus: tuple[Path, Path], tmp_path: Path) -> None:
    _, db_path = corpus
    config = _config(tmp_path)

    with store.open_index(db_path) as conn:
        first = facepool.pending_photos(conn)
        list(facepool.run(conn, first[:1], config, workers=1))

        remaining = facepool.pending_photos(conn)
        assert len(remaining) == len(first) - 1

        list(facepool.run(conn, remaining, config, workers=1))
        assert facepool.pending_photos(conn) == []


def test_rerun_does_not_duplicate_faces(corpus: tuple[Path, Path], tmp_path: Path) -> None:
    """Reprocessing a photo must replace its faces, never append to them."""
    _, db_path = corpus
    config = _config(tmp_path)

    with store.open_index(db_path) as conn:
        tasks = facepool.pending_photos(conn)
        list(facepool.run(conn, tasks, config, workers=1))
        before = conn.execute("SELECT COUNT(*) AS n FROM faces").fetchone()["n"]

        # Force the same photos through again.
        conn.execute("DELETE FROM photo_pool_status")
        conn.commit()
        list(facepool.run(conn, facepool.pending_photos(conn), config, workers=1))
        after = conn.execute("SELECT COUNT(*) AS n FROM faces").fetchone()["n"]

    assert before == after


def test_orientation_is_applied_before_detection(corpus: tuple[Path, Path], tmp_path: Path) -> None:
    """The rotated copy must yield faces too; skipping EXIF rotation loses most portraits."""
    _, db_path = corpus
    with store.open_index(db_path) as conn:
        list(facepool.run(conn, facepool.pending_photos(conn), _config(tmp_path), workers=1))
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM faces f JOIN photos p ON p.id = f.photo_id "
            "WHERE p.rel_path LIKE 'portrait%'"
        ).fetchone()

    assert row["n"] >= 1


def test_decode_respects_the_long_edge_cap() -> None:
    array, scale = facepool.decode(SAMPLE, 512)
    assert max(array.shape[:2]) <= 512
    assert array.dtype == np.uint8
    assert array.shape[2] == 3
    assert scale > 1.0, "scale maps decoded pixels back to original pixels"


def test_unreadable_photo_is_recorded_not_raised(tmp_path: Path) -> None:
    """One corrupt file must never abort a multi-hour run."""
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"\xff\xd8 not a jpeg")

    facepool._init_worker(_config(tmp_path))
    result = facepool.process_photo((999, str(broken)))

    assert result.faces == []
    assert result.error is not None and "decode" in result.error
