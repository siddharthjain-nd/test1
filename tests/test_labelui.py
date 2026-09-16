"""Labelling UI tests.

Labels are the only durable artifact Phase 1 produces, so the store must never accept a
malformed one and must never lose a saved one. The queue behaviour matters too: if the
noise bucket is never served, the gold set silently becomes "the faces the baseline
already finds easy".
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from faceindex import labelui, store


def _seed(tmp_path: Path) -> Path:
    db_path = tmp_path / "index.db"
    conn = store.connect(db_path)
    conn.execute(
        "INSERT INTO photos (id, path, rel_path, size_bytes, mtime, kind, taken_at, "
        "scanned_at, scan_version) VALUES "
        "(1,'/p/a.jpg','a.jpg',10,0,'photo','2016-05-01T10:00:00','now','2')"
    )

    strata = json.dumps({"size": "medium", "pose": "frontal", "era": "oldest"})
    for face_id in range(1, 9):
        cluster_id = 1 if face_id <= 5 else -1
        conn.execute(
            "INSERT INTO faces (id, photo_id, face_index, bbox_x1, bbox_y1, bbox_x2, bbox_y2,"
            " landmarks, det_score, interocular_px, yaw_deg, decode_scale, crop_path,"
            " context_path, detector, pool_version, created_at)"
            " VALUES (?,1,?,0,0,10,10,'[]',0.9,50.0,2.0,1.0,?,?, 'det','1','now')",
            (face_id, face_id - 1, f"/crops/{face_id}.jpg", f"/ctx/{face_id}.jpg"),
        )
        conn.execute(
            "INSERT INTO gold_candidates (face_id, stratum, bootstrap_cluster, sample_run,"
            " created_at) VALUES (?,?,?,'run','now')",
            (face_id, strata, cluster_id),
        )
    conn.commit()
    conn.close()
    return db_path


def test_progress_counts_from_empty(tmp_path: Path) -> None:
    gold = labelui.GoldStore(_seed(tmp_path))
    progress = gold.progress()
    assert progress["total"] == 8
    assert progress["done"] == 0
    assert progress["remaining"] == 8


def test_next_group_serves_largest_cluster_first(tmp_path: Path) -> None:
    """Confirming a big clean cluster is the highest-value keystroke available."""
    gold = labelui.GoldStore(_seed(tmp_path))
    group = gold.next_group()
    assert group["kind"] == "cluster"
    assert group["cluster_id"] == 1
    assert len(group["faces"]) == 5


def test_noise_bucket_is_eventually_served(tmp_path: Path) -> None:
    """PLAN.md is explicit: the noise bucket must be reviewed, not skipped."""
    gold = labelui.GoldStore(_seed(tmp_path))
    gold.save_labels(
        [{"face_id": i, "label": "person", "person_id": "person_1"} for i in range(1, 6)]
    )
    group = gold.next_group()
    assert group["kind"] == "leftovers"
    assert {face["face_id"] for face in group["faces"]} == {6, 7, 8}


def test_next_group_reports_done_when_everything_is_labelled(tmp_path: Path) -> None:
    gold = labelui.GoldStore(_seed(tmp_path))
    gold.save_labels([{"face_id": i, "label": "unsure"} for i in range(1, 9)])
    assert gold.next_group()["kind"] == "done"


def test_save_and_reread_labels(tmp_path: Path) -> None:
    gold = labelui.GoldStore(_seed(tmp_path))
    written = gold.save_labels(
        [
            {"face_id": 1, "label": "person", "person_id": "person_1"},
            {"face_id": 2, "label": "not_of_interest"},
            {"face_id": 3, "label": "non_face"},
        ]
    )
    assert written == 3
    progress = gold.progress()
    assert progress["done"] == 3
    assert progress["by_label"]["non_face"] == 1


def test_relabelling_a_face_overwrites_rather_than_duplicating(tmp_path: Path) -> None:
    gold = labelui.GoldStore(_seed(tmp_path))
    gold.save_labels([{"face_id": 1, "label": "unsure"}])
    gold.save_labels([{"face_id": 1, "label": "person", "person_id": "person_2"}])
    assert gold.progress()["done"] == 1
    assert gold.progress()["by_label"] == {"person": 1}


def test_rejects_unknown_label(tmp_path: Path) -> None:
    gold = labelui.GoldStore(_seed(tmp_path))
    with pytest.raises(ValueError, match="unknown label"):
        gold.save_labels([{"face_id": 1, "label": "probably_dad"}])


def test_person_label_requires_a_person_id(tmp_path: Path) -> None:
    """A person row without an id would silently become its own singleton cluster."""
    gold = labelui.GoldStore(_seed(tmp_path))
    with pytest.raises(ValueError, match="requires a person_id"):
        gold.save_labels([{"face_id": 1, "label": "person"}])


def test_non_person_labels_do_not_carry_a_person_id(tmp_path: Path) -> None:
    gold = labelui.GoldStore(_seed(tmp_path))
    gold.save_labels([{"face_id": 1, "label": "non_face", "person_id": "person_9"}])
    row = gold.conn.execute("SELECT person_id FROM gold_labels WHERE face_id = 1").fetchone()
    assert row["person_id"] is None


def test_next_person_id_increments_past_existing(tmp_path: Path) -> None:
    gold = labelui.GoldStore(_seed(tmp_path))
    assert gold.next_person_id() == "person_1"
    gold.save_labels([{"face_id": 1, "label": "person", "person_id": "person_4"}])
    assert gold.next_person_id() == "person_5"


def test_persons_sort_numerically_not_lexically(tmp_path: Path) -> None:
    """person_10 must not sort between person_1 and person_2 in the merge prompt."""
    gold = labelui.GoldStore(_seed(tmp_path))
    gold.save_labels(
        [
            {"face_id": 1, "label": "person", "person_id": "person_10"},
            {"face_id": 2, "label": "person", "person_id": "person_2"},
        ]
    )
    assert [p["person_id"] for p in gold.persons()] == ["person_2", "person_10"]


def test_undo_removes_a_batch(tmp_path: Path) -> None:
    gold = labelui.GoldStore(_seed(tmp_path))
    gold.save_labels([{"face_id": i, "label": "unsure"} for i in (1, 2, 3)])
    gold.undo([1, 2, 3])
    assert gold.progress()["done"] == 0


def test_crop_path_falls_back_to_aligned_when_context_missing(tmp_path: Path) -> None:
    db_path = _seed(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE faces SET context_path = NULL WHERE id = 1")
    conn.commit()
    conn.close()

    gold = labelui.GoldStore(db_path)
    assert gold.crop_path(1, context=True) == Path("/crops/1.jpg")


def test_labels_constant_matches_the_plan() -> None:
    assert set(labelui.LABELS) == {"person", "not_of_interest", "non_face", "unsure"}
