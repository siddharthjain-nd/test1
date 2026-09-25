"""Review layer: ranking, index building, and the read-only server.

The ranking is the one thing no measurement can grade -- the gold set covers 1,499 faces of
63,878 -- so what is pinned here is everything around it: that piles contain the faces they
claim to, that the order is the order that was computed, and that every route answers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from faceindex import cluster, embed, review, reviewui, store

MODEL = "w600k_r50.onnx"


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def test_score_rewards_size_but_only_logarithmically() -> None:
    """The hundredth face of a person is worth far less than the second."""
    small = review.score_pile(2, 70, 0.8)
    medium = review.score_pile(20, 70, 0.8)
    large = review.score_pile(200, 70, 0.8)
    assert small < medium < large
    # A hundredfold more faces, nowhere near a hundredfold the score.
    assert large < 5 * small


def test_unreadable_pile_sinks_however_large_and_tight() -> None:
    """The measured failure: tiny faces score 0.887 and cluster tightly anyway."""
    tiny_but_huge = review.score_pile(500, 18, 0.95)
    small_but_clear = review.score_pile(6, 70, 0.8)
    assert tiny_but_huge < small_but_clear


def test_incoherent_pile_sinks() -> None:
    assert review.score_pile(50, 70, 0.2) < review.score_pile(50, 70, 0.9)


def test_score_is_zero_when_any_factor_is_zero() -> None:
    """A product, not a sum: one fatal flaw must sink a pile rather than be averaged away."""
    assert review.score_pile(100, 0, 0.9) == 0.0
    assert review.score_pile(100, 70, 0.0) == 0.0


def test_score_tolerates_missing_quality() -> None:
    assert review.score_pile(10, None, None) == 0.0


def test_readability_saturates() -> None:
    """A 200-pixel face is not worth four times a 50-pixel one; both are simply legible."""
    assert review.score_pile(10, 50, 0.9) == pytest.approx(review.score_pile(10, 200, 0.9))


def test_run_id_is_stable_and_readable() -> None:
    assert review.run_id_for("w600k_r50.onnx", "components", 0.51) == "r50-components-0.51"
    assert review.run_id_for("w600k_r50.onnx", "components", 0.5) == "r50-components-0.5"


# --------------------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------------------


def _seed(tmp_path: Path, plan: list[tuple[int, float]], n_lone: int = 0) -> Path:
    """A database of faces, embeddings and crops. ``plan`` is (n_faces, interocular_px)."""
    db_path = tmp_path / "index.db"
    crops = tmp_path / "crops"
    crops.mkdir(exist_ok=True)
    conn = store.connect(db_path)
    conn.execute(
        "INSERT INTO photos (id,path,rel_path,size_bytes,mtime,kind,scanned_at,scan_version)"
        " VALUES (1,'/p/a.jpg','a.jpg',10,0,'photo','now','2')"
    )
    rng = np.random.default_rng(11)

    def unit(v: np.ndarray) -> np.ndarray:
        return embed.l2_normalise(np.asarray(v, dtype=np.float32)[None, :])[0]

    face_id = 0
    for n_faces, eye in plan:
        centre = unit(rng.normal(size=embed.EMBED_DIM))
        for _ in range(n_faces):
            face_id += 1
            vector = unit(centre + 0.033 * rng.normal(size=embed.EMBED_DIM).astype(np.float32))
            _insert(conn, crops, face_id, vector, eye)
    for _ in range(n_lone):
        face_id += 1
        _insert(conn, crops, face_id, unit(rng.normal(size=embed.EMBED_DIM)), 12.0)
    conn.commit()
    conn.close()
    return db_path


def _insert(
    conn: sqlite3.Connection, crops: Path, face_id: int, vector: np.ndarray, eye: float
) -> None:
    crop = crops / f"{face_id}.jpg"
    crop.write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")
    conn.execute(
        "INSERT INTO faces (id,photo_id,face_index,bbox_x1,bbox_y1,bbox_x2,bbox_y2,landmarks,"
        "det_score,interocular_px,blur,decode_scale,crop_path,detector,pool_version,created_at)"
        " VALUES (?,1,?,0,0,10,10,'[]',0.9,?,100.0,1.0,?,'det','1','now')",
        (face_id, face_id, eye, str(crop)),
    )
    conn.execute(
        "INSERT INTO face_embeddings (face_id,model,embedding,dim,embed_version,platform,"
        "onnxruntime_version,created_at) VALUES (?,?,?,512,?,'p','v','now')",
        (face_id, MODEL, embed.to_blob(vector), embed.EMBED_VERSION),
    )


def _fake_clusterer(labels: list[int]):
    def run(features: np.ndarray, config: cluster.ClusterConfig) -> cluster.ClusterResult:
        array = np.array(labels, dtype=np.int64)
        return cluster.ClusterResult(
            labels=array,
            probabilities=np.ones(len(array)),
            n_clusters=len(set(labels) - {-1}),
            n_noise=labels.count(-1),
        )

    return run


# --------------------------------------------------------------------------------------
# Index building
# --------------------------------------------------------------------------------------


def test_build_index_separates_piles_from_lone_faces(tmp_path: Path) -> None:
    db_path = _seed(tmp_path, [(4, 70.0)], n_lone=3)
    with store.open_index(db_path) as conn:
        summary = review.build_index(
            conn, model=MODEL, cluster_fn=_fake_clusterer([0, 0, 0, 0, -1, -1, -1])
        )
        assert summary.n_piles == 1
        assert summary.n_lone == 3
        assert summary.n_faces == 7


def test_a_group_of_one_is_not_a_pile(tmp_path: Path) -> None:
    """Counting lone faces as piles is what made the review job look ten times bigger."""
    db_path = _seed(tmp_path, [(3, 70.0)])
    with store.open_index(db_path) as conn:
        summary = review.build_index(conn, model=MODEL, cluster_fn=_fake_clusterer([0, 0, 1]))
        assert summary.n_piles == 1
        assert summary.n_lone == 1


def test_members_keep_the_right_pile(tmp_path: Path) -> None:
    """Regression: np.split returns original indices, not positions in the sorted view.

    Reading labels out of the sorted array with those indices put faces in other people's
    piles while the counts still looked plausible.
    """
    db_path = _seed(tmp_path, [(3, 70.0), (3, 70.0)])
    with store.open_index(db_path) as conn:
        review.build_index(
            conn, model=MODEL, cluster_fn=_fake_clusterer([0, 0, 0, 1, 1, 1])
        )
        rows = conn.execute(
            "SELECT face_id, pile_id FROM review_members ORDER BY face_id"
        ).fetchall()
    assignment = {int(r["face_id"]): int(r["pile_id"]) for r in rows}
    assert assignment == {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}


def test_faces_are_ordered_most_recognisable_first(tmp_path: Path) -> None:
    db_path = _seed(tmp_path, [(3, 70.0)])
    with store.open_index(db_path) as conn:
        conn.execute("UPDATE faces SET interocular_px = 10 WHERE id = 1")
        conn.execute("UPDATE faces SET interocular_px = 90 WHERE id = 3")
        conn.commit()
        review.build_index(conn, model=MODEL, cluster_fn=_fake_clusterer([0, 0, 0]))
        faces = review.pile_faces(conn, review.run_id_for(MODEL, "components", 0.51), 0)
    assert [f["face_id"] for f in faces] == [3, 2, 1]


def test_unreadable_pile_ranks_below_a_smaller_clear_one(tmp_path: Path) -> None:
    """The behaviour the whole ranking exists for."""
    db_path = _seed(tmp_path, [(12, 15.0), (4, 75.0)])
    labels = [0] * 12 + [1] * 4
    with store.open_index(db_path) as conn:
        review.build_index(conn, model=MODEL, cluster_fn=_fake_clusterer(labels))
        piles = review.list_piles(conn, review.run_id_for(MODEL, "components", 0.51))
    assert [int(p["pile_id"]) for p in piles] == [1, 0]


def test_rebuilding_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    db_path = _seed(tmp_path, [(4, 70.0)])
    with store.open_index(db_path) as conn:
        for _ in range(3):
            review.build_index(conn, model=MODEL, cluster_fn=_fake_clusterer([0, 0, 0, 0]))
        assert conn.execute("SELECT COUNT(*) AS n FROM review_runs").fetchone()["n"] == 1
        assert conn.execute("SELECT COUNT(*) AS n FROM review_piles").fetchone()["n"] == 1
        assert conn.execute("SELECT COUNT(*) AS n FROM review_members").fetchone()["n"] == 4


def test_two_thresholds_coexist(tmp_path: Path) -> None:
    """So a better setting can be indexed and compared before the old one is discarded."""
    db_path = _seed(tmp_path, [(4, 70.0)])
    with store.open_index(db_path) as conn:
        review.build_index(
            conn, model=MODEL, threshold=0.51, cluster_fn=_fake_clusterer([0, 0, 0, 0])
        )
        review.build_index(
            conn, model=MODEL, threshold=0.60, cluster_fn=_fake_clusterer([0, 0, 1, 1])
        )
        assert len(review.runs(conn)) == 2
        assert review.latest_run(conn) is not None


def test_build_index_refuses_an_empty_pool(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.db"
    store.connect(db_path).close()
    with store.open_index(db_path) as conn, pytest.raises(ValueError, match="nothing to index"):
        review.build_index(conn, model=MODEL)


def test_real_clusterer_recovers_planted_groups(tmp_path: Path) -> None:
    """One test that exercises the actual clustering rather than an injected answer."""
    db_path = _seed(tmp_path, [(8, 70.0), (8, 70.0)], n_lone=5)
    with store.open_index(db_path) as conn:
        summary = review.build_index(conn, model=MODEL, threshold=0.51, jobs=1)
        rows = conn.execute("SELECT face_id, pile_id FROM review_members").fetchall()
    assert summary.n_piles == 2
    assert summary.n_lone == 5
    piles: dict[int, set[int]] = {}
    for row in rows:
        piles.setdefault(int(row["pile_id"]), set()).add(int(row["face_id"]))
    assert piles[-1] == set(range(17, 22))
    assert {frozenset(v) for k, v in piles.items() if k != -1} == {
        frozenset(range(1, 9)),
        frozenset(range(9, 17)),
    }


def test_paging_is_stable_and_covers_everything(tmp_path: Path) -> None:
    db_path = _seed(tmp_path, [(4, 70.0), (6, 70.0), (8, 70.0)])
    labels = [0] * 4 + [1] * 6 + [2] * 8
    with store.open_index(db_path) as conn:
        review.build_index(conn, model=MODEL, cluster_fn=_fake_clusterer(labels))
        run_id = review.run_id_for(MODEL, "components", 0.51)
        whole = [p["pile_id"] for p in review.list_piles(conn, run_id, limit=10)]
        paged = [
            p["pile_id"]
            for offset in (0, 2)
            for p in review.list_piles(conn, run_id, offset=offset, limit=2)
        ]
    assert whole == paged
    assert len(whole) == 3


def test_known_people_comes_from_the_gold_set(tmp_path: Path) -> None:
    db_path = _seed(tmp_path, [(4, 70.0)])
    with store.open_index(db_path) as conn:
        for face_id, person in ((1, "person_1"), (2, "person_1"), (3, "person_2")):
            conn.execute(
                "INSERT INTO gold_labels (face_id,label,person_id,labelled_at) "
                "VALUES (?,'person',?,'now')",
                (face_id, person),
            )
        conn.execute(
            "INSERT INTO gold_labels (face_id,label,labelled_at) VALUES (4,'non_face','now')"
        )
        conn.commit()
        people = review.known_people(conn)
    assert [p["person_id"] for p in people] == ["person_1", "person_2"]
    assert people[0]["n_faces"] == 2


# --------------------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------------------


@pytest.fixture
def served(tmp_path: Path):
    db_path = _seed(tmp_path, [(6, 70.0), (4, 15.0)], n_lone=2)
    labels = [0] * 6 + [1] * 4 + [-1, -1]
    with store.open_index(db_path) as conn:
        review.build_index(conn, model=MODEL, cluster_fn=_fake_clusterer(labels))

    server = reviewui.serve(db_path, port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(url: str) -> tuple[int, str, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read()


def test_index_page_is_served(served: str) -> None:
    status, content_type, body = _get(served + "/")
    assert status == 200
    assert "text/html" in content_type
    assert b"Review order" in body


def test_summary_reports_piles_and_lone_separately(served: str) -> None:
    payload = json.loads(_get(served + "/api/summary")[2])
    assert payload["run"]["n_piles"] == 2
    assert payload["run"]["n_lone"] == 2


def test_piles_come_back_ranked_with_their_faces(served: str) -> None:
    payload = json.loads(_get(served + "/api/piles?offset=0&limit=20")[2])
    assert [p["rank"] for p in payload["piles"]] == [1, 2]
    scores = [p["score"] for p in payload["piles"]]
    assert scores == sorted(scores, reverse=True)
    assert payload["piles"][0]["faces"]


def test_paging_past_the_end_is_empty_not_an_error(served: str) -> None:
    status, _, body = _get(served + "/api/piles?offset=500")
    assert status == 200
    assert json.loads(body)["piles"] == []


def test_unparseable_paging_falls_back_to_defaults(served: str) -> None:
    status, _, body = _get(served + "/api/piles?offset=nonsense&limit=nonsense")
    assert status == 200
    assert len(json.loads(body)["piles"]) == 2


def test_crop_is_served(served: str) -> None:
    status, content_type, body = _get(served + "/crop/1")
    assert status == 200
    assert content_type == "image/jpeg"
    assert body


def test_missing_crop_is_404_not_a_crash(served: str) -> None:
    assert _get(served + "/crop/999999")[0] == 404


def test_bad_face_id_is_400(served: str) -> None:
    assert _get(served + "/crop/not-a-number")[0] == 400


def test_unknown_route_is_404(served: str) -> None:
    assert _get(served + "/nope")[0] == 404


def test_server_refuses_when_no_index_exists(tmp_path: Path) -> None:
    db_path = tmp_path / "bare.db"
    store.connect(db_path).close()
    with pytest.raises(ValueError, match="build_review_index"):
        reviewui.ReviewStore(db_path)


def test_server_names_the_runs_it_has(tmp_path: Path) -> None:
    db_path = _seed(tmp_path, [(4, 70.0)])
    with store.open_index(db_path) as conn:
        review.build_index(conn, model=MODEL, cluster_fn=_fake_clusterer([0, 0, 0, 0]))
    with pytest.raises(ValueError, match=r"r50-components-0\.51"):
        reviewui.ReviewStore(db_path, run_id="does-not-exist")
