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
    assert b"Face review" in body


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


# --------------------------------------------------------------------------------------
# Decisions (v1)
# --------------------------------------------------------------------------------------


def _indexed(tmp_path: Path, plan: list[tuple[int, float]], labels: list[int]) -> Path:
    db_path = _seed(tmp_path, plan)
    with store.open_index(db_path) as conn:
        review.build_index(conn, model=MODEL, cluster_fn=_fake_clusterer(labels))
    return db_path


RUN = review.run_id_for(MODEL, "components", 0.51)


def test_seeding_brings_gold_identities_in_with_their_faces(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(4, 70.0)], [0, 0, 0, 0])
    with store.open_index(db_path) as conn:
        for face_id, person in ((1, "person_1"), (2, "person_1"), (3, "person_2")):
            conn.execute(
                "INSERT INTO gold_labels (face_id,label,person_id,labelled_at) "
                "VALUES (?,'person',?,'now')",
                (face_id, person),
            )
        conn.commit()
        assert review.seed_people_from_gold(conn) == 3
        assert {p["person_id"] for p in review.people(conn)} == {"person_1", "person_2"}
        assert review.people(conn)[0]["n_faces"] == 2


def test_seeding_twice_changes_nothing(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(3, 70.0)], [0, 0, 0])
    with store.open_index(db_path) as conn:
        conn.execute(
            "INSERT INTO gold_labels (face_id,label,person_id,labelled_at) "
            "VALUES (1,'person','person_1','now')"
        )
        conn.commit()
        assert review.seed_people_from_gold(conn) == 1
        assert review.seed_people_from_gold(conn) == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM review_decisions").fetchone()["n"] == 1


def test_seeding_never_overwrites_a_later_human_decision(tmp_path: Path) -> None:
    """Re-seeding must not undo a correction made after the first seed."""
    db_path = _indexed(tmp_path, [(3, 70.0)], [0, 0, 0])
    with store.open_index(db_path) as conn:
        conn.execute(
            "INSERT INTO gold_labels (face_id,label,person_id,labelled_at) "
            "VALUES (1,'person','person_1','now')"
        )
        conn.commit()
        review.seed_people_from_gold(conn)
        review.assign_pile(conn, RUN, 0, name="Anita")
        before = review.people(conn)
        assert review.seed_people_from_gold(conn) == 0
        assert review.people(conn) == before


def test_naming_a_pile_attributes_its_faces(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(5, 70.0)], [0] * 5)
    with store.open_index(db_path) as conn:
        result = review.assign_pile(conn, RUN, 0, name="Anita")
        assert result["n_faces"] == 5
        named = [p for p in review.people(conn) if p["display_name"] == "Anita"]
        assert named and named[0]["n_faces"] == 5


def test_the_same_name_twice_is_the_same_person(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(3, 70.0), (3, 70.0)], [0, 0, 0, 1, 1, 1])
    with store.open_index(db_path) as conn:
        first = review.assign_pile(conn, RUN, 0, name="Anita")
        second = review.assign_pile(conn, RUN, 1, name="anita")
        assert first["person_id"] == second["person_id"]
        assert len(review.people(conn)) == 1
        assert review.people(conn)[0]["n_faces"] == 6


def test_assigning_leaves_already_decided_faces_alone(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(4, 70.0)], [0] * 4)
    with store.open_index(db_path) as conn:
        conn.execute(
            "INSERT INTO review_people (person_id, display_name, origin, created_at) "
            "VALUES ('keep','Keep','review','now')"
        )
        conn.execute(
            "INSERT INTO review_decisions (face_id,kind,person_id,source,batch_id,decided_at) "
            "VALUES (1,'person','keep','manual','b0','now')"
        )
        conn.commit()
        assert review.assign_pile(conn, RUN, 0, name="Anita")["n_faces"] == 3
        counts = {p["display_name"]: p["n_faces"] for p in review.people(conn)}
        assert counts == {"Keep": 1, "Anita": 3}


def test_junking_a_pile_attributes_it_to_nobody(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(4, 70.0)], [0] * 4)
    with store.open_index(db_path) as conn:
        assert review.junk_pile(conn, RUN, 0)["n_faces"] == 4
        assert review.people(conn) == []
        assert review.progress(conn, RUN)["faces_decided"] == 4


def test_next_pile_follows_the_ranking(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(4, 15.0), (6, 75.0)], [0] * 4 + [1] * 6)
    with store.open_index(db_path) as conn:
        assert int(review.next_pile(conn, RUN)["pile_id"]) == 1  # the clear one first


def test_next_pile_moves_on_once_a_pile_is_decided(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(6, 75.0), (4, 70.0)], [0] * 6 + [1] * 4)
    with store.open_index(db_path) as conn:
        first = int(review.next_pile(conn, RUN)["pile_id"])
        review.assign_pile(conn, RUN, first, name="Anita")
        assert int(review.next_pile(conn, RUN)["pile_id"]) != first


def test_a_skipped_pile_is_not_served_again(tmp_path: Path) -> None:
    """The labelling tool re-served the same pile forever because skips lived in the page."""
    db_path = _indexed(tmp_path, [(6, 75.0), (4, 70.0)], [0] * 6 + [1] * 4)
    with store.open_index(db_path) as conn:
        first = int(review.next_pile(conn, RUN)["pile_id"])
        review.skip_pile(conn, RUN, first)
        assert int(review.next_pile(conn, RUN)["pile_id"]) != first
    with store.open_index(db_path) as conn:  # survives a restart
        assert int(review.next_pile(conn, RUN)["pile_id"]) != first


def test_next_pile_is_none_when_everything_is_handled(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(3, 70.0)], [0, 0, 0])
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        assert review.next_pile(conn, RUN) is None


def test_undo_reverses_an_assignment(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(5, 70.0)], [0] * 5)
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        assert review.undo_last(conn, RUN)["n_faces"] == 5
        assert review.progress(conn, RUN)["faces_decided"] == 0
        assert int(review.next_pile(conn, RUN)["pile_id"]) == 0


def test_undo_removes_a_person_it_had_just_created(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(3, 70.0)], [0, 0, 0])
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        review.undo_last(conn, RUN)
        assert review.people(conn) == []


def test_undo_keeps_a_person_who_still_holds_faces(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(3, 70.0), (3, 70.0)], [0, 0, 0, 1, 1, 1])
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        review.assign_pile(conn, RUN, 1, name="Anita")
        review.undo_last(conn, RUN)
        remaining = review.people(conn)
        assert len(remaining) == 1
        assert remaining[0]["n_faces"] == 3


def test_undo_reverses_a_skip_when_that_was_the_last_action(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(6, 75.0), (4, 70.0)], [0] * 6 + [1] * 4)
    with store.open_index(db_path) as conn:
        first = int(review.next_pile(conn, RUN)["pile_id"])
        review.skip_pile(conn, RUN, first)
        assert review.undo_last(conn, RUN)["undone"] == "skip"
        assert int(review.next_pile(conn, RUN)["pile_id"]) == first


def test_undo_on_a_clean_slate_is_harmless(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(3, 70.0)], [0, 0, 0])
    with store.open_index(db_path) as conn:
        assert review.undo_last(conn, RUN)["undone"] is None


def test_undo_never_reaches_the_gold_seed(tmp_path: Path) -> None:
    """Seeded attributions are not a human action, so undo must not peel them off."""
    db_path = _indexed(tmp_path, [(3, 70.0)], [0, 0, 0])
    with store.open_index(db_path) as conn:
        conn.execute(
            "INSERT INTO gold_labels (face_id,label,person_id,labelled_at) "
            "VALUES (1,'person','person_1','now')"
        )
        conn.commit()
        review.seed_people_from_gold(conn)
        assert review.undo_last(conn, RUN)["undone"] is None
        assert review.progress(conn, RUN)["faces_decided"] == 1


def test_reassigning_a_face_moves_it_rather_than_double_counting(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(3, 70.0)], [0, 0, 0])
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        anita = review.people(conn)[0]["person_id"]
        other = review.resolve_person(conn, name="Bharat")
        conn.execute(
            "INSERT INTO review_decisions (face_id,kind,person_id,source,batch_id,decided_at) "
            "VALUES (1,'person',?, 'manual','b9','now')",
            (other,),
        )
        conn.commit()
        counts = {p["person_id"]: p["n_faces"] for p in review.people(conn)}
    assert counts[anita] == 2
    assert counts[other] == 1


def test_renaming_a_person(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(3, 70.0)], [0, 0, 0])
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        person_id = review.people(conn)[0]["person_id"]
        review.rename_person(conn, person_id, "Anita Sharma")
        assert review.people(conn)[0]["display_name"] == "Anita Sharma"


def test_renaming_onto_an_existing_name_is_refused(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(3, 70.0), (3, 70.0)], [0, 0, 0, 1, 1, 1])
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        review.assign_pile(conn, RUN, 1, name="Bharat")
        bharat = next(p for p in review.people(conn) if p["display_name"] == "Bharat")
        with pytest.raises(ValueError, match="already the name"):
            review.rename_person(conn, str(bharat["person_id"]), "Anita")


def test_a_blank_name_is_refused(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(3, 70.0)], [0, 0, 0])
    with (
        store.open_index(db_path) as conn,
        pytest.raises(ValueError, match="either an existing id or a name"),
    ):
        review.assign_pile(conn, RUN, 0, name="   ")


def test_an_unknown_person_id_is_refused(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(3, 70.0)], [0, 0, 0])
    with store.open_index(db_path) as conn, pytest.raises(ValueError, match="No such person"):
        review.assign_pile(conn, RUN, 0, person_id="nobody")


def test_progress_counts_what_it_says(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(4, 70.0), (3, 70.0)], [0] * 4 + [1] * 3)
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        review.skip_pile(conn, RUN, 1)
        state = review.progress(conn, RUN)
    assert state["faces_total"] == 7
    assert state["faces_decided"] == 4
    assert state["people_named"] == 1
    assert state["piles_skipped"] == 1


# --------------------------------------------------------------------------------------
# Server, v1
# --------------------------------------------------------------------------------------


def _post(url: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post_raw(url: str, raw: bytes) -> int:
    request = urllib.request.Request(
        url, data=raw, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


@pytest.fixture
def review_server(tmp_path: Path):
    """Two piles: a clear six-face one, then a blurry four-face one."""
    db_path = _seed(tmp_path, [(6, 75.0), (4, 15.0)])
    with store.open_index(db_path) as conn:
        review.build_index(conn, model=MODEL, cluster_fn=_fake_clusterer([0] * 6 + [1] * 4))

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


def test_next_serves_the_best_pile_first(review_server: str) -> None:
    data = json.loads(_get(review_server + "/api/next")[2])
    assert data["pile"]["pile_id"] == 0
    assert data["pile"]["faces"]
    assert data["progress"]["faces_decided"] == 0


def test_naming_advances_to_the_next_pile(review_server: str) -> None:
    status, result = _post(review_server + "/api/assign", {"pile_id": 0, "name": "Anita"})
    assert status == 200
    assert result["n_faces"] == 6
    assert result["progress"]["faces_decided"] == 6

    data = json.loads(_get(review_server + "/api/next")[2])
    assert data["pile"]["pile_id"] == 1


def test_a_named_person_appears_in_the_picker(review_server: str) -> None:
    _post(review_server + "/api/assign", {"pile_id": 0, "name": "Anita"})
    people = json.loads(_get(review_server + "/api/people")[2])["people"]
    assert [p["display_name"] for p in people] == ["Anita"]
    assert people[0]["n_faces"] == 6


def test_naming_by_person_id_reuses_that_person(review_server: str) -> None:
    _post(review_server + "/api/assign", {"pile_id": 0, "name": "Anita"})
    person_id = json.loads(_get(review_server + "/api/people")[2])["people"][0]["person_id"]
    _post(review_server + "/api/assign", {"pile_id": 1, "person_id": person_id})
    people = json.loads(_get(review_server + "/api/people")[2])["people"]
    assert len(people) == 1
    assert people[0]["n_faces"] == 10


def test_skipping_moves_on_and_undo_brings_it_back(review_server: str) -> None:
    assert _post(review_server + "/api/skip", {"pile_id": 0})[0] == 200
    assert json.loads(_get(review_server + "/api/next")[2])["pile"]["pile_id"] == 1
    assert _post(review_server + "/api/undo", {})[1]["undone"] == "skip"
    assert json.loads(_get(review_server + "/api/next")[2])["pile"]["pile_id"] == 0


def test_junking_a_pile_files_nobody(review_server: str) -> None:
    status, result = _post(review_server + "/api/junk", {"pile_id": 1})
    assert status == 200
    assert result["n_faces"] == 4
    assert json.loads(_get(review_server + "/api/people")[2])["people"] == []


def test_undo_after_naming_restores_the_pile(review_server: str) -> None:
    _post(review_server + "/api/assign", {"pile_id": 0, "name": "Anita"})
    assert _post(review_server + "/api/undo", {})[1]["undone"] == "decision"
    assert json.loads(_get(review_server + "/api/next")[2])["pile"]["pile_id"] == 0
    assert json.loads(_get(review_server + "/api/people")[2])["people"] == []


def test_next_is_null_once_everything_is_handled(review_server: str) -> None:
    _post(review_server + "/api/assign", {"pile_id": 0, "name": "Anita"})
    _post(review_server + "/api/junk", {"pile_id": 1})
    data = json.loads(_get(review_server + "/api/next")[2])
    assert data["pile"] is None
    assert data["progress"]["faces_decided"] == 10


def test_renaming_through_the_api(review_server: str) -> None:
    _post(review_server + "/api/assign", {"pile_id": 0, "name": "Anita"})
    person_id = json.loads(_get(review_server + "/api/people")[2])["people"][0]["person_id"]
    assert _post(review_server + "/api/rename", {"person_id": person_id, "name": "Anita S"})[0] == 200
    assert json.loads(_get(review_server + "/api/people")[2])["people"][0]["display_name"] == "Anita S"


def test_blank_name_is_rejected_without_writing(review_server: str) -> None:
    status, body = _post(review_server + "/api/assign", {"pile_id": 0, "name": "   "})
    assert status == 400
    assert "name" in body["error"]
    assert json.loads(_get(review_server + "/api/next")[2])["progress"]["faces_decided"] == 0


def test_unknown_person_id_is_rejected(review_server: str) -> None:
    status, body = _post(review_server + "/api/assign", {"pile_id": 0, "person_id": "ghost"})
    assert status == 400
    assert "No such person" in body["error"]


def test_missing_field_is_a_400_not_a_500(review_server: str) -> None:
    assert _post(review_server + "/api/assign", {"name": "Anita"})[0] == 400


def test_malformed_json_is_a_400(review_server: str) -> None:
    assert _post_raw(review_server + "/api/assign", b"{not json") == 400
    assert _post_raw(review_server + "/api/assign", b"[1,2,3]") == 400


def test_unknown_post_route_is_404(review_server: str) -> None:
    assert _post(review_server + "/api/nope", {})[0] == 404


def test_repeated_assign_of_the_same_pile_is_harmless(review_server: str) -> None:
    """A double submit from an impatient click must not double-file anything."""
    _post(review_server + "/api/assign", {"pile_id": 0, "name": "Anita"})
    status, second = _post(review_server + "/api/assign", {"pile_id": 0, "name": "Anita"})
    assert status == 200
    assert second["n_faces"] == 0
    assert json.loads(_get(review_server + "/api/people")[2])["people"][0]["n_faces"] == 6


def test_concurrent_writes_do_not_collide(review_server: str) -> None:
    """Handlers run on many threads and SQLite takes one writer."""
    results: list[int] = []

    def hit(pile_id: int) -> None:
        results.append(_post(review_server + "/api/assign", {"pile_id": pile_id, "name": "X"})[0])

    threads = [threading.Thread(target=hit, args=(i % 2,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert results == [200] * 8
    people = json.loads(_get(review_server + "/api/people")[2])["people"]
    assert len(people) == 1
    assert people[0]["n_faces"] == 10  # every face filed exactly once


def test_a_part_decided_pile_reports_who_already_owns_it(tmp_path: Path) -> None:
    """Without this the naming screen hides its most useful fact and one person becomes two."""
    db_path = _indexed(tmp_path, [(6, 70.0)], [0] * 6)
    with store.open_index(db_path) as conn:
        for face_id in (1, 2, 3):
            conn.execute(
                "INSERT INTO gold_labels (face_id,label,person_id,labelled_at) "
                "VALUES (?,'person','person_1','now')",
                (face_id,),
            )
        conn.commit()
        review.seed_people_from_gold(conn)
        already = review.pile_attribution(conn, RUN, 0)
    assert already == [
        {"person_id": "person_1", "display_name": None, "kind": "person", "n_faces": 3}
    ]


def test_attribution_is_empty_for_an_untouched_pile(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(4, 70.0)], [0] * 4)
    with store.open_index(db_path) as conn:
        assert review.pile_attribution(conn, RUN, 0) == []


def test_next_pile_carries_its_attribution_to_the_screen(review_server: str) -> None:
    _post(review_server + "/api/assign", {"pile_id": 0, "name": "Anita"})
    data = json.loads(_get(review_server + "/api/next")[2])
    assert data["pile"]["already"] == []  # pile 1 is untouched


# --------------------------------------------------------------------------------------
# Merge suggestions (v2)
# --------------------------------------------------------------------------------------


def _seed_split_person(tmp_path: Path, *, apart: float = 0.55) -> Path:
    """One person whose faces sit in two sub-groups, plus an unrelated third group.

    This is the measured failure the merge screen exists for: recall 0.906 means a person
    routinely occupies more than one pile. ``apart`` controls how alike the two halves are.
    """
    db_path = tmp_path / "index.db"
    crops = tmp_path / "crops"
    crops.mkdir(exist_ok=True)
    conn = store.connect(db_path)
    conn.execute(
        "INSERT INTO photos (id,path,rel_path,size_bytes,mtime,kind,scanned_at,scan_version)"
        " VALUES (1,'/p/a.jpg','a.jpg',10,0,'photo','now','2')"
    )
    rng = np.random.default_rng(3)

    def unit(v: np.ndarray) -> np.ndarray:
        return embed.l2_normalise(np.asarray(v, dtype=np.float32)[None, :])[0]

    shared = unit(rng.normal(size=embed.EMBED_DIM))
    half_a = unit(apart * shared + (1 - apart) * unit(rng.normal(size=embed.EMBED_DIM)))
    half_b = unit(apart * shared + (1 - apart) * unit(rng.normal(size=embed.EMBED_DIM)))
    stranger = unit(rng.normal(size=embed.EMBED_DIM))

    face_id = 0
    for centre, count in ((half_a, 6), (half_b, 5), (stranger, 4)):
        for _ in range(count):
            face_id += 1
            vector = unit(centre + 0.033 * rng.normal(size=embed.EMBED_DIM).astype(np.float32))
            _insert(conn, crops, face_id, vector, 70.0)
    conn.commit()
    conn.close()
    return db_path


def test_a_split_person_is_suggested_for_merging(tmp_path: Path) -> None:
    db_path = _seed_split_person(tmp_path)
    with store.open_index(db_path) as conn:
        review.build_index(
            conn, model=MODEL, cluster_fn=_fake_clusterer([0] * 6 + [1] * 5 + [2] * 4)
        )
        review.assign_pile(conn, RUN, 0, name="Anita")
        found = review.merge_candidates(conn, RUN, limit=3)
    assert found, "the other half of Anita should have been offered"
    assert found[0]["display_name"] == "Anita"
    assert int(found[0]["pile"]["pile_id"]) == 1  # her other half, not the stranger
    assert found[0]["similarity"] > found[-1]["similarity"] or len(found) == 1


def test_accepting_a_suggestion_files_the_faces(tmp_path: Path) -> None:
    db_path = _seed_split_person(tmp_path)
    with store.open_index(db_path) as conn:
        review.build_index(
            conn, model=MODEL, cluster_fn=_fake_clusterer([0] * 6 + [1] * 5 + [2] * 4)
        )
        review.assign_pile(conn, RUN, 0, name="Anita")
        review.assign_pile(conn, RUN, 1, person_id=review.people(conn)[0]["person_id"], source="merge")
        assert review.people(conn)[0]["n_faces"] == 11


def test_a_rejected_pair_is_never_offered_again(tmp_path: Path) -> None:
    db_path = _seed_split_person(tmp_path)
    with store.open_index(db_path) as conn:
        review.build_index(
            conn, model=MODEL, cluster_fn=_fake_clusterer([0] * 6 + [1] * 5 + [2] * 4)
        )
        review.assign_pile(conn, RUN, 0, name="Anita")
        first = review.merge_candidates(conn, RUN, limit=1)[0]
        review.reject_merge(conn, RUN, str(first["person_id"]), int(first["pile"]["pile_id"]))
        again = review.merge_candidates(conn, RUN, limit=5)
    assert all(int(c["pile"]["pile_id"]) != int(first["pile"]["pile_id"]) for c in again)


def test_undo_reverses_a_rejection(tmp_path: Path) -> None:
    db_path = _seed_split_person(tmp_path)
    with store.open_index(db_path) as conn:
        review.build_index(
            conn, model=MODEL, cluster_fn=_fake_clusterer([0] * 6 + [1] * 5 + [2] * 4)
        )
        review.assign_pile(conn, RUN, 0, name="Anita")
        first = review.merge_candidates(conn, RUN, limit=1)[0]
        review.reject_merge(conn, RUN, str(first["person_id"]), int(first["pile"]["pile_id"]))
        assert review.undo_last(conn, RUN)["undone"] == "rejection"
        back = review.merge_candidates(conn, RUN, limit=1)[0]
    assert int(back["pile"]["pile_id"]) == int(first["pile"]["pile_id"])


def test_no_suggestions_before_anyone_is_named(tmp_path: Path) -> None:
    db_path = _seed_split_person(tmp_path)
    with store.open_index(db_path) as conn:
        review.build_index(
            conn, model=MODEL, cluster_fn=_fake_clusterer([0] * 6 + [1] * 5 + [2] * 4)
        )
        assert review.merge_candidates(conn, RUN, limit=5) == []


def test_a_fully_decided_pile_is_not_suggested(tmp_path: Path) -> None:
    """There would be nothing to gain, and it would waste the human's attention."""
    db_path = _seed_split_person(tmp_path)
    with store.open_index(db_path) as conn:
        review.build_index(
            conn, model=MODEL, cluster_fn=_fake_clusterer([0] * 6 + [1] * 5 + [2] * 4)
        )
        review.assign_pile(conn, RUN, 0, name="Anita")
        review.assign_pile(conn, RUN, 1, name="Bharat")
        review.junk_pile(conn, RUN, 2)
        assert review.merge_candidates(conn, RUN, limit=5) == []


def test_a_weak_pair_is_not_offered(tmp_path: Path) -> None:
    db_path = _seed_split_person(tmp_path)
    with store.open_index(db_path) as conn:
        review.build_index(
            conn, model=MODEL, cluster_fn=_fake_clusterer([0] * 6 + [1] * 5 + [2] * 4)
        )
        review.assign_pile(conn, RUN, 0, name="Anita")
        assert review.merge_candidates(conn, RUN, limit=5, min_similarity=0.999) == []


def test_person_centroid_ignores_people_with_no_faces(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(4, 70.0)], [0] * 4)
    with store.open_index(db_path) as conn:
        conn.execute(
            "INSERT INTO review_people (person_id, display_name, origin, created_at) "
            "VALUES ('ghost','Ghost','review','now')"
        )
        conn.commit()
        ids, matrix = review.person_centroids(conn, RUN)
    assert "ghost" not in ids
    assert len(ids) == matrix.shape[0]


def test_person_centroids_are_unit_vectors(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(4, 70.0)], [0] * 4)
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        _ids, matrix = review.person_centroids(conn, RUN)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-5)


@pytest.fixture
def merge_server(tmp_path: Path):
    db_path = _seed_split_person(tmp_path)
    with store.open_index(db_path) as conn:
        review.build_index(
            conn, model=MODEL, cluster_fn=_fake_clusterer([0] * 6 + [1] * 5 + [2] * 4)
        )
        review.assign_pile(conn, RUN, 0, name="Anita")
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


def test_merge_endpoint_offers_a_pair(merge_server: str) -> None:
    data = json.loads(_get(merge_server + "/api/merge")[2])
    assert data["suggestion"]["display_name"] == "Anita"
    assert data["suggestion"]["person_faces"]
    assert data["suggestion"]["pile_faces"]


def test_accepting_through_the_api(merge_server: str) -> None:
    suggestion = json.loads(_get(merge_server + "/api/merge")[2])["suggestion"]
    status, result = _post(
        merge_server + "/api/merge/accept",
        {"person_id": suggestion["person_id"], "pile_id": suggestion["pile"]["pile_id"]},
    )
    assert status == 200
    assert result["n_faces"] == 5
    people = json.loads(_get(merge_server + "/api/people")[2])["people"]
    assert people[0]["n_faces"] == 11


def test_rejecting_through_the_api_retires_the_pair(merge_server: str) -> None:
    suggestion = json.loads(_get(merge_server + "/api/merge")[2])["suggestion"]
    assert (
        _post(
            merge_server + "/api/merge/reject",
            {"person_id": suggestion["person_id"], "pile_id": suggestion["pile"]["pile_id"]},
        )[0]
        == 200
    )
    nxt = json.loads(_get(merge_server + "/api/merge")[2])["suggestion"]
    assert nxt is None or nxt["pile"]["pile_id"] != suggestion["pile"]["pile_id"]


def test_merge_endpoints_reject_malformed_bodies(merge_server: str) -> None:
    assert _post(merge_server + "/api/merge/accept", {"pile_id": 1})[0] == 400
    assert _post(merge_server + "/api/merge/reject", {"person_id": "x"})[0] == 400
    assert _post(merge_server + "/api/merge/accept", {"person_id": "ghost", "pile_id": 1})[0] == 400


# --------------------------------------------------------------------------------------
# Repairs, bucket and search (v3)
# --------------------------------------------------------------------------------------


def test_a_person_lists_their_faces_clearest_first(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(4, 70.0)], [0] * 4)
    with store.open_index(db_path) as conn:
        conn.execute("UPDATE faces SET interocular_px = 95 WHERE id = 3")
        conn.commit()
        review.assign_pile(conn, RUN, 0, name="Anita")
        person_id = review.people(conn)[0]["person_id"]
        detail = review.person_faces(conn, str(person_id))
    assert detail["n_faces"] == 4
    assert detail["faces"][0]["face_id"] == 3


def test_person_faces_pages(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(6, 70.0)], [0] * 6)
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        person_id = str(review.people(conn)[0]["person_id"])
        first = review.person_faces(conn, person_id, offset=0, limit=4)
        second = review.person_faces(conn, person_id, offset=4, limit=4)
    assert len(first["faces"]) == 4
    assert len(second["faces"]) == 2
    assert not {f["face_id"] for f in first["faces"]} & {f["face_id"] for f in second["faces"]}


def test_an_unknown_person_is_refused(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(3, 70.0)], [0] * 3)
    with store.open_index(db_path) as conn, pytest.raises(ValueError, match="No such person"):
        review.person_faces(conn, "ghost")


def test_removing_a_face_returns_it_to_the_queue(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(4, 70.0)], [0] * 4)
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        person_id = str(review.people(conn)[0]["person_id"])
        assert review.next_pile(conn, RUN) is None
        review.remove_faces(conn, person_id, [1])
        assert review.people(conn)[0]["n_faces"] == 3
        assert review.undecided_faces(conn, RUN, 0) == [1]
        assert review.next_pile(conn, RUN) is not None


def test_a_removed_face_is_never_suggested_back_to_that_person(tmp_path: Path) -> None:
    db_path = _seed_split_person(tmp_path)
    with store.open_index(db_path) as conn:
        review.build_index(
            conn, model=MODEL, cluster_fn=_fake_clusterer([0] * 6 + [1] * 5 + [2] * 4)
        )
        review.assign_pile(conn, RUN, 0, name="Anita")
        review.assign_pile(conn, RUN, 1, person_id=str(review.people(conn)[0]["person_id"]))
        person_id = str(review.people(conn)[0]["person_id"])
        review.remove_faces(conn, person_id, [7, 8, 9, 10, 11])
        offered = review.merge_candidates(conn, RUN, limit=5)
    assert all(int(c["pile"]["pile_id"]) != 1 for c in offered)


def test_removing_faces_is_undoable_in_full(tmp_path: Path) -> None:
    """One action wrote to two tables; undo must clear both or the face stays blacklisted."""
    db_path = _indexed(tmp_path, [(4, 70.0)], [0] * 4)
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        person_id = str(review.people(conn)[0]["person_id"])
        review.remove_faces(conn, person_id, [1, 2])
        review.undo_last(conn, RUN)
        assert review.people(conn)[0]["n_faces"] == 4
        assert conn.execute("SELECT COUNT(*) AS n FROM review_not_person").fetchone()["n"] == 0


def test_splitting_a_person_moves_faces_to_a_new_one(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(6, 70.0)], [0] * 6)
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        person_id = str(review.people(conn)[0]["person_id"])
        review.split_person(conn, person_id, [1, 2], "Bharat")
        counts = {p["display_name"]: p["n_faces"] for p in review.people(conn)}
    assert counts == {"Anita": 4, "Bharat": 2}


def test_splitting_onto_the_same_person_is_refused(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(4, 70.0)], [0] * 4)
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        person_id = str(review.people(conn)[0]["person_id"])
        with pytest.raises(ValueError, match="same person"):
            review.split_person(conn, person_id, [1], "Anita")


def test_splitting_nothing_is_refused(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(4, 70.0)], [0] * 4)
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        person_id = str(review.people(conn)[0]["person_id"])
        with pytest.raises(ValueError, match="at least one face"):
            review.split_person(conn, person_id, [], "Bharat")


def test_setting_aside_and_restoring_individual_faces(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(4, 70.0)], [0] * 4)
    with store.open_index(db_path) as conn:
        review.set_aside(conn, [1, 2])
        assert review.bucket(conn, RUN, kind="junk")["n_faces"] == 2
        assert review.undecided_faces(conn, RUN, 0) == [3, 4]
        review.restore_faces(conn, [1])
        assert review.bucket(conn, RUN, kind="junk")["n_faces"] == 1
        assert sorted(review.undecided_faces(conn, RUN, 0)) == [1, 3, 4]


def test_the_lone_bucket_holds_what_never_matched(tmp_path: Path) -> None:
    db_path = _seed(tmp_path, [(4, 70.0)], n_lone=3)
    with store.open_index(db_path) as conn:
        review.build_index(
            conn, model=MODEL, cluster_fn=_fake_clusterer([0, 0, 0, 0, -1, -1, -1])
        )
        lone = review.bucket(conn, RUN, kind="lone")
    assert lone["n_faces"] == 3
    assert {f["face_id"] for f in lone["faces"]} == {5, 6, 7}


def test_an_unknown_bucket_is_refused(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(3, 70.0)], [0] * 3)
    with store.open_index(db_path) as conn, pytest.raises(ValueError, match="junk"):
        review.bucket(conn, RUN, kind="nonsense")


def test_search_matches_name_and_id(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(3, 70.0), (3, 70.0)], [0, 0, 0, 1, 1, 1])
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita Sharma")
        review.assign_pile(conn, RUN, 1, name="Bharat")
        assert [p["display_name"] for p in review.search_people(conn, "anita")] == ["Anita Sharma"]
        assert [p["display_name"] for p in review.search_people(conn, "SHARMA")] == ["Anita Sharma"]
        assert len(review.search_people(conn, "")) == 2
        assert review.search_people(conn, "nobody") == []


def test_progress_stops_counting_a_face_taken_off_a_person(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(4, 70.0)], [0] * 4)
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        assert review.progress(conn, RUN)["faces_decided"] == 4
        review.remove_faces(conn, str(review.people(conn)[0]["person_id"]), [1])
        assert review.progress(conn, RUN)["faces_decided"] == 3


# --------------------------------------------------------------------------------------
# Server, v3
# --------------------------------------------------------------------------------------


@pytest.fixture
def repair_server(tmp_path: Path):
    """Anita owns a six-face pile; a four-face pile and three lone faces are untouched."""
    db_path = _seed(tmp_path, [(6, 75.0), (4, 40.0)], n_lone=3)
    with store.open_index(db_path) as conn:
        review.build_index(
            conn, model=MODEL, cluster_fn=_fake_clusterer([0] * 6 + [1] * 4 + [-1] * 3)
        )
        review.assign_pile(conn, RUN, 0, name="Anita")
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


def _person_id(base: str) -> str:
    return json.loads(_get(base + "/api/people")[2])["people"][0]["person_id"]


def test_person_detail_endpoint(repair_server: str) -> None:
    pid = _person_id(repair_server)
    data = json.loads(_get(repair_server + f"/api/person?person_id={pid}")[2])
    assert data["display_name"] == "Anita"
    assert data["n_faces"] == 6
    assert len(data["faces"]) == 6


def test_person_detail_for_a_stranger_is_400(repair_server: str) -> None:
    status, _, body = _get(repair_server + "/api/person?person_id=ghost")
    assert status == 400
    assert "No such person" in json.loads(body)["error"]


def test_removing_faces_through_the_api(repair_server: str) -> None:
    pid = _person_id(repair_server)
    status, result = _post(
        repair_server + "/api/person/remove", {"person_id": pid, "face_ids": [1, 2]}
    )
    assert status == 200
    assert result["n_faces"] == 2
    assert json.loads(_get(repair_server + f"/api/person?person_id={pid}")[2])["n_faces"] == 4
    assert json.loads(_get(repair_server + "/api/next")[2])["pile"]["pile_id"] == 0


def test_splitting_through_the_api(repair_server: str) -> None:
    pid = _person_id(repair_server)
    status, result = _post(
        repair_server + "/api/person/split",
        {"person_id": pid, "face_ids": [1, 2], "name": "Bharat"},
    )
    assert status == 200
    assert result["n_faces"] == 2
    counts = {
        p["display_name"]: p["n_faces"]
        for p in json.loads(_get(repair_server + "/api/people")[2])["people"]
    }
    assert counts == {"Anita": 4, "Bharat": 2}


def test_bucket_endpoints(repair_server: str) -> None:
    assert json.loads(_get(repair_server + "/api/bucket?kind=lone")[2])["n_faces"] == 3
    assert json.loads(_get(repair_server + "/api/bucket?kind=junk")[2])["n_faces"] == 0
    _post(repair_server + "/api/faces/aside", {"face_ids": [7, 8]})
    assert json.loads(_get(repair_server + "/api/bucket?kind=junk")[2])["n_faces"] == 2


def test_restoring_from_the_bucket(repair_server: str) -> None:
    _post(repair_server + "/api/faces/aside", {"face_ids": [7, 8]})
    status, result = _post(repair_server + "/api/faces/restore", {"face_ids": [7]})
    assert status == 200
    assert result["n_faces"] == 1
    assert json.loads(_get(repair_server + "/api/bucket?kind=junk")[2])["n_faces"] == 1


def test_an_unknown_bucket_is_400(repair_server: str) -> None:
    assert _get(repair_server + "/api/bucket?kind=nonsense")[0] == 400


def test_search_endpoint(repair_server: str) -> None:
    found = json.loads(_get(repair_server + "/api/search?q=ani")[2])["people"]
    assert [p["display_name"] for p in found] == ["Anita"]
    assert json.loads(_get(repair_server + "/api/search?q=zzz")[2])["people"] == []


def test_face_id_lists_are_validated(repair_server: str) -> None:
    pid = _person_id(repair_server)
    assert _post(repair_server + "/api/person/remove", {"person_id": pid})[0] == 400
    assert (
        _post(repair_server + "/api/person/remove", {"person_id": pid, "face_ids": "1,2"})[0]
        == 400
    )
    assert _post(repair_server + "/api/faces/aside", {"face_ids": ["x"]})[0] == 400


def test_splitting_with_no_name_is_400(repair_server: str) -> None:
    pid = _person_id(repair_server)
    status, body = _post(
        repair_server + "/api/person/split", {"person_id": pid, "face_ids": [1], "name": " "}
    )
    assert status == 400
    assert "name" in body["error"]


def test_every_screen_boots_against_the_same_server(repair_server: str) -> None:
    """A smoke test across all of v0-v3: nothing 500s on a fresh database."""
    for path in (
        "/",
        "/api/summary",
        "/api/next",
        "/api/people",
        "/api/merge",
        "/api/piles?offset=0&limit=5",
        "/api/bucket?kind=junk",
        "/api/bucket?kind=lone",
        "/api/search?q=",
    ):
        status, _, _ = _get(repair_server + path)
        assert status == 200, path


def test_removing_only_touches_faces_that_person_owns(tmp_path: Path) -> None:
    """Found by a walk-through: the endpoint unassigned another person's face."""
    db_path = _indexed(tmp_path, [(6, 70.0)], [0] * 6)
    with store.open_index(db_path) as conn:
        conn.execute(
            "INSERT INTO review_people (person_id, display_name, origin, created_at) "
            "VALUES ('other','Other','review','now')"
        )
        conn.execute(
            "INSERT INTO review_decisions (face_id,kind,person_id,source,batch_id,decided_at) "
            "VALUES (1,'person','other','manual','b0','now')"
        )
        conn.commit()
        review.assign_pile(conn, RUN, 0, name="Anita")
        anita = next(p["person_id"] for p in review.people(conn) if p["display_name"] == "Anita")

        result = review.remove_faces(conn, str(anita), [1, 2])
        counts = {p["display_name"]: p["n_faces"] for p in review.people(conn)}
        marked = conn.execute("SELECT COUNT(*) AS n FROM review_not_person").fetchone()["n"]

    assert result["n_faces"] == 1  # face 2 only; face 1 is Other's
    assert counts == {"Other": 1, "Anita": 4}
    assert marked == 1


def test_splitting_only_moves_faces_that_person_owns(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(6, 70.0)], [0] * 6)
    with store.open_index(db_path) as conn:
        conn.execute(
            "INSERT INTO review_people (person_id, display_name, origin, created_at) "
            "VALUES ('other','Other','review','now')"
        )
        conn.execute(
            "INSERT INTO review_decisions (face_id,kind,person_id,source,batch_id,decided_at) "
            "VALUES (1,'person','other','manual','b0','now')"
        )
        conn.commit()
        review.assign_pile(conn, RUN, 0, name="Anita")
        anita = next(p["person_id"] for p in review.people(conn) if p["display_name"] == "Anita")
        assert review.split_person(conn, str(anita), [1, 2], "Bharat")["n_faces"] == 1
        counts = {p["display_name"]: p["n_faces"] for p in review.people(conn)}
    assert counts == {"Other": 1, "Anita": 4, "Bharat": 1}


def test_splitting_none_of_a_persons_faces_is_refused(tmp_path: Path) -> None:
    db_path = _indexed(tmp_path, [(4, 70.0)], [0] * 4)
    with store.open_index(db_path) as conn:
        review.assign_pile(conn, RUN, 0, name="Anita")
        anita = str(review.people(conn)[0]["person_id"])
        with pytest.raises(ValueError, match="belong to that person"):
            review.split_person(conn, anita, [9999], "Bharat")
