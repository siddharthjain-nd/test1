"""The review layer: rank the clustering's piles so a human meets the useful ones first.

At cosine 0.51 the pool breaks into thousands of groups, most of them one or two faces too
small to recognise. Presenting them in cluster-id order would bury a few hundred real people
among them, and the tool would be unusable however good its buttons were. Ordering is
therefore not a nicety here, it is the feature.

Nothing is discarded. Poor piles sort to the bottom and lone faces go to their own bucket,
both still reachable. The measured reason: tiny faces score 0.887 and bad-quality faces
0.515, so they are worth hiding from the front of the queue -- but 61% of tiny faces were
still readable when labelled by hand, so throwing them away would lose real people.

This module computes and stores that order. It never edits a cluster.
"""

from __future__ import annotations

import math
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from faceindex import cluster, embed

# An interocular distance at which a face is comfortably recognisable. Measured context:
# the library's median is 31 px, the one clean large pile sat at 72 px, and the blob that
# welded fifteen people together at cosine 0.49 sat at 22 px. Fifty is between the library
# and the good pile, and it is a dial, not a law.
READABLE_AT_PX = 50.0

# Below this a group is not a pile anyone would review; it is a face that matched nothing.
MIN_PILE = 2


@dataclass
class RunSummary:
    run_id: str
    model: str
    algorithm: str
    threshold: float
    n_faces: int
    n_piles: int
    n_lone: int


def run_id_for(model: str, algorithm: str, threshold: float) -> str:
    """Deterministic, readable, and stable across re-runs of the same configuration.

    Re-indexing the same settings replaces its own rows rather than accumulating copies,
    while a different model or threshold gets its own run and the two can coexist.
    """
    short = model.replace("w600k_", "").replace(".onnx", "")
    return f"{short}-{algorithm}-{threshold:g}"


def score_pile(n_faces: int, median_eye: float | None, coherence: float | None) -> float:
    """How much of your attention this pile deserves, as a product of three readable parts.

    * reach -- a 200-face pile is worth more of one naming action than a 2-face pile, but
      only logarithmically: the hundredth face of a person adds little.
    * readable -- a pile of faces too small to recognise cannot be named at all, whatever
      its size.
    * coherent -- faces that sit close together are probably one person; a loose pile is
      probably several, and answering "who is this?" about it is the wrong question.

    A product, not a sum, so that a zero in any one of them sinks the pile. A huge pile of
    unrecognisable faces is not a good pile.
    """
    reach = math.log10(1.0 + max(n_faces, 0))
    legible = min(max((median_eye or 0.0) / READABLE_AT_PX, 0.0), 1.0)
    coherent = min(max(coherence if coherence is not None else 0.0, 0.0), 1.0)
    # Squared, because a linear term let size win outright: 500 faces at 18 px outranked a
    # clear pile, and 500 faces nobody can identify is worth nothing at all. It enters twice
    # on purpose -- readability decides both whether the pile can be named and whether its
    # faces are worth having. The shape matches the labelling data, where unreadability rose
    # 3.3% -> 16.4% -> 39.1% across medium, small and tiny faces rather than linearly.
    return reach * legible * legible * coherent


def build_index(
    conn: sqlite3.Connection,
    *,
    model: str,
    algorithm: str = "components",
    threshold: float = 0.51,
    neighbors: int = 50,
    jobs: int | None = None,
    min_cluster_size: int = 3,
    epsilon: float = 0.0,
    cluster_fn: Callable[[np.ndarray, cluster.ClusterConfig], cluster.ClusterResult] | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> RunSummary:
    """Cluster the pool, score every pile, and store the review order."""
    say = on_stage or (lambda _message: None)

    say(f"loading embeddings for {model}")
    face_ids, matrix = cluster.load_embeddings(conn, model=model)
    if len(face_ids) < MIN_PILE:
        raise ValueError(f"Only {len(face_ids)} embedded faces for {model}; nothing to index.")

    config = cluster.ClusterConfig(
        algorithm=algorithm,
        similarity_threshold=threshold,
        selection_epsilon=epsilon,
        min_cluster_size=min_cluster_size,
        n_neighbors=neighbors,
        n_jobs=jobs if jobs is not None else cluster.default_jobs(),
    )
    say(f"clustering {len(face_ids):,} faces")
    result = (cluster_fn or cluster.bootstrap_cluster)(matrix, config)
    labels = np.asarray(result.labels)

    say("reading face quality")
    quality = {
        int(r["id"]): (r["interocular_px"], r["det_score"])
        for r in conn.execute("SELECT id, interocular_px, det_score FROM faces")
    }

    say("scoring piles")
    features = embed.l2_normalise(matrix.astype(np.float32))
    # np.split cuts `order`, so each group holds indices into the ORIGINAL arrays. The
    # sorted view is only there to find the cut points -- reading a label out of it with one
    # of these indices silently returns some other pile's label, which is exactly the bug
    # this comment exists to prevent a second time.
    order = np.argsort(labels, kind="stable")
    boundaries = np.flatnonzero(np.diff(labels[order])) + 1
    groups = np.split(order, boundaries)

    piles: list[tuple[int, int, float, float | None, float | None, bytes]] = []
    members: list[tuple[int, int, int]] = []
    n_lone = 0

    for group in groups:
        if len(group) == 0:
            continue
        pile_id = int(labels[group[0]])

        if pile_id < 0 or len(group) < MIN_PILE:
            # Ungrouped. Recorded so the bucket is browsable, but never ranked as a pile.
            n_lone += len(group)
            for index in group:
                members.append((int(face_ids[index]), -1, 0))
            continue

        block = features[group]
        centroid = block.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        coherence = float((block @ (centroid / norm)).mean()) if norm > 1e-9 else 0.0

        eyes = [quality.get(int(face_ids[i]), (None, None))[0] for i in group]
        usable = [float(e) for e in eyes if e is not None]
        median_eye = float(np.median(usable)) if usable else None

        # Most recognisable face first, so the grid leads with faces worth looking at.
        ranked = sorted(
            group,
            key=lambda i: (
                -(quality.get(int(face_ids[i]), (0.0, 0.0))[0] or 0.0),
                -(quality.get(int(face_ids[i]), (0.0, 0.0))[1] or 0.0),
                int(face_ids[i]),
            ),
        )
        for position, index in enumerate(ranked):
            members.append((int(face_ids[index]), pile_id, position))

        piles.append(
            (
                pile_id,
                len(group),
                score_pile(len(group), median_eye, coherence),
                median_eye,
                coherence,
                embed.to_blob(centroid / norm if norm > 1e-9 else centroid),
            )
        )

    run_id = run_id_for(model, algorithm, threshold)
    say(f"writing {len(piles):,} piles")

    conn.execute("DELETE FROM review_members WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM review_piles WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM review_runs WHERE run_id = ?", (run_id,))
    conn.execute(
        "INSERT INTO review_runs (run_id, model, algorithm, threshold, n_faces, n_piles, "
        "n_lone, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            run_id,
            model,
            algorithm,
            threshold,
            len(face_ids),
            len(piles),
            n_lone,
            datetime.now(UTC).isoformat(),
        ),
    )
    conn.executemany(
        "INSERT INTO review_piles (run_id, pile_id, n_faces, score, median_eye, coherence, "
        "centroid) VALUES (?,?,?,?,?,?,?)",
        [(run_id, *row) for row in piles],
    )
    conn.executemany(
        "INSERT INTO review_members (run_id, face_id, pile_id, position) VALUES (?,?,?,?)",
        [(run_id, *row) for row in members],
    )
    conn.commit()

    return RunSummary(
        run_id=run_id,
        model=model,
        algorithm=algorithm,
        threshold=threshold,
        n_faces=len(face_ids),
        n_piles=len(piles),
        n_lone=n_lone,
    )


# --------------------------------------------------------------------------------------
# Reading, for the server
# --------------------------------------------------------------------------------------


def runs(conn: sqlite3.Connection) -> list[dict[str, object]]:
    return [dict(r) for r in conn.execute("SELECT * FROM review_runs ORDER BY created_at DESC")]


def latest_run(conn: sqlite3.Connection) -> dict[str, object] | None:
    found = runs(conn)
    return found[0] if found else None


def list_piles(
    conn: sqlite3.Connection, run_id: str, *, offset: int = 0, limit: int = 20
) -> list[dict[str, object]]:
    """Piles in review order, best first. Ties broken by size then id so paging is stable."""
    return [
        dict(r)
        for r in conn.execute(
            "SELECT pile_id, n_faces, score, median_eye, coherence FROM review_piles "
            "WHERE run_id = ? ORDER BY score DESC, n_faces DESC, pile_id ASC LIMIT ? OFFSET ?",
            (run_id, max(limit, 0), max(offset, 0)),
        )
    ]


def pile_faces(
    conn: sqlite3.Connection, run_id: str, pile_id: int, *, limit: int = 24
) -> list[dict[str, object]]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT m.face_id, f.interocular_px, f.det_score, f.blur "
            "FROM review_members m JOIN faces f ON f.id = m.face_id "
            "WHERE m.run_id = ? AND m.pile_id = ? ORDER BY m.position LIMIT ?",
            (run_id, pile_id, max(limit, 0)),
        )
    ]


def known_people(conn: sqlite3.Connection) -> list[dict[str, object]]:
    """The people already named during labelling, newest-largest first.

    Reused deliberately: naming from scratch would mean retyping 124 identities that are
    already recorded. The gold set stays read-only -- this is a lookup, never a write.
    """
    return [
        {"person_id": str(r["person_id"]), "n_faces": int(r["n"]), "sample": int(r["sample"])}
        for r in conn.execute(
            "SELECT person_id, COUNT(*) AS n, MIN(face_id) AS sample FROM gold_labels "
            "WHERE label = 'person' AND person_id IS NOT NULL AND person_id <> '' "
            "GROUP BY person_id ORDER BY n DESC"
        )
    ]


def crop_path(conn: sqlite3.Connection, face_id: int, *, context: bool) -> Path | None:
    row = conn.execute(
        "SELECT crop_path, context_path FROM faces WHERE id = ?", (face_id,)
    ).fetchone()
    if row is None:
        return None
    chosen = row["context_path"] if context else row["crop_path"]
    if not chosen:
        chosen = row["crop_path"]
    return Path(chosen) if chosen else None


# --------------------------------------------------------------------------------------
# Decisions (v1)
#
# Append-only, keyed on face ids. The current decision for a face is its highest row, so an
# undo is a delete of one batch rather than a reconstruction of what came before, and a
# re-clustering cannot invalidate anything: pile ids change, face ids never do.
# --------------------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_batch() -> str:
    return uuid.uuid4().hex


def seed_people_from_gold(conn: sqlite3.Connection) -> int:
    """Copy the gold set's identities in as people, with their faces already attributed.

    Deliberate (Siddharth's call): 124 identities and 1,499 faces are already decided, so
    review starts from them rather than from nothing, and merge suggestions have anchors
    immediately. The gold tables are only read -- they stay the frozen yardstick.

    Idempotent: a face already carrying any decision is left alone, so re-running never
    overwrites a human's later correction with the original gold answer.
    """
    rows = conn.execute(
        "SELECT face_id, person_id FROM gold_labels "
        "WHERE label = 'person' AND person_id IS NOT NULL AND person_id <> ''"
    ).fetchall()
    if not rows:
        return 0

    now = _now()
    conn.executemany(
        "INSERT OR IGNORE INTO review_people (person_id, display_name, origin, created_at) "
        "VALUES (?, NULL, 'gold', ?)",
        sorted({(str(r["person_id"]), now) for r in rows}),
    )
    already = {
        int(r["face_id"]) for r in conn.execute("SELECT DISTINCT face_id FROM review_decisions")
    }
    fresh = [(int(r["face_id"]), str(r["person_id"])) for r in rows if int(r["face_id"]) not in already]
    conn.executemany(
        "INSERT INTO review_decisions (face_id, kind, person_id, source, batch_id, decided_at) "
        "VALUES (?, 'person', ?, 'gold', 'seed', ?)",
        [(face_id, person_id, now) for face_id, person_id in fresh],
    )
    conn.commit()
    return len(fresh)


def resolve_person(
    conn: sqlite3.Connection, *, person_id: str | None = None, name: str | None = None
) -> str:
    """Return the person id to attribute faces to, creating one if this is a new name."""
    if person_id:
        row = conn.execute(
            "SELECT person_id FROM review_people WHERE person_id = ?", (person_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No such person: {person_id}")
        return str(row["person_id"])

    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("A person needs either an existing id or a name.")

    existing = conn.execute(
        "SELECT person_id FROM review_people WHERE display_name = ? COLLATE NOCASE", (cleaned,)
    ).fetchone()
    if existing is not None:
        return str(existing["person_id"])

    # A readable, collision-free id. Names can be edited later; the id never changes.
    new_id = f"p_{uuid.uuid4().hex[:10]}"
    conn.execute(
        "INSERT INTO review_people (person_id, display_name, origin, created_at) "
        "VALUES (?, ?, 'review', ?)",
        (new_id, cleaned, _now()),
    )
    return new_id


def rename_person(conn: sqlite3.Connection, person_id: str, name: str) -> None:
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("A name cannot be blank.")
    clash = conn.execute(
        "SELECT person_id FROM review_people WHERE display_name = ? COLLATE NOCASE "
        "AND person_id <> ?",
        (cleaned, person_id),
    ).fetchone()
    if clash is not None:
        raise ValueError(f"{cleaned!r} is already the name of {clash['person_id']}.")
    changed = conn.execute(
        "UPDATE review_people SET display_name = ? WHERE person_id = ?", (cleaned, person_id)
    ).rowcount
    if not changed:
        raise ValueError(f"No such person: {person_id}")
    conn.commit()


# A face is open if nothing has been said about it, or if the last thing said was that it
# was taken back off a person. Deleting the row instead would lose the history and, with it,
# the record of why the face is loose again.
_OPEN_FACE = (
    "(NOT EXISTS (SELECT 1 FROM review_decisions d WHERE d.face_id = {face}) "
    " OR (SELECT kind FROM review_decisions d WHERE d.face_id = {face} "
    "     ORDER BY d.id DESC LIMIT 1) = 'unassigned')"
)


def undecided_faces(conn: sqlite3.Connection, run_id: str, pile_id: int) -> list[int]:
    return [
        int(r["face_id"])
        for r in conn.execute(
            "SELECT m.face_id FROM review_members m WHERE m.run_id = ? AND m.pile_id = ? AND "
            + _OPEN_FACE.format(face="m.face_id")
            + " ORDER BY m.position",
            (run_id, pile_id),
        )
    ]


def pile_attribution(conn: sqlite3.Connection, run_id: str, pile_id: int) -> list[dict[str, object]]:
    """Who the already-decided faces in this pile belong to, biggest share first.

    Without this the naming screen hides the most useful fact on it. A pile can arrive
    part-attributed -- from the gold seed, or from an earlier merge -- and a human who
    cannot see that types a fresh name and splits one person into two. Showing it turns the
    common case into confirming a suggestion rather than recalling a name.
    """
    return [
        {
            "person_id": None if r["person_id"] is None else str(r["person_id"]),
            "display_name": r["display_name"],
            "kind": str(r["kind"]),
            "n_faces": int(r["n"]),
        }
        for r in conn.execute(
            "SELECT d.kind, d.person_id, p.display_name, COUNT(*) AS n "
            "FROM review_members m "
            "JOIN review_decisions d ON d.id = "
            "  (SELECT MAX(x.id) FROM review_decisions x WHERE x.face_id = m.face_id) "
            "LEFT JOIN review_people p ON p.person_id = d.person_id "
            "WHERE m.run_id = ? AND m.pile_id = ? "
            "GROUP BY d.kind, d.person_id, p.display_name ORDER BY n DESC",
            (run_id, pile_id),
        )
    ]


def next_pile(conn: sqlite3.Connection, run_id: str, *, chunk: int = 200) -> dict[str, object] | None:
    """The best-ranked pile that still has an undecided face and has not been skipped.

    Walks the ranking in chunks rather than asking SQLite for "the first pile containing an
    undecided face", which makes it scan every pile's members before it can sort.
    """
    skipped = {
        int(r["pile_id"])
        for r in conn.execute("SELECT pile_id FROM review_skips WHERE run_id = ?", (run_id,))
    }
    offset = 0
    while True:
        batch = list_piles(conn, run_id, offset=offset, limit=chunk)
        if not batch:
            return None
        for rank, pile in enumerate(batch, start=offset + 1):
            pile_id = int(pile["pile_id"])
            if pile_id in skipped:
                continue
            if undecided_faces(conn, run_id, pile_id):
                pile["rank"] = rank
                return pile
        offset += chunk


def assign_pile(
    conn: sqlite3.Connection,
    run_id: str,
    pile_id: int,
    *,
    person_id: str | None = None,
    name: str | None = None,
    source: str = "pile",
) -> dict[str, object]:
    """Attribute every not-yet-decided face in a pile to one person."""
    resolved = resolve_person(conn, person_id=person_id, name=name)
    faces = undecided_faces(conn, run_id, pile_id)
    batch = _new_batch()
    now = _now()
    conn.executemany(
        "INSERT INTO review_decisions (face_id, kind, person_id, source, batch_id, decided_at) "
        "VALUES (?, 'person', ?, ?, ?, ?)",
        [(face_id, resolved, source, batch, now) for face_id in faces],
    )
    conn.commit()
    return {"batch_id": batch, "person_id": resolved, "n_faces": len(faces)}


def junk_pile(conn: sqlite3.Connection, run_id: str, pile_id: int) -> dict[str, object]:
    faces = undecided_faces(conn, run_id, pile_id)
    batch = _new_batch()
    now = _now()
    conn.executemany(
        "INSERT INTO review_decisions (face_id, kind, person_id, source, batch_id, decided_at) "
        "VALUES (?, 'junk', NULL, 'pile', ?, ?)",
        [(face_id, batch, now) for face_id in faces],
    )
    conn.commit()
    return {"batch_id": batch, "n_faces": len(faces)}


def skip_pile(conn: sqlite3.Connection, run_id: str, pile_id: int) -> dict[str, object]:
    """Pass over a pile without saying anything about its faces.

    Recorded in its own table: the labelling tool re-served the same pile forever because
    skipping was held only in the page, and a reload lost it.
    """
    conn.execute(
        "INSERT OR REPLACE INTO review_skips (run_id, pile_id, skipped_at) VALUES (?,?,?)",
        (run_id, pile_id, _now()),
    )
    conn.commit()
    return {"skipped": pile_id}


def undo_last(conn: sqlite3.Connection, run_id: str) -> dict[str, object]:
    """Reverse the most recent action: an assignment, a junking, a skip or a rejection.

    All four are timestamped, so "most recent" is decided by comparing them rather than by
    remembering which kind came last. The gold seed is excluded: it is not a human action,
    and peeling it off would quietly discard 1,499 attributions.
    """
    last_decision = conn.execute(
        "SELECT batch_id, decided_at FROM review_decisions WHERE batch_id <> 'seed' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    last_skip = conn.execute(
        "SELECT pile_id, skipped_at FROM review_skips WHERE run_id = ? "
        "ORDER BY skipped_at DESC, pile_id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    last_refusal = conn.execute(
        "SELECT batch_id, decided_at FROM review_not_person ORDER BY decided_at DESC, rowid DESC "
        "LIMIT 1"
    ).fetchone()

    options = [
        ("decision", last_decision["decided_at"] if last_decision else "", last_decision),
        ("skip", last_skip["skipped_at"] if last_skip else "", last_skip),
        ("rejection", last_refusal["decided_at"] if last_refusal else "", last_refusal),
    ]
    kind, when, row = max(options, key=lambda item: item[1])
    if not when or row is None:
        return {"undone": None}

    if kind == "skip":
        conn.execute(
            "DELETE FROM review_skips WHERE run_id = ? AND pile_id = ?",
            (run_id, int(row["pile_id"])),
        )
        conn.commit()
        return {"undone": "skip", "pile_id": int(row["pile_id"])}

    # One human action can touch both tables -- taking a face off a person records that it
    # is loose again AND that it is not them -- so a batch is always cleared from both,
    # whichever table led us to it.
    batch = str(row["batch_id"])
    removed = conn.execute("DELETE FROM review_decisions WHERE batch_id = ?", (batch,)).rowcount
    refused = conn.execute(
        "DELETE FROM review_not_person WHERE batch_id = ?", (batch,)
    ).rowcount
    if kind == "rejection":
        conn.commit()
        return {"undone": "rejection", "batch_id": batch, "n_faces": removed + refused}

    # A person created solely by the action just undone should not linger in the picker.
    conn.execute(
        "DELETE FROM review_people WHERE origin = 'review' AND person_id NOT IN "
        "(SELECT DISTINCT person_id FROM review_decisions WHERE person_id IS NOT NULL)"
    )
    conn.commit()
    return {"undone": "decision", "batch_id": batch, "n_faces": removed}


def people(conn: sqlite3.Connection) -> list[dict[str, object]]:
    """Everyone known, with how many faces they currently hold. Most faces first."""
    return [
        {
            "person_id": str(r["person_id"]),
            "display_name": r["display_name"],
            "origin": str(r["origin"]),
            "n_faces": int(r["n_faces"]),
            "sample": None if r["sample"] is None else int(r["sample"]),
        }
        for r in conn.execute(
            "SELECT p.person_id, p.display_name, p.origin, "
            "       COUNT(d.face_id) AS n_faces, MIN(d.face_id) AS sample "
            "FROM review_people p "
            "LEFT JOIN review_decisions d ON d.person_id = p.person_id AND d.kind = 'person' "
            "  AND d.id = (SELECT MAX(id) FROM review_decisions x WHERE x.face_id = d.face_id) "
            "GROUP BY p.person_id, p.display_name, p.origin "
            "ORDER BY n_faces DESC, p.person_id ASC"
        )
    ]


def progress(conn: sqlite3.Connection, run_id: str) -> dict[str, object]:
    total_faces = conn.execute(
        "SELECT COUNT(*) AS n FROM review_members WHERE run_id = ?", (run_id,)
    ).fetchone()["n"]
    decided = conn.execute(
        "SELECT COUNT(*) AS n FROM (SELECT face_id FROM review_decisions GROUP BY face_id "
        "HAVING (SELECT kind FROM review_decisions x WHERE x.face_id = review_decisions.face_id "
        "        ORDER BY x.id DESC LIMIT 1) <> 'unassigned')"
    ).fetchone()["n"]
    named = conn.execute(
        "SELECT COUNT(*) AS n FROM review_people WHERE display_name IS NOT NULL"
    ).fetchone()["n"]
    skipped = conn.execute(
        "SELECT COUNT(*) AS n FROM review_skips WHERE run_id = ?", (run_id,)
    ).fetchone()["n"]
    known = conn.execute("SELECT COUNT(*) AS n FROM review_people").fetchone()["n"]
    return {
        "faces_total": int(total_faces),
        "faces_decided": int(decided),
        "people_known": int(known),
        "people_named": int(named),
        "piles_skipped": int(skipped),
    }


# --------------------------------------------------------------------------------------
# Merge suggestions (v2)
#
# The measured reason this exists: recall 0.906 against precision 0.976, and 27 piles for 24
# people. The machine splits people far more often than it mixes them, so the highest-value
# human action is confirming that two groups are one person.
# --------------------------------------------------------------------------------------

# Below this, side-by-side pairs stop being worth a human's glance. Deliberately well under
# the 0.51 clustering cut: a suggestion only has to be worth *looking at*, and the pairs that
# matter are exactly the ones the clusterer was not confident enough to join itself.
MIN_SUGGESTION = 0.30


def _pile_centroids(conn: sqlite3.Connection, run_id: str) -> tuple[list[int], np.ndarray]:
    rows = conn.execute(
        "SELECT pile_id, centroid FROM review_piles WHERE run_id = ? AND centroid IS NOT NULL "
        "ORDER BY pile_id",
        (run_id,),
    ).fetchall()
    if not rows:
        return [], np.zeros((0, embed.EMBED_DIM), dtype=np.float32)
    ids = [int(r["pile_id"]) for r in rows]
    matrix = embed.load_matrix([bytes(r["centroid"]) for r in rows])
    return ids, embed.l2_normalise(matrix.astype(np.float32))


def person_centroids(
    conn: sqlite3.Connection, run_id: str
) -> tuple[list[str], np.ndarray]:
    """A direction in face space per person, built from the piles their faces sit in.

    Approximate on purpose: a person's vector is the size-weighted mean of the centroids of
    the piles they own faces in, not the mean of their own faces. That needs only the 8,820
    stored pile centroids (about 18 MB) instead of all 63,878 embeddings (about 131 MB), and
    the output ranks suggestions for a human to judge -- it is never a measurement.
    """
    pile_ids, pile_matrix = _pile_centroids(conn, run_id)
    if not pile_ids:
        return [], np.zeros((0, embed.EMBED_DIM), dtype=np.float32)
    index = {pile_id: position for position, pile_id in enumerate(pile_ids)}

    weights: dict[str, dict[int, int]] = {}
    for row in conn.execute(
        "SELECT d.person_id AS person_id, m.pile_id AS pile_id, COUNT(*) AS n "
        "FROM review_decisions d "
        "JOIN review_members m ON m.face_id = d.face_id AND m.run_id = ? "
        "WHERE d.kind = 'person' AND m.pile_id >= 0 AND d.id = "
        "  (SELECT MAX(x.id) FROM review_decisions x WHERE x.face_id = d.face_id) "
        "GROUP BY d.person_id, m.pile_id",
        (run_id,),
    ):
        weights.setdefault(str(row["person_id"]), {})[int(row["pile_id"])] = int(row["n"])

    people_ids: list[str] = []
    vectors: list[np.ndarray] = []
    for person_id, spread in sorted(weights.items()):
        usable = [(index[p], n) for p, n in spread.items() if p in index]
        if not usable:
            continue
        stacked = np.zeros(embed.EMBED_DIM, dtype=np.float32)
        for position, count in usable:
            stacked += pile_matrix[position] * float(count)
        people_ids.append(person_id)
        vectors.append(stacked)
    if not people_ids:
        return [], np.zeros((0, embed.EMBED_DIM), dtype=np.float32)
    return people_ids, embed.l2_normalise(np.stack(vectors))


def merge_candidates(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    limit: int = 1,
    min_similarity: float = MIN_SUGGESTION,
) -> list[dict[str, object]]:
    """The strongest unanswered "are these the same person?" pairs, best first.

    A pair is offered only when the pile still has undecided faces (there is something to
    gain), the person does not already own most of it, and nobody has said no to this
    pairing before.
    """
    people_ids, person_matrix = person_centroids(conn, run_id)
    pile_ids, pile_matrix = _pile_centroids(conn, run_id)
    if not people_ids or not pile_ids:
        return []

    open_piles = {
        int(r["pile_id"])
        for r in conn.execute(
            "SELECT DISTINCT m.pile_id FROM review_members m "
            "WHERE m.run_id = ? AND m.pile_id >= 0 AND "
            + _OPEN_FACE.format(face="m.face_id"),
            (run_id,),
        )
    }
    if not open_piles:
        return []

    refused: set[tuple[str, int]] = {
        (str(r["person_id"]), int(r["pile_id"]))
        for r in conn.execute(
            "SELECT DISTINCT n.person_id, m.pile_id FROM review_not_person n "
            "JOIN review_members m ON m.face_id = n.face_id AND m.run_id = ?",
            (run_id,),
        )
    }

    similarity = person_matrix @ pile_matrix.T
    pairs: list[tuple[float, str, int]] = []
    for row, person_id in enumerate(people_ids):
        for column, pile_id in enumerate(pile_ids):
            if pile_id not in open_piles or (person_id, pile_id) in refused:
                continue
            value = float(similarity[row, column])
            if value >= min_similarity:
                pairs.append((value, person_id, pile_id))

    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    names = {
        str(r["person_id"]): r["display_name"]
        for r in conn.execute("SELECT person_id, display_name FROM review_people")
    }

    out: list[dict[str, object]] = []
    for value, person_id, pile_id in pairs[: max(limit, 0)]:
        pile = conn.execute(
            "SELECT pile_id, n_faces, score, median_eye, coherence FROM review_piles "
            "WHERE run_id = ? AND pile_id = ?",
            (run_id, pile_id),
        ).fetchone()
        out.append(
            {
                "similarity": round(value, 4),
                "person_id": person_id,
                "display_name": names.get(person_id),
                "person_faces": person_face_sample(conn, person_id),
                "pile": dict(pile) if pile is not None else None,
                "pile_faces": pile_faces(conn, run_id, pile_id),
                "undecided": len(undecided_faces(conn, run_id, pile_id)),
            }
        )
    return out


def person_face_sample(
    conn: sqlite3.Connection, person_id: str, *, limit: int = 12
) -> list[dict[str, object]]:
    """The most recognisable faces this person currently holds, for a side-by-side."""
    return [
        dict(r)
        for r in conn.execute(
            "SELECT d.face_id, f.interocular_px, f.det_score FROM review_decisions d "
            "JOIN faces f ON f.id = d.face_id "
            "WHERE d.kind = 'person' AND d.person_id = ? AND d.id = "
            "  (SELECT MAX(x.id) FROM review_decisions x WHERE x.face_id = d.face_id) "
            "ORDER BY f.interocular_px DESC, f.det_score DESC LIMIT ?",
            (person_id, max(limit, 0)),
        )
    ]


def reject_merge(
    conn: sqlite3.Connection, run_id: str, person_id: str, pile_id: int
) -> dict[str, object]:
    """Record that this pile is not this person, so the pair is never offered again."""
    faces = [
        int(r["face_id"])
        for r in conn.execute(
            "SELECT face_id FROM review_members WHERE run_id = ? AND pile_id = ?",
            (run_id, pile_id),
        )
    ]
    batch = _new_batch()
    now = _now()
    conn.executemany(
        "INSERT OR REPLACE INTO review_not_person (face_id, person_id, batch_id, decided_at) "
        "VALUES (?,?,?,?)",
        [(face_id, person_id, batch, now) for face_id in faces],
    )
    conn.commit()
    return {"batch_id": batch, "person_id": person_id, "pile_id": pile_id, "n_faces": len(faces)}


# --------------------------------------------------------------------------------------
# Repairs, junk bucket and search (v3)
#
# The rarer half of the work. Precision is 0.976, so strangers in an album are uncommon --
# but when one appears it is the error that actually annoys, because it takes hunting rather
# than a click. These screens are deliberately not the default.
# --------------------------------------------------------------------------------------


def person_faces(
    conn: sqlite3.Connection, person_id: str, *, offset: int = 0, limit: int = 60
) -> dict[str, object]:
    """Every face currently attributed to one person, clearest first."""
    row = conn.execute(
        "SELECT person_id, display_name, origin FROM review_people WHERE person_id = ?",
        (person_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"No such person: {person_id}")

    total = conn.execute(
        "SELECT COUNT(*) AS n FROM review_decisions d WHERE d.kind = 'person' "
        "AND d.person_id = ? AND d.id = "
        "  (SELECT MAX(x.id) FROM review_decisions x WHERE x.face_id = d.face_id)",
        (person_id,),
    ).fetchone()["n"]

    faces = [
        dict(r)
        for r in conn.execute(
            "SELECT d.face_id, f.interocular_px, f.det_score, f.blur FROM review_decisions d "
            "JOIN faces f ON f.id = d.face_id "
            "WHERE d.kind = 'person' AND d.person_id = ? AND d.id = "
            "  (SELECT MAX(x.id) FROM review_decisions x WHERE x.face_id = d.face_id) "
            "ORDER BY f.interocular_px DESC, f.det_score DESC, d.face_id "
            "LIMIT ? OFFSET ?",
            (person_id, max(limit, 0), max(offset, 0)),
        )
    ]
    return {
        "person_id": str(row["person_id"]),
        "display_name": row["display_name"],
        "origin": str(row["origin"]),
        "n_faces": int(total),
        "offset": offset,
        "faces": faces,
    }


def _owned_by(conn: sqlite3.Connection, person_id: str, face_ids: list[int]) -> list[int]:
    """Of these faces, the ones this person currently holds.

    Guards both repairs against acting on somebody else's face. The screens only ever offer
    faces from the person on display, so this cannot be reached through the UI -- but the
    endpoint is reachable, and silently unassigning another person's face while recording
    "not this person" against the wrong name is the kind of damage nobody would notice.
    """
    wanted = [int(f) for f in face_ids]
    if not wanted:
        return []
    marks = ",".join("?" * len(wanted))
    return [
        int(r["face_id"])
        for r in conn.execute(
            f"SELECT d.face_id FROM review_decisions d WHERE d.face_id IN ({marks}) "
            "AND d.kind = 'person' AND d.person_id = ? AND d.id = "
            "  (SELECT MAX(x.id) FROM review_decisions x WHERE x.face_id = d.face_id)",
            (*wanted, person_id),
        )
    ]


def remove_faces(
    conn: sqlite3.Connection, person_id: str, face_ids: list[int]
) -> dict[str, object]:
    """Take faces off a person: they go back to the queue and are never re-suggested here."""
    wanted = _owned_by(conn, person_id, face_ids)
    if not wanted:
        return {"n_faces": 0, "batch_id": None}
    batch = _new_batch()
    now = _now()
    conn.executemany(
        "INSERT INTO review_decisions (face_id, kind, person_id, source, batch_id, decided_at) "
        "VALUES (?, 'unassigned', NULL, 'manual', ?, ?)",
        [(face_id, batch, now) for face_id in wanted],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO review_not_person (face_id, person_id, batch_id, decided_at) "
        "VALUES (?,?,?,?)",
        [(face_id, person_id, batch, now) for face_id in wanted],
    )
    conn.commit()
    return {"n_faces": len(wanted), "batch_id": batch, "person_id": person_id}


def split_person(
    conn: sqlite3.Connection, person_id: str, face_ids: list[int], name: str
) -> dict[str, object]:
    """Move some of a person's faces onto a different person: two people had been merged."""
    if not face_ids:
        raise ValueError("Choose at least one face to split off.")
    wanted = _owned_by(conn, person_id, face_ids)
    if not wanted:
        raise ValueError("None of those faces belong to that person.")
    target = resolve_person(conn, name=name)
    if target == person_id:
        raise ValueError("That is the same person; choose a different name.")
    batch = _new_batch()
    now = _now()
    conn.executemany(
        "INSERT INTO review_decisions (face_id, kind, person_id, source, batch_id, decided_at) "
        "VALUES (?, 'person', ?, 'manual', ?, ?)",
        [(face_id, target, batch, now) for face_id in wanted],
    )
    conn.commit()
    return {"n_faces": len(wanted), "batch_id": batch, "person_id": target}


def set_aside(conn: sqlite3.Connection, face_ids: list[int]) -> dict[str, object]:
    """Mark faces as junk from anywhere, not only a whole pile at a time."""
    wanted = [int(f) for f in face_ids]
    if not wanted:
        return {"n_faces": 0, "batch_id": None}
    batch = _new_batch()
    now = _now()
    conn.executemany(
        "INSERT INTO review_decisions (face_id, kind, person_id, source, batch_id, decided_at) "
        "VALUES (?, 'junk', NULL, 'manual', ?, ?)",
        [(face_id, batch, now) for face_id in wanted],
    )
    conn.commit()
    return {"n_faces": len(wanted), "batch_id": batch}


def restore_faces(conn: sqlite3.Connection, face_ids: list[int]) -> dict[str, object]:
    """Bring junked faces back into the queue."""
    wanted = [int(f) for f in face_ids]
    if not wanted:
        return {"n_faces": 0, "batch_id": None}
    batch = _new_batch()
    now = _now()
    conn.executemany(
        "INSERT INTO review_decisions (face_id, kind, person_id, source, batch_id, decided_at) "
        "VALUES (?, 'unassigned', NULL, 'manual', ?, ?)",
        [(face_id, batch, now) for face_id in wanted],
    )
    conn.commit()
    return {"n_faces": len(wanted), "batch_id": batch}


def bucket(
    conn: sqlite3.Connection, run_id: str, *, kind: str = "junk", offset: int = 0, limit: int = 60
) -> dict[str, object]:
    """What is not in anybody's album, and why.

    ``junk`` is what a human set aside. ``lone`` is what the clustering never matched to
    anything -- the honest measure of how much of the library the system could not handle.
    """
    if kind == "lone":
        where = (
            "m.run_id = ? AND m.pile_id = -1 AND " + _OPEN_FACE.format(face="m.face_id")
        )
        params: tuple[object, ...] = (run_id,)
    elif kind == "junk":
        where = (
            "m.run_id = ? AND (SELECT d.kind FROM review_decisions d WHERE d.face_id = m.face_id "
            " ORDER BY d.id DESC LIMIT 1) = 'junk'"
        )
        params = (run_id,)
    else:
        raise ValueError("kind must be 'junk' or 'lone'")

    total = conn.execute(
        f"SELECT COUNT(*) AS n FROM review_members m WHERE {where}", params
    ).fetchone()["n"]
    faces = [
        dict(r)
        for r in conn.execute(
            "SELECT m.face_id, f.interocular_px, f.det_score FROM review_members m "
            f"JOIN faces f ON f.id = m.face_id WHERE {where} "
            "ORDER BY f.interocular_px DESC, m.face_id LIMIT ? OFFSET ?",
            (*params, max(limit, 0), max(offset, 0)),
        )
    ]
    return {"kind": kind, "n_faces": int(total), "offset": offset, "faces": faces}


def search_people(
    conn: sqlite3.Connection, query: str, *, limit: int = 40
) -> list[dict[str, object]]:
    """Find a person by name or id. Needed the moment there are more than about thirty."""
    cleaned = (query or "").strip()
    everyone = people(conn)
    if not cleaned:
        return everyone[:limit]
    needle = cleaned.lower()
    matches = [
        person
        for person in everyone
        if needle in (str(person["display_name"] or "")).lower()
        or needle in str(person["person_id"]).lower()
    ]
    return matches[:limit]
