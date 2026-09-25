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

    piles: list[tuple[int, int, float, float | None, float | None]] = []
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
        "INSERT INTO review_piles (run_id, pile_id, n_faces, score, median_eye, coherence) "
        "VALUES (?,?,?,?,?,?)",
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
