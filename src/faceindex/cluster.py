"""Bootstrap clustering -- pre-grouping for the labelling UI, and nothing more.

This is deliberately throwaway (Register C6). Its only job is to turn ~64k loose crops
into groups a human can confirm with one keystroke instead of sorting individually. The
gold set is *labels*, not embeddings, and every number in the project comes from
re-clustering later with properly tuned parameters.

So: do not tune this. Time spent optimising bootstrap quality is time not spent labelling,
and the labels are what actually matter.

Distance note
    Embeddings are L2-normalised, so squared Euclidean distance is ``2 - 2*cos``: a
    monotonic function of cosine distance. Ranking, and therefore the clustering, is
    identical, but Euclidean lets sklearn use a tree instead of materialising a dense
    64k x 64k distance matrix (32 GB) that would not fit on the 8 GB target machine.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from faceindex import embed


@dataclass(frozen=True)
class ClusterConfig:
    min_cluster_size: int = 3
    min_samples: int | None = None
    # PCA is OFF by default: measure before optimising (CONTEXT.md rule 1). Turn it on
    # only if the full-dimension run is measurably too slow on the target machine.
    pca_components: int | None = None
    random_seed: int = 20260906


@dataclass
class ClusterResult:
    labels: np.ndarray  # -1 means noise
    probabilities: np.ndarray
    n_clusters: int
    n_noise: int

    @property
    def noise_fraction(self) -> float:
        return float(self.n_noise) / len(self.labels) if len(self.labels) else 0.0


def bootstrap_cluster(embeddings: np.ndarray, config: ClusterConfig) -> ClusterResult:
    """Group embeddings with HDBSCAN. Returns per-row labels, ``-1`` for noise."""
    if len(embeddings) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return ClusterResult(empty, np.zeros(0, dtype=np.float64), 0, 0)

    features = embeddings.astype(np.float32)

    if config.pca_components:
        components = min(config.pca_components, features.shape[0], features.shape[1])
        features = PCA(n_components=components, random_state=config.random_seed).fit_transform(
            features
        )
        # PCA output is no longer unit-norm, so restore it to keep Euclidean monotonic
        # with cosine on the reduced space.
        features = embed.l2_normalise(features.astype(np.float32))

    model = HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = model.fit_predict(features)
    probabilities = getattr(model, "probabilities_", np.ones(len(labels)))

    return ClusterResult(
        labels=labels,
        probabilities=np.asarray(probabilities, dtype=np.float64),
        n_clusters=len({int(x) for x in labels} - {-1}),
        n_noise=int((labels == -1).sum()),
    )


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def load_embeddings(
    conn: sqlite3.Connection, *, limit: int | None = None
) -> tuple[list[int], np.ndarray]:
    """All stored embeddings at the current version, as ``(face_ids, matrix)``."""
    sql = "SELECT face_id, embedding FROM face_embeddings WHERE embed_version = ? ORDER BY face_id"
    params: list[object] = [embed.EMBED_VERSION]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    face_ids = [int(r["face_id"]) for r in rows]
    matrix = embed.load_matrix([bytes(r["embedding"]) for r in rows])
    return face_ids, matrix


def write_clusters(
    conn: sqlite3.Connection, face_ids: list[int], result: ClusterResult, run_id: str
) -> None:
    now = datetime.now(UTC).isoformat()
    conn.execute("DELETE FROM bootstrap_clusters")
    conn.executemany(
        "INSERT INTO bootstrap_clusters (face_id, cluster_id, probability, run_id, created_at) "
        "VALUES (?,?,?,?,?)",
        [
            (face_id, int(label), float(probability), run_id, now)
            for face_id, label, probability in zip(
                face_ids, result.labels, result.probabilities, strict=True
            )
        ],
    )
    conn.commit()


def cluster_sizes(conn: sqlite3.Connection) -> list[tuple[int, int]]:
    """``(cluster_id, size)`` ordered largest first. Noise (-1) is included."""
    return [
        (int(r["cluster_id"]), int(r["n"]))
        for r in conn.execute(
            "SELECT cluster_id, COUNT(*) AS n FROM bootstrap_clusters "
            "GROUP BY cluster_id ORDER BY n DESC"
        )
    ]
