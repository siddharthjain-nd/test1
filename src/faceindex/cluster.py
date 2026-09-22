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
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

from faceindex import embed


@dataclass(frozen=True)
class ClusterConfig:
    min_cluster_size: int = 3
    min_samples: int | None = None
    # PCA is OFF by default: measure before optimising (CONTEXT.md rule 1). Turn it on
    # only if the full-dimension run is measurably too slow on the target machine.
    pca_components: int | None = None
    random_seed: int = 20260906

    # Which algorithm. HDBSCAN groups by *density*, which is why it shatters a person into
    # minimum-size fragments and leaves anyone with few photos unclustered. Agglomerative
    # merges by distance instead: it has no density requirement and the threshold controls
    # merging directly, which is the lever the measured failure actually calls for.
    algorithm: str = "hdbscan"

    # HDBSCAN only. Merges clusters closer together than this, directly countering the
    # over-splitting the baseline showed. 0.0 disables it.
    selection_epsilon: float = 0.0

    # Agglomerative only. Distance below which two groups merge, on L2-normalised vectors
    # where squared Euclidean is 2 - 2*cos. 0.8 corresponds to cosine similarity 0.6.
    distance_threshold: float = 0.8
    linkage: str = "average"
    n_neighbors: int = 50
    # sklearn's HDBSCAN defaults n_jobs to None, which means ONE core -- so a 70-minute run
    # left three of four cores idle. Only the neighbour search parallelises; the tree and
    # hierarchy construction stay sequential, so expect a useful speedup rather than 4x.
    n_jobs: int = -1


@dataclass
class ClusterResult:
    labels: np.ndarray  # -1 means noise
    probabilities: np.ndarray
    n_clusters: int
    n_noise: int

    @property
    def noise_fraction(self) -> float:
        return float(self.n_noise) / len(self.labels) if len(self.labels) else 0.0


def _agglomerative(features: np.ndarray, config: ClusterConfig) -> np.ndarray:
    """Merge groups until nothing is closer than the threshold. No density requirement.

    Two properties matter here, and both address what the baseline measured. Any two faces
    close enough become a group, so a person with two photographs is no longer discarded --
    that alone should reclaim a large part of the 41% left unclustered. And the threshold
    controls merging directly, rather than emerging from a density estimate, so
    over-splitting becomes a number to tune rather than a property to live with.

    The naive form needs the full 64k x 64k distance matrix, which is 32 GB. A nearest
    neighbour graph is passed instead, so only nearby pairs are ever considered; merges are
    then restricted to faces that are someone's neighbour, which is exactly the intent.
    """
    graph = kneighbors_graph(
        features,
        n_neighbors=min(config.n_neighbors, len(features) - 1),
        mode="connectivity",
        include_self=False,
        n_jobs=config.n_jobs,
    )
    # Symmetrise: A being a neighbour of B must imply the reverse, or the merge order
    # depends on which of the two happened to be listed first.
    graph = ((graph + graph.T) > 0).astype(np.int8)

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=config.distance_threshold,
        metric="euclidean",
        linkage=config.linkage,
        connectivity=graph,
    )
    labels: np.ndarray = model.fit_predict(features)

    # Singletons are reported as ungrouped, matching HDBSCAN's convention so the two
    # algorithms produce comparable numbers.
    counts = np.bincount(labels)
    return np.where(counts[labels] < 2, -1, labels)


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

    if config.algorithm == "agglomerative":
        labels = _agglomerative(features, config)
        probabilities = np.ones(len(labels))
    else:
        model = HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            cluster_selection_epsilon=config.selection_epsilon,
            n_jobs=config.n_jobs,
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
