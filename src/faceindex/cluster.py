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

import os
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

from faceindex import embed


def default_jobs() -> int:
    """Every core but one.

    The target machine has 8 GB and is used interactively while jobs run. Saturating it
    crashed the laptop once already, and wall-clock time is not the scarce resource here.
    """
    return max(1, (os.cpu_count() or 2) - 1)


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
    # Neighbours considered per face. Measured at 63,878 faces: k=10, 20 and 50 all peak at
    # ~570 MB, so this costs nothing and is left generous. More neighbours means more chances
    # to link a person's photos that sit far apart -- exactly the cross-age pairs that matter
    # most -- so lowering it would trade real quality for no saving.
    n_neighbors: int = 50
    # Graph methods: edges weaker than this are cut before clustering. Cosine similarity,
    # so 0.5 is permissive and 0.7 strict. This is the main quality dial for them.
    similarity_threshold: float = 0.5
    # Chinese Whispers rounds. It converges early and stops on its own.
    iterations: int = 20
    # Run agglomerative even when it is predicted not to fit. Swap absorbs the overflow;
    # expect thrashing rather than a crash.
    force_memory: bool = False
    # Leaves one core free by default.
    #
    # sklearn's HDBSCAN defaults this to None, meaning ONE core, which left three of four
    # idle for 70 minutes. But -1 is the opposite mistake on an 8 GB machine: every worker
    # carries its share of the data, and a full-throttle run alongside a browser was enough
    # to exhaust memory and take the laptop down. Time is cheap here; the machine staying
    # usable is not.
    n_jobs: int = field(default_factory=lambda: default_jobs())


@dataclass
class ClusterResult:
    labels: np.ndarray  # -1 means noise
    probabilities: np.ndarray
    n_clusters: int
    n_noise: int

    @property
    def noise_fraction(self) -> float:
        return float(self.n_noise) / len(self.labels) if len(self.labels) else 0.0


def estimate_agglomerative_gb(n_faces: int) -> float:
    """Measured at 63,878 faces: sklearn's AgglomerativeClustering peaks at 6.1 GB."""
    return 6.1 * (n_faces / 63_878)


def available_gb() -> float:
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")) / 1e9
    except (ValueError, OSError):
        return float("inf")


def _guard_memory(n_faces: int, *, force: bool = False) -> None:
    """Stop an agglomerative run that would exhaust the machine, unless overridden.

    Not a ban. Swap can absorb the overflow and the estimate is close enough to the limit
    that it may well fit with nothing else running -- but agglomerative walks its tree in
    random order, so anything that does spill thrashes rather than merely slowing down.
    Worth knowing before committing an evening to it.
    """
    estimate = estimate_agglomerative_gb(n_faces)
    free = available_gb()

    if force or estimate <= free * 0.8:
        return

    raise MemoryError(
        f"Agglomerative clustering of {n_faces:,} faces needs roughly {estimate:.1f} GB "
        f"and {free:.1f} GB is free (measured, not guessed).\n"
        f"  --algorithm chinese_whispers  does the same job in ~0.8 GB and 8x faster\n"
        f"  --force-memory                try anyway; close other applications first, and "
        f"expect swap thrashing if it overflows"
    )


def similarity_graph(features: np.ndarray, config: ClusterConfig) -> csr_matrix:
    """Sparse graph of each face to its nearest neighbours, weighted by cosine similarity.

    Measured at 63,878 faces: ~570 MB regardless of whether k is 10 or 50, because the graph
    is sparse and the cost is dominated by the embeddings themselves. Everything downstream
    of here works on this graph rather than on pairwise distances, which is what keeps the
    whole path inside 8 GB.
    """
    neighbours = min(config.n_neighbors, len(features) - 1)
    graph = kneighbors_graph(
        features,
        n_neighbors=neighbours,
        mode="distance",
        include_self=False,
        n_jobs=config.n_jobs,
    )
    # On L2-normalised vectors squared Euclidean is 2 - 2*cos, so cos = 1 - d^2/2.
    graph.data = 1.0 - (graph.data**2) / 2.0

    # Symmetrise: A being a neighbour of B must imply the reverse, or the result depends on
    # which of the two happened to be listed first.
    graph = graph.maximum(graph.T)

    # Drop weak links. Everything that survives is a pair the clusterer may join.
    graph.data[graph.data < config.similarity_threshold] = 0.0
    graph.eliminate_zeros()
    return graph.tocsr()


def _connected_components(features: np.ndarray, config: ClusterConfig) -> np.ndarray:
    """Join anything reachable through the graph. Cheapest possible, and chains badly.

    Equivalent to single-linkage at a fixed threshold: if A resembles B and B resembles C,
    all three land together even when A and C look nothing alike. On faces that can zip a
    whole family into one person. Included as the honest floor -- if something more careful
    cannot beat it, the extra machinery is not earning its place.
    """
    graph = similarity_graph(features, config)
    _, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels)
    return np.where(counts[labels] < 2, -1, labels).astype(np.int64)


def _chinese_whispers(features: np.ndarray, config: ClusterConfig) -> np.ndarray:
    """Label propagation over the neighbour graph. What dlib uses for face clustering.

    Every face starts as its own person, then repeatedly adopts whichever label carries the
    most total similarity among its neighbours. Strongly-connected groups reinforce
    themselves while a single weak link between two people gets outvoted -- so it resists the
    chaining that sinks connected components, without needing the 6 GB that agglomerative
    wants at this scale.

    Memory is the size of the graph, a few hundred MB, and it never materialises a distance
    matrix.
    """
    graph = similarity_graph(features, config)
    n = graph.shape[0]
    labels = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(config.random_seed)

    indptr, indices, data = graph.indptr, graph.indices, graph.data

    for _ in range(config.iterations):
        changed = 0
        # Asynchronous updates in random order: each face sees the labels its neighbours
        # already took this round, which is what lets groups consolidate quickly.
        for node in rng.permutation(n):
            start, end = indptr[node], indptr[node + 1]
            if start == end:
                continue

            neighbour_labels = labels[indices[start:end]]
            weights = data[start:end]

            unique, inverse = np.unique(neighbour_labels, return_inverse=True)
            totals = np.zeros(len(unique))
            np.add.at(totals, inverse, weights)

            winner = unique[int(np.argmax(totals))]
            if winner != labels[node]:
                labels[node] = winner
                changed += 1

        # Converged: nothing moved, so further rounds cannot change anything.
        if changed == 0:
            break

    counts = np.bincount(labels)
    return np.where(counts[labels] < 2, -1, labels).astype(np.int64)


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
        _guard_memory(len(features), force=config.force_memory)
        labels = _agglomerative(features, config)
        probabilities = np.ones(len(labels))
    elif config.algorithm == "chinese_whispers":
        labels = _chinese_whispers(features, config)
        probabilities = np.ones(len(labels))
    elif config.algorithm == "components":
        labels = _connected_components(features, config)
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


def available_models(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """``(model, n_faces)`` for every model with stored embeddings, most complete first."""
    return [
        (str(r["model"]), int(r["n"]))
        for r in conn.execute(
            "SELECT model, COUNT(*) AS n FROM face_embeddings WHERE embed_version = ? "
            "GROUP BY model ORDER BY n DESC",
            (embed.EMBED_VERSION,),
        )
    ]


def load_embeddings(
    conn: sqlite3.Connection, *, model: str | None = None, limit: int | None = None
) -> tuple[list[int], np.ndarray]:
    """Stored embeddings as ``(face_ids, matrix)``, for one model.

    Several models can now be stored side by side, so which one to score has to be stated.
    Defaults to whichever has the most faces, which is the one just finished.
    """
    if model is None:
        models = available_models(conn)
        if not models:
            return [], embed.load_matrix([])
        model = models[0][0]

    sql = (
        "SELECT face_id, embedding FROM face_embeddings "
        "WHERE embed_version = ? AND model = ? ORDER BY face_id"
    )
    params: list[object] = [embed.EMBED_VERSION, model]
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
