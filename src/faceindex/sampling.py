"""Stratified sampler for the gold set.

Picks ~1,800 faces out of ~64k such that the resulting evaluation set exercises the
failure modes that actually break clustering, rather than the easy frontal portraits a
naive sample would be dominated by.

Three ideas do the real work:

**Caps before quotas.** One person with 500 faces contributes ~125,000 pairs and owns the
pairwise metric outright; 20 burst frames of one person contribute 190 trivially-easy
pairs. So the pool is trimmed by per-person-per-day and per-person caps *before* any
quota filling. Identity is unknown at this stage, so the bootstrap cluster stands in for
it -- which is exactly why clustering runs before sampling.

**Greedy marginal matching.** Six categorical dimensions cannot be hit independently by
simple proportional selection. Each pick instead goes to whichever candidate most reduces
the largest outstanding deficit across all dimensions at once.

**Reserved buckets.** Faces the baseline already finds easy are the ones a confident
sampler picks. A fixed share is therefore reserved for the bootstrap *noise* bucket and
for the lowest-confidence detections, because reviewing only clean clusters produces a
gold set that inherits the baseline's blind spots and can never measure them.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

# Dimension -> {category: target share}. Defaults are PLAN.md section 4's Phase 1 table.
# Deviating from the plan of record should be a logged decision, not a silent code change.
DEFAULT_TARGETS: dict[str, dict[str, float]] = {
    # tiny raised 0.10 -> 0.15 on 2026-09-19 (decision 29). The contact sheet was inspected
    # and the sub-20px faces are overwhelmingly *genuine* faces -- background people in group
    # shots, and faces inside framed photographs -- not the detector false positives the
    # original target assumed. They are still unidentifiable and will be gated out of cluster
    # formation in Phase 3, which is precisely why the gold set must contain enough of them to
    # tune that gate against. The remaining three are scaled down proportionally.
    "size": {"tiny": 0.15, "small": 0.24, "medium": 0.38, "large": 0.23},
    "pose": {"frontal": 0.55, "semi": 0.30, "profile": 0.15},
    "era": {"oldest": 0.35, "middle": 0.25, "recent": 0.35, "undated": 0.05},
    "kind": {"photo": 0.67, "forwarded": 0.33},
    "filtered": {"plain": 0.88, "beauty": 0.12},
    "group": {"solo": 0.85, "group": 0.15},
    "quality": {"good": 0.60, "marginal": 0.25, "bad": 0.15},
}

DIMENSIONS = tuple(DEFAULT_TARGETS)


@dataclass(frozen=True)
class SampleConfig:
    target_size: int = 1800
    per_cluster_per_day: int = 3
    per_cluster_total: int = 70
    # PLAN.md: "The noise/rejected bucket must be reviewed too."
    noise_review_share: float = 0.10
    # PLAN.md target composition: ~50 detector false positives.
    detector_fp_count: int = 50
    # Faces whose bootstrap cluster spans the oldest and most recent eras are the highest
    # value faces in the set -- cross-age drift is what actually breaks clustering.
    cross_era_bonus: float = 0.5
    targets: dict[str, dict[str, float]] = field(default_factory=lambda: DEFAULT_TARGETS)
    random_seed: int = 20260906


@dataclass
class Candidate:
    face_id: int
    cluster_id: int
    det_score: float
    day: str
    strata: dict[str, str]
    cross_era: bool = False
    reserved_for: str | None = None


def _bucket(value: float | None, edges: tuple[float, ...], names: tuple[str, ...]) -> str:
    if value is None:
        return names[0]
    for edge, name in zip(edges, names[:-1], strict=True):
        if value < edge:
            return name
    return names[-1]


def load_candidates(conn: sqlite3.Connection, *, beauty_marker: str) -> list[Candidate]:
    """Every pooled face, annotated with the categorical cell it occupies.

    ``beauty_marker`` is a path substring identifying the retouching-app folder. Those
    filters alter face *geometry*, which is what the embedding encodes, so the slice is
    tracked rather than assumed harmless.
    """
    rows = conn.execute(
        """
        SELECT
            f.id, f.interocular_px, f.yaw_deg, f.blur, f.det_score,
            f.dark_fraction, f.bright_fraction,
            p.kind, p.path, p.taken_at,
            COALESCE(b.cluster_id, -1) AS cluster_id,
            (SELECT COUNT(*) FROM faces g WHERE g.photo_id = f.photo_id) AS n_faces
        FROM faces f
        JOIN photos p ON p.id = f.photo_id
        LEFT JOIN bootstrap_clusters b ON b.face_id = f.id
        JOIN face_embeddings e ON e.face_id = f.id
        ORDER BY f.id
        """
    ).fetchall()
    if not rows:
        return []

    years = sorted(int(r["taken_at"][:4]) for r in rows if r["taken_at"])
    if years:
        newest = years[-1]
        recent_from = newest - 1  # the most recent ~2 calendar years
        oldest_to = years[max(0, len(years) // 3 - 1)]  # oldest third of the dated population
    else:
        recent_from, oldest_to = 9999, 0

    # Quality is defined by rank within this corpus, not by absolute thresholds. Absolute
    # numbers would not transfer between a 2006 feature phone and a DSLR, and this library
    # contains both.
    blur_rank = _percentile_rank(np.array([float(r["blur"] or 0.0) for r in rows]))
    score_rank = _percentile_rank(np.array([float(r["det_score"]) for r in rows]))
    exposure = np.array(
        [float(r["dark_fraction"] or 0.0) + float(r["bright_fraction"] or 0.0) for r in rows]
    )
    composite = 0.45 * blur_rank + 0.45 * score_rank + 0.10 * (1.0 - np.clip(exposure, 0, 1))

    candidates: list[Candidate] = []
    for row, quality_score in zip(rows, composite, strict=True):
        year = int(row["taken_at"][:4]) if row["taken_at"] else None
        if year is None:
            era = "undated"
        elif year >= recent_from:
            era = "recent"
        elif year <= oldest_to:
            era = "oldest"
        else:
            era = "middle"

        strata = {
            "size": _bucket(
                row["interocular_px"], (20.0, 40.0, 80.0), ("tiny", "small", "medium", "large")
            ),
            "pose": _bucket(
                abs(float(row["yaw_deg"] or 0.0)), (15.0, 45.0), ("frontal", "semi", "profile")
            ),
            "era": era,
            "kind": "forwarded" if row["kind"] == "forwarded" else "photo",
            "filtered": "beauty" if beauty_marker and beauty_marker in row["path"] else "plain",
            "group": "group" if int(row["n_faces"]) >= 4 else "solo",
            "quality": _bucket(float(quality_score), (0.15, 0.40), ("bad", "marginal", "good")),
        }

        candidates.append(
            Candidate(
                face_id=int(row["id"]),
                cluster_id=int(row["cluster_id"]),
                det_score=float(row["det_score"]),
                day=str(row["taken_at"])[:10] if row["taken_at"] else "undated",
                strata=strata,
            )
        )

    _mark_cross_era(candidates)
    return candidates


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Map values to ``[0, 1]`` by rank, so thresholds are corpus-relative."""
    if len(values) == 0:
        return values
    order = values.argsort()
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(1, len(values) - 1)


def _mark_cross_era(candidates: list[Candidate]) -> None:
    """Flag faces whose bootstrap cluster appears in both the oldest and most recent eras."""
    eras: dict[int, set[str]] = defaultdict(set)
    for candidate in candidates:
        if candidate.cluster_id != -1:
            eras[candidate.cluster_id].add(candidate.strata["era"])

    spanning = {cid for cid, seen in eras.items() if {"oldest", "recent"} <= seen}
    for candidate in candidates:
        candidate.cross_era = candidate.cluster_id in spanning


def apply_caps(candidates: list[Candidate], config: SampleConfig) -> list[Candidate]:
    """Trim burst frames and dominant people before any quota filling.

    Pre-trimming rather than constraining the greedy loop makes the caps a guarantee
    instead of a best effort, and keeps the selection deterministic.
    """
    rng = np.random.default_rng(config.random_seed)
    by_cluster: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_cluster[candidate.cluster_id].append(candidate)

    kept: list[Candidate] = []
    for cluster_id, members in sorted(by_cluster.items()):
        # Noise is not a person; per-person caps are meaningless there and would discard
        # most of the bucket the plan explicitly requires reviewing.
        if cluster_id == -1:
            kept.extend(members)
            continue

        per_day: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in members:
            per_day[candidate.day].append(candidate)

        survivors: list[Candidate] = []
        for day in sorted(per_day):
            group = per_day[day]
            if len(group) > config.per_cluster_per_day:
                picks = rng.choice(len(group), config.per_cluster_per_day, replace=False)
                group = [group[i] for i in sorted(picks)]
            survivors.extend(group)

        if len(survivors) > config.per_cluster_total:
            picks = rng.choice(len(survivors), config.per_cluster_total, replace=False)
            survivors = [survivors[i] for i in sorted(picks)]

        kept.extend(survivors)

    return sorted(kept, key=lambda c: c.face_id)


def _trim(chosen: list[Candidate], total: int, keep: set[int]) -> list[Candidate]:
    """Cut the selection to size without ever discarding an already-labelled face.

    A plain ``[:total]`` sorts by face id and truncates, which silently drops pinned faces
    that happen to sort late -- throwing away finished labelling to satisfy a target count.
    If the pinned set alone exceeds the target, the target loses.
    """
    if len(chosen) <= total:
        return sorted(chosen, key=lambda c: c.face_id)

    pinned_items = [c for c in chosen if c.face_id in keep]
    others = [c for c in chosen if c.face_id not in keep]
    room = max(0, total - len(pinned_items))
    return sorted(pinned_items + others[:room], key=lambda c: c.face_id)


def select(
    candidates: list[Candidate],
    config: SampleConfig,
    *,
    pinned: set[int] | None = None,
) -> list[Candidate]:
    """Greedy marginal matching against the target shares. Deterministic.

    ``pinned`` faces are kept unconditionally. Resampling after hours of labelling must not
    discard that work, so already-judged faces are carried into the new sample and the
    quotas are filled around them.
    """
    if not candidates:
        return []

    total = min(config.target_size, len(candidates))
    keep = pinned or set()

    chosen: list[Candidate] = [c for c in candidates if c.face_id in keep]
    chosen_ids = {c.face_id for c in chosen}

    for candidate in _reserve(candidates, config, total):
        if candidate.face_id not in chosen_ids:
            chosen.append(candidate)
            chosen_ids.add(candidate.face_id)

    # Noise enters ONLY through the reserve, never through the greedy fill.
    #
    # Noise is exempt from the per-person caps -- correct, since it is not a person -- but
    # that leaves it untrimmed while the clustered faces are cut hard. On the real corpus
    # noise went from 41% of the pool to 63% of what remained after capping, and a greedy
    # fill draws from that proportionally: `noise_review_share` said 10% and the sample came
    # out 63% noise. That is not just slow to label. Unclustered faces are overwhelmingly
    # strangers and unreadable crops, so the set starves of the person labels it exists to
    # provide, and the shortfall only surfaces at export, after all the work is done.
    pool = [c for c in candidates if c.face_id not in chosen_ids and c.cluster_id != -1]
    if not pool or len(chosen) >= total:
        return _trim(chosen, total, keep)

    codes = {
        dim: np.array([list(config.targets[dim]).index(c.strata[dim]) for c in pool])
        for dim in config.targets
    }
    bonus = np.array([config.cross_era_bonus if c.cross_era else 0.0 for c in pool])

    counts = {
        dim: np.array([sum(1 for c in chosen if c.strata[dim] == cat) for cat in cats], float)
        for dim, cats in config.targets.items()
    }
    quotas = {
        dim: np.array([share * total for share in cats.values()])
        for dim, cats in config.targets.items()
    }

    available = np.ones(len(pool), dtype=bool)
    for _ in range(total - len(chosen)):
        if not available.any():
            break

        score = bonus.copy()
        for dim in config.targets:
            deficit = np.maximum(0.0, quotas[dim] - counts[dim]) / max(1.0, total)
            score = score + deficit[codes[dim]]

        score[~available] = -np.inf
        best = int(np.argmax(score))

        available[best] = False
        picked = pool[best]
        chosen.append(picked)
        for dim in config.targets:
            counts[dim][codes[dim][best]] += 1

    return _trim(chosen, total, keep)


def _reserve(candidates: list[Candidate], config: SampleConfig, total: int) -> list[Candidate]:
    """Fixed allocations for the two buckets a confident sampler would never pick."""
    rng = np.random.default_rng(config.random_seed)
    reserved: list[Candidate] = []
    taken: set[int] = set()

    # Lowest-confidence detections: the most likely detector false positives.
    by_score = sorted(candidates, key=lambda c: (c.det_score, c.face_id))
    for candidate in by_score[: config.detector_fp_count]:
        candidate.reserved_for = "detector_fp"
        reserved.append(candidate)
        taken.add(candidate.face_id)

    noise = [c for c in candidates if c.cluster_id == -1 and c.face_id not in taken]
    want = int(total * config.noise_review_share)
    if noise and want:
        picks = rng.choice(len(noise), min(want, len(noise)), replace=False)
        for index in sorted(picks):
            noise[index].reserved_for = "noise_review"
            reserved.append(noise[index])

    return reserved


# --------------------------------------------------------------------------------------
# Composition report -- the sampler asserts its own output rather than assuming it
# --------------------------------------------------------------------------------------


@dataclass
class DimensionReport:
    dimension: str
    rows: list[tuple[str, int, float, float]]  # category, count, actual, target
    max_deviation: float


@dataclass
class CompositionReport:
    n_selected: int
    dimensions: list[DimensionReport]
    n_clusters: int
    n_cross_era_clusters: int
    n_noise_review: int
    n_detector_fp: int
    failures: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def composition_report(
    selected: list[Candidate], config: SampleConfig, *, tolerance: float = 0.05
) -> CompositionReport:
    """Verify the sample against the targets. A sampler that only claims a mix is useless."""
    total = len(selected)
    dimensions: list[DimensionReport] = []
    failures: list[str] = []

    for dim, targets in config.targets.items():
        rows: list[tuple[str, int, float, float]] = []
        worst = 0.0
        for category, target in targets.items():
            count = sum(1 for c in selected if c.strata[dim] == category)
            actual = count / total if total else 0.0
            rows.append((category, count, actual, target))
            deviation = abs(actual - target)
            worst = max(worst, deviation)
            if deviation > tolerance:
                failures.append(
                    f"{dim}/{category}: {actual:.1%} vs target {target:.1%} "
                    f"(off by {deviation:.1%}, tolerance {tolerance:.0%})"
                )
        dimensions.append(DimensionReport(dim, rows, worst))

    clusters = {c.cluster_id for c in selected if c.cluster_id != -1}
    cross_era = {c.cluster_id for c in selected if c.cross_era and c.cluster_id != -1}

    # A warning, deliberately not a failure.
    #
    # This counts bootstrap clusters holding faces from both the oldest and newest eras --
    # but cross-age drift is precisely what stops one person's old and new photos from
    # landing in the same cluster. Requiring spanning clusters therefore asks the bootstrap
    # to have already solved the problem the gold set is being built to detect, and the
    # measurement bears that out: on the real corpus only 7 of 4,023 clusters span eras,
    # while nearest-neighbour search finds hundreds of genuine cross-era faces.
    #
    # The real check lives in export_gold_set.py, where human labels decide whether a child
    # and an adult are the same person. That is the only thing that can decide it.
    warnings: list[str] = []
    if len(cross_era) < 10:
        warnings.append(
            f"only {len(cross_era)} sampled bootstrap clusters span the oldest and newest eras. "
            f"Expected: cross-age drift splits a person across clusters, which is the failure "
            f"this project exists to measure. Cross-era identities are recovered during "
            f"labelling, by assigning one person_id across several clusters, and verified by "
            f"export_gold_set.py against the >=10 requirement."
        )

    return CompositionReport(
        n_selected=total,
        dimensions=dimensions,
        n_clusters=len(clusters),
        n_cross_era_clusters=len(cross_era),
        n_noise_review=sum(1 for c in selected if c.reserved_for == "noise_review"),
        n_detector_fp=sum(1 for c in selected if c.reserved_for == "detector_fp"),
        failures=failures,
        warnings=warnings,
    )


def write_candidates(conn: sqlite3.Connection, selected: list[Candidate], run_id: str) -> None:
    now = datetime.now(UTC).isoformat()
    conn.execute("DELETE FROM gold_candidates")
    conn.executemany(
        "INSERT INTO gold_candidates "
        "(face_id, stratum, bootstrap_cluster, reserved_for, sample_run, created_at) "
        "VALUES (?,?,?,?,?,?)",
        [
            (
                c.face_id,
                json.dumps(c.strata, sort_keys=True),
                c.cluster_id,
                c.reserved_for,
                run_id,
                now,
            )
            for c in selected
        ],
    )
    conn.commit()
