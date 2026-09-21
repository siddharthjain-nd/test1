"""Sampler tests.

The caps and the noise reservation are the load-bearing behaviours. A sampler that
silently lets one person dominate, or that never surfaces the noise bucket, produces a
gold set whose numbers look excellent and mean nothing -- and the error is undetectable
once labelling has started.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceindex import cluster, sampling


def make_candidate(
    face_id: int,
    *,
    cluster_id: int = 1,
    day: str = "2020-01-01",
    size: str = "medium",
    pose: str = "frontal",
    era: str = "middle",
    kind: str = "photo",
    filtered: str = "plain",
    group: str = "solo",
    quality: str = "good",
    det_score: float = 0.9,
) -> sampling.Candidate:
    return sampling.Candidate(
        face_id=face_id,
        cluster_id=cluster_id,
        det_score=det_score,
        day=day,
        strata={
            "size": size,
            "pose": pose,
            "era": era,
            "kind": kind,
            "filtered": filtered,
            "group": group,
            "quality": quality,
            "clustered": "noise" if cluster_id == -1 else "clustered",
        },
    )


def test_per_day_cap_trims_burst_frames() -> None:
    """20 near-identical frames of one person would contribute 190 trivially-easy pairs."""
    candidates = [make_candidate(i, cluster_id=7, day="2020-05-01") for i in range(20)]
    config = sampling.SampleConfig(per_cluster_per_day=3)
    assert len(sampling.apply_caps(candidates, config)) == 3


def test_per_day_cap_is_per_day_not_global() -> None:
    candidates = [
        make_candidate(i, cluster_id=7, day=f"2020-05-{day:02d}")
        for day in (1, 2, 3)
        for i in range(day * 100, day * 100 + 5)
    ]
    kept = sampling.apply_caps(candidates, sampling.SampleConfig(per_cluster_per_day=2))
    assert len(kept) == 6


def test_per_cluster_total_cap_limits_a_dominant_person() -> None:
    """One person with 500 faces contributes ~125,000 pairs and owns the metric."""
    candidates = [
        make_candidate(i, cluster_id=7, day=f"2020-{1 + i % 12:02d}-{1 + i % 28:02d}")
        for i in range(500)
    ]
    config = sampling.SampleConfig(per_cluster_per_day=3, per_cluster_total=70)
    assert len(sampling.apply_caps(candidates, config)) == 70


def test_noise_is_exempt_from_person_caps() -> None:
    """Noise is not a person. Capping it would discard the bucket PLAN.md demands reviewing."""
    candidates = [make_candidate(i, cluster_id=-1, day="2020-05-01") for i in range(50)]
    kept = sampling.apply_caps(candidates, sampling.SampleConfig(per_cluster_per_day=3))
    assert len(kept) == 50


def test_caps_are_deterministic() -> None:
    """Same input plus same config must give the same sample, or experiments stop comparing."""
    candidates = [make_candidate(i, cluster_id=7, day="2020-05-01") for i in range(40)]
    config = sampling.SampleConfig(per_cluster_per_day=5, random_seed=99)
    first = [c.face_id for c in sampling.apply_caps(candidates, config)]
    second = [c.face_id for c in sampling.apply_caps(candidates, config)]
    assert first == second


def test_select_reserves_noise_bucket() -> None:
    """Reviewing only confident clusters inherits the baseline's blind spots."""
    candidates = [make_candidate(i, cluster_id=i % 20) for i in range(400)]
    candidates += [make_candidate(1000 + i, cluster_id=-1) for i in range(200)]

    config = sampling.SampleConfig(target_size=100, noise_review_share=0.2, detector_fp_count=0)
    selected = sampling.select(candidates, config)

    assert sum(1 for c in selected if c.reserved_for == "noise_review") == 20


def test_select_reserves_lowest_confidence_detections() -> None:
    """These are the likely detector false positives -- the non_face labels PLAN.md wants."""
    candidates = [make_candidate(i, det_score=0.95) for i in range(100)]
    candidates += [make_candidate(500 + i, det_score=0.51) for i in range(10)]

    config = sampling.SampleConfig(target_size=50, detector_fp_count=10, noise_review_share=0.0)
    selected = sampling.select(candidates, config)

    reserved = [c for c in selected if c.reserved_for == "detector_fp"]
    assert len(reserved) == 10
    assert all(c.det_score == pytest.approx(0.51) for c in reserved)


def test_unclustered_faces_are_held_to_their_target_share() -> None:
    """Regression: the real corpus produced a 63% unclustered sample.

    Unclustered faces escape the per-person caps -- correct, they are not a person -- but
    that left them untrimmed while clustered faces were cut hard, so they dominated the pool
    the greedy fill drew from. They must be held to their stratum target instead: unmanaged
    they bury the labelling in face-by-face work, and excluded entirely the gold set contains
    only faces the baseline already groups successfully.
    """
    candidates = [make_candidate(i, cluster_id=i % 20) for i in range(600)]
    candidates += [make_candidate(5000 + i, cluster_id=-1) for i in range(4000)]

    config = sampling.SampleConfig(target_size=400, detector_fp_count=0)
    selected = sampling.select(candidates, config)

    share = sum(1 for c in selected if c.cluster_id == -1) / len(selected)
    target = sampling.DEFAULT_TARGETS["clustered"]["noise"]
    assert abs(share - target) <= 0.05, f"expected ~{target:.0%} unclustered, got {share:.0%}"


def test_pinned_faces_survive_a_resample() -> None:
    """Resampling after hours of labelling must not discard the work already done."""
    candidates = [make_candidate(i, cluster_id=i % 20) for i in range(400)]
    pinned = {3, 17, 42, 88}

    selected = sampling.select(candidates, sampling.SampleConfig(target_size=50), pinned=pinned)
    assert pinned <= {c.face_id for c in selected}


def test_pinned_faces_are_not_duplicated() -> None:
    candidates = [make_candidate(i, cluster_id=i % 10) for i in range(200)]
    candidates += [make_candidate(900 + i, cluster_id=-1) for i in range(40)]
    pinned = {1, 2, 3, 901, 902}

    selected = sampling.select(candidates, sampling.SampleConfig(target_size=60), pinned=pinned)
    ids = [c.face_id for c in selected]
    assert len(ids) == len(set(ids))


def test_select_never_exceeds_target_size() -> None:
    candidates = [make_candidate(i, cluster_id=i % 30) for i in range(500)]
    selected = sampling.select(candidates, sampling.SampleConfig(target_size=120))
    assert len(selected) <= 120


def test_select_returns_unique_faces() -> None:
    """Reserved buckets and greedy filling must not double-count a face."""
    candidates = [make_candidate(i, cluster_id=i % 10) for i in range(200)]
    candidates += [make_candidate(900 + i, cluster_id=-1) for i in range(60)]
    selected = sampling.select(candidates, sampling.SampleConfig(target_size=100))
    assert len({c.face_id for c in selected}) == len(selected)


def test_select_moves_marginals_toward_targets() -> None:
    """A pool skewed 90/10 should come out much closer to the 67/33 target."""
    candidates = [make_candidate(i, cluster_id=i % 40, kind="photo") for i in range(900)]
    candidates += [
        make_candidate(5000 + i, cluster_id=40 + i % 10, kind="forwarded") for i in range(300)
    ]

    config = sampling.SampleConfig(target_size=300, detector_fp_count=0, noise_review_share=0.0)
    selected = sampling.select(candidates, config)

    forwarded = sum(1 for c in selected if c.strata["kind"] == "forwarded") / len(selected)
    assert 0.25 <= forwarded <= 0.40


def test_select_is_deterministic() -> None:
    candidates = [make_candidate(i, cluster_id=i % 25) for i in range(400)]
    config = sampling.SampleConfig(target_size=80)
    first = [c.face_id for c in sampling.select(candidates, config)]
    second = [c.face_id for c in sampling.select(candidates, config)]
    assert first == second


def test_composition_report_flags_a_skewed_sample() -> None:
    """The report must fail loudly, not quietly describe a bad mix."""
    selected = [make_candidate(i, size="tiny") for i in range(100)]
    report = sampling.composition_report(selected, sampling.SampleConfig())
    assert not report.passed
    assert any("size/tiny" in failure for failure in report.failures)


def test_cross_era_shortfall_warns_but_does_not_block() -> None:
    """Measured on the real corpus: only 7 of 4,023 clusters span eras.

    Cross-age drift is exactly what stops one person's old and new photos from clustering
    together, so demanding spanning clusters asks the bootstrap to have already solved the
    problem the gold set exists to measure. It must not block sampling. The real check runs
    in export_gold_set.py against human labels.
    """
    selected = [make_candidate(i, cluster_id=i % 3) for i in range(100)]
    report = sampling.composition_report(selected, sampling.SampleConfig())

    assert any("span" in warning for warning in report.warnings)
    assert not any("span" in failure for failure in report.failures)


def test_strata_failures_still_block() -> None:
    """Demoting the cross-era check must not have softened the checks that do work."""
    selected = [make_candidate(i, size="tiny") for i in range(100)]
    report = sampling.composition_report(selected, sampling.SampleConfig())
    assert not report.passed


def test_mark_cross_era_detects_spanning_clusters() -> None:
    candidates = [
        make_candidate(1, cluster_id=5, era="oldest"),
        make_candidate(2, cluster_id=5, era="recent"),
        make_candidate(3, cluster_id=6, era="recent"),
    ]
    sampling._mark_cross_era(candidates)
    assert [c.cross_era for c in candidates] == [True, True, False]


def test_percentile_rank_spans_zero_to_one() -> None:
    ranks = sampling._percentile_rank(np.array([5.0, 1.0, 3.0]))
    assert ranks.min() == 0.0
    assert ranks.max() == 1.0


def test_bucket_assigns_edges_consistently() -> None:
    names = ("tiny", "small", "medium", "large")
    edges = (20.0, 40.0, 80.0)
    assert sampling._bucket(19.9, edges, names) == "tiny"
    assert sampling._bucket(20.0, edges, names) == "small"
    assert sampling._bucket(80.0, edges, names) == "large"


def test_bootstrap_cluster_groups_separated_points() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.01, size=(30, 8)) + np.array([1.0] + [0.0] * 7)
    b = rng.normal(0, 0.01, size=(30, 8)) + np.array([0.0, 1.0] + [0.0] * 6)
    from faceindex import embed as embed_module

    matrix = embed_module.l2_normalise(np.vstack([a, b]).astype(np.float32))
    result = cluster.bootstrap_cluster(matrix, cluster.ClusterConfig(min_cluster_size=5))
    assert result.n_clusters == 2


def test_bootstrap_cluster_handles_empty_input() -> None:
    result = cluster.bootstrap_cluster(
        np.zeros((0, 512), dtype=np.float32), cluster.ClusterConfig()
    )
    assert result.n_clusters == 0
    assert result.noise_fraction == 0.0
