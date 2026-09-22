"""Metric tests.

These are the numbers every later decision rests on, so the edge cases matter more than the
happy path. A scorer that quietly rewards refusing to cluster, or that penalises the system
for correctly grouping two photographs of the same stranger, would send the whole project
chasing the wrong improvements.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from faceindex.eval import load_gold_set, score, score_by_slice


def test_perfect_clustering_scores_one() -> None:
    truth = {1: "a", 2: "a", 3: "b", 4: "b"}
    predicted = {1: 10, 2: 10, 3: 20, 4: 20}
    result = score(truth, predicted)
    assert result.pairwise_f1 == pytest.approx(1.0)
    assert result.bcubed_f1 == pytest.approx(1.0)
    assert result.ari == pytest.approx(1.0)


def test_cluster_ids_are_arbitrary() -> None:
    """Renaming clusters must change nothing -- only the grouping is meaningful."""
    truth = {1: "a", 2: "a", 3: "b", 4: "b"}
    first = score(truth, {1: 10, 2: 10, 3: 20, 4: 20})
    second = score(truth, {1: 99, 2: 99, 3: 7, 4: 7})
    assert first.bcubed_f1 == pytest.approx(second.bcubed_f1)


def test_everything_in_one_cluster_has_perfect_recall_poor_precision() -> None:
    truth = {1: "a", 2: "a", 3: "b", 4: "b"}
    result = score(truth, dict.fromkeys((1, 2, 3, 4), 1))
    assert result.pairwise_recall == pytest.approx(1.0)
    assert result.pairwise_precision < 0.5


def test_refusing_to_cluster_is_not_rewarded() -> None:
    """The critical one.

    If every ungrouped face kept the shared id -1, each pair of them would count as a
    successful grouping and a system that clusters nothing would score perfect recall.
    """
    truth = {1: "a", 2: "a", 3: "a", 4: "a"}
    result = score(truth, dict.fromkeys((1, 2, 3, 4), -1))
    assert result.pairwise_recall == pytest.approx(0.0)
    assert result.pairwise_f1 == pytest.approx(0.0)
    assert result.pct_noise == pytest.approx(1.0)


def test_noise_faces_are_not_grouped_with_each_other() -> None:
    truth = {1: "a", 2: "a", 3: "b"}
    result = score(truth, {1: -1, 2: -1, 3: -1})
    assert result.pairwise_precision == pytest.approx(0.0)
    assert result.n_clusters_pred == 0


def test_splitting_one_person_costs_recall_not_precision() -> None:
    truth = {1: "a", 2: "a", 3: "a", 4: "a"}
    result = score(truth, {1: 1, 2: 1, 3: 2, 4: 2})
    assert result.pairwise_precision == pytest.approx(1.0)
    assert result.pairwise_recall < 1.0


def test_merging_two_people_costs_precision_not_recall() -> None:
    truth = {1: "a", 2: "a", 3: "b", 4: "b"}
    result = score(truth, dict.fromkeys((1, 2, 3, 4), 1))
    assert result.pairwise_recall == pytest.approx(1.0)
    assert result.pairwise_precision < 1.0


def test_bcubed_is_less_dominated_by_one_large_person() -> None:
    """One person with many faces owns the pairwise metric; BCubed resists that.

    Pairwise weights a person by their *pairs*, which grows with the square of their photo
    count; BCubed weights every face equally. So 50 faces of 'a' contribute 1,225 pairs while
    4 faces of 'b' contribute 6 -- and completely failing 'b' costs pairwise recall almost
    nothing while BCubed registers it.

    This is exactly why PLAN.md tracks both and treats BCubed as the headline.
    """
    truth = {i: "a" for i in range(50)}
    truth.update({100 + i: "b" for i in range(4)})

    predicted = dict.fromkeys(range(50), 1)  # 'a' perfectly grouped
    predicted.update({100 + i: 50 + i for i in range(4)})  # 'b' shattered into singletons

    result = score(truth, predicted)

    assert result.pairwise_recall > 0.99, "pairwise barely notices the small person failing"
    assert result.bcubed_recall < 0.95, "BCubed registers it"
    assert result.bcubed_recall < result.pairwise_recall


def test_only_shared_faces_are_scored() -> None:
    """The pool is clustered whole; only labelled faces have ground truth."""
    truth = {1: "a", 2: "a"}
    predicted = {1: 5, 2: 5, 999: 7, 1000: 7}
    assert score(truth, predicted).n_scored == 2


def test_contamination_counts_junk_inside_person_clusters() -> None:
    truth = {1: "a", 2: "a"}
    predicted = {1: 5, 2: 5, 90: 5, 91: -1}
    result = score(truth, predicted, contaminants={90: "non_face", 91: "non_face"})
    assert result.contamination == pytest.approx(0.5)
    assert result.n_contaminants_checked == 2


def test_contamination_is_zero_without_contaminants() -> None:
    result = score({1: "a", 2: "a"}, {1: 5, 2: 5})
    assert result.contamination == pytest.approx(0.0)


def test_too_few_faces_returns_zeros_rather_than_raising() -> None:
    assert score({1: "a"}, {1: 5}).n_scored == 1


def test_slices_report_where_it_fails() -> None:
    truth = {i: "a" for i in range(20)}
    truth.update({100 + i: "b" for i in range(20)})
    # 'a' clustered perfectly; 'b' scattered into singletons.
    predicted = dict.fromkeys(range(20), 1)
    predicted.update({100 + i: 50 + i for i in range(20)})
    slices = {i: "large" for i in range(20)}
    slices.update({100 + i: "tiny" for i in range(20)})

    results = {s.name: s for s in score_by_slice(truth, predicted, slices)}
    assert results["large"].f1 > results["tiny"].f1


def test_small_slices_are_skipped() -> None:
    truth = {1: "a", 2: "a", 3: "a"}
    predicted = {1: 1, 2: 1, 3: 1}
    slices = {1: "big", 2: "big", 3: "rare"}
    assert [s.name for s in score_by_slice(truth, predicted, slices, min_faces=2)] == ["big"]


def _write_gold(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "face_id",
        "label",
        "person_id",
        "taken_at",
        "kind",
        "size_bucket",
        "pose_bucket",
        "era_bucket",
        "quality_bucket",
        "filtered_bucket",
        "group_bucket",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def test_gold_set_separates_the_three_groups(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    _write_gold(
        path,
        [
            {"face_id": "1", "label": "person", "person_id": "mom", "size_bucket": "large"},
            {"face_id": "2", "label": "person", "person_id": "mom", "size_bucket": "tiny"},
            {"face_id": "3", "label": "not_of_interest"},
            {"face_id": "4", "label": "non_face"},
            {"face_id": "5", "label": "unsure"},
        ],
    )
    gold = load_gold_set(path)

    assert gold.identities == {1: "mom", 2: "mom"}
    assert gold.contaminants == {3: "not_of_interest", 4: "non_face"}
    assert gold.excluded == {5: "unsure"}
    assert gold.n_people == 1


def test_strangers_never_enter_identity_scoring(tmp_path: Path) -> None:
    """A stranger grouped with another photo of themselves is not an error.

    Treating not_of_interest as one shared class would mark exactly that as a mistake.
    """
    path = tmp_path / "labels.csv"
    _write_gold(
        path,
        [
            {"face_id": "1", "label": "person", "person_id": "mom"},
            {"face_id": "2", "label": "person", "person_id": "mom"},
            {"face_id": "3", "label": "not_of_interest"},
            {"face_id": "4", "label": "not_of_interest"},
        ],
    )
    gold = load_gold_set(path)
    result = score(gold.identities, {1: 1, 2: 1, 3: 9, 4: 9}, contaminants=gold.contaminants)

    assert result.n_scored == 2
    assert result.bcubed_f1 == pytest.approx(1.0)


def test_slice_values_cover_only_identity_faces(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    _write_gold(
        path,
        [
            {"face_id": "1", "label": "person", "person_id": "mom", "size_bucket": "large"},
            {"face_id": "3", "label": "non_face", "size_bucket": "tiny"},
        ],
    )
    gold = load_gold_set(path)
    assert gold.slice_values("size_bucket") == {1: "large"}


def test_missing_gold_set_says_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="export_gold_set"):
        load_gold_set(tmp_path / "nope.csv")


def test_split_is_deterministic() -> None:
    """Same gold set in, same split out -- or results from different days stop comparing."""
    from faceindex.eval.split import make_split

    identities = {i: f"p{i % 40}" for i in range(400)}
    first = make_split(identities).assignment
    second = make_split(identities).assignment
    assert first == second


def test_split_holds_back_roughly_a_fifth() -> None:
    from faceindex.eval.split import HOLDOUT, make_split

    identities = {i: f"p{i % 100}" for i in range(1000)}
    split = make_split(identities)
    assert 15 <= len(split.people(HOLDOUT)) <= 25


def test_split_never_puts_one_person_on_both_sides() -> None:
    """Splitting by face instead of identity would leak the answer into the tuning set."""
    from faceindex.eval.split import HOLDOUT, TUNE, make_split

    identities = {i: f"p{i % 30}" for i in range(300)}
    split = make_split(identities)
    assert not (split.people(TUNE) & split.people(HOLDOUT))


def test_split_balances_large_and_small_people() -> None:
    """A random split can hand every frequently-photographed person to one side."""
    from faceindex.eval.split import HOLDOUT, TUNE, make_split

    identities: dict[int, str] = {}
    face = 0
    for person in range(40):
        for _ in range(1 + person):  # sizes 1..40
            identities[face] = f"p{person}"
            face += 1

    split = make_split(identities)
    counts: dict[str, int] = {}
    for person in identities.values():
        counts[person] = counts.get(person, 0) + 1

    biggest_tune = max(counts[p] for p in split.people(TUNE))
    biggest_holdout = max(counts[p] for p in split.people(HOLDOUT))
    assert biggest_holdout > 0.5 * biggest_tune, "holdout got only small people"


def test_filter_identities_keeps_only_one_side() -> None:
    from faceindex.eval.split import HOLDOUT, TUNE, Split

    split = Split(assignment={"a": TUNE, "b": HOLDOUT})
    identities = {1: "a", 2: "a", 3: "b"}
    assert split.filter_identities(identities, TUNE) == {1: "a", 2: "a"}
    assert split.filter_identities(identities, HOLDOUT) == {3: "b"}
    assert split.filter_identities(identities, "all") == identities


def test_missing_split_says_what_to_run(tmp_path: Path) -> None:
    from faceindex.eval.split import load_split

    with pytest.raises(FileNotFoundError, match="make_holdout"):
        load_split(tmp_path / "nope.csv")
