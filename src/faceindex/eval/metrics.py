"""Scoring a predicted clustering against human labels.

Four label kinds come out of the gold set and they are *not* interchangeable, so each gets
its own treatment. Getting this wrong is the easiest way to produce a confident number that
measures the wrong thing.

``person_N``
    The only faces scored for identity. Everything below is excluded from the grouping
    metrics.

``not_of_interest``
    A real face, but a stranger. **Excluded.** The system is explicitly permitted to call
    them noise *or* to group them -- neither is an error. Scoring them as one giant "class
    of strangers" would mark the system wrong for correctly noticing that two photographs
    of the same passer-by are the same person.

``non_face``
    No face present. Excluded from identity scoring, but counted separately: junk landing
    inside a person's album is a real product failure and needs its own number.

``unsure``
    Excluded entirely, as promised to whoever pressed the key. Never guessed, never scored.

Noise in the *prediction* is treated as "no cluster", not as a cluster of its own. Two faces
the system declined to group are not grouped, so they must never count as a correct pairing.
Each is expanded into its own singleton before any metric is computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

NOISE = -1

# Labels that carry a real identity. Only these are scored for grouping quality.
IDENTITY_LABEL = "person"


@dataclass
class ClusteringMetrics:
    """One row of the results table."""

    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1: float

    bcubed_precision: float
    bcubed_recall: float
    bcubed_f1: float

    nmi: float
    ari: float

    n_clusters_pred: int
    n_clusters_true: int
    n_scored: int
    pct_noise: float

    # Junk reaching a person's album: the share of non_face / not_of_interest faces the
    # system placed inside a cluster that also holds identity-labelled faces.
    contamination: float = 0.0
    n_contaminants_checked: int = 0

    def as_row(self) -> dict[str, float | int]:
        return {
            "pairwise_p": round(self.pairwise_precision, 4),
            "pairwise_r": round(self.pairwise_recall, 4),
            "pairwise_f1": round(self.pairwise_f1, 4),
            "bcubed_p": round(self.bcubed_precision, 4),
            "bcubed_r": round(self.bcubed_recall, 4),
            "bcubed_f1": round(self.bcubed_f1, 4),
            "nmi": round(self.nmi, 4),
            "ari": round(self.ari, 4),
            "n_clusters_pred": self.n_clusters_pred,
            "n_clusters_true": self.n_clusters_true,
            "n_scored": self.n_scored,
            "pct_noise": round(self.pct_noise, 4),
            "contamination": round(self.contamination, 4),
        }


@dataclass
class SliceMetrics:
    """Per-face BCubed scores averaged within one slice.

    BCubed is used rather than pairwise because it is defined *per face*, so averaging it
    over a subset is meaningful. A pair can straddle two slices, which makes "pairwise F1 on
    profile faces" ambiguous; "the average BCubed score of profile faces" is not.
    """

    name: str
    n_faces: int
    precision: float
    recall: float
    f1: float


@dataclass
class Prediction:
    """A clustering to be scored: ``face_id -> cluster_id``, with -1 meaning ungrouped."""

    clusters: dict[int, int] = field(default_factory=dict)


def _expand_noise(labels: np.ndarray) -> np.ndarray:
    """Give every ungrouped face its own cluster id.

    Without this, all noise shares the id -1 and every pair of ungrouped faces counts as a
    successful grouping -- which would reward a system that simply refuses to cluster.
    """
    out = labels.astype(np.int64).copy()
    noise_at = np.flatnonzero(out == NOISE)
    if len(noise_at):
        out[noise_at] = np.arange(len(noise_at)) + out.max() + 1
    return out


def _codes(values: list[str]) -> np.ndarray:
    lookup: dict[str, int] = {}
    return np.array([lookup.setdefault(v, len(lookup)) for v in values], dtype=np.int64)


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def score(
    truth: dict[int, str],
    predicted: dict[int, int],
    *,
    contaminants: dict[int, str] | None = None,
) -> ClusteringMetrics:
    """Score a clustering. ``truth`` holds only identity-labelled faces.

    ``contaminants`` optionally maps face_id -> label for non_face / not_of_interest faces,
    used for the contamination figure only. They never enter the grouping metrics.
    """
    shared = sorted(set(truth) & set(predicted))
    if len(shared) < 2:
        empty = ClusteringMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, len(shared), 0.0)
        return empty

    true_codes = _codes([truth[f] for f in shared])
    raw_pred = np.array([predicted[f] for f in shared], dtype=np.int64)
    pred_codes = _expand_noise(raw_pred)

    same_true = true_codes[:, None] == true_codes[None, :]
    same_pred = pred_codes[:, None] == pred_codes[None, :]
    upper = np.triu(np.ones_like(same_true, dtype=bool), k=1)

    tp = int((same_true & same_pred & upper).sum())
    fp = int((~same_true & same_pred & upper).sum())
    fn = int((same_true & ~same_pred & upper).sum())

    pairwise_p = tp / (tp + fp) if tp + fp else 0.0
    pairwise_r = tp / (tp + fn) if tp + fn else 0.0

    # BCubed: for each face, how pure is its cluster and how much of its true class did it
    # recover. Less dominated by one large person than pairwise, which is why both are kept.
    correct = (same_true & same_pred).sum(axis=1).astype(np.float64)
    per_face_p = correct / same_pred.sum(axis=1)
    per_face_r = correct / same_true.sum(axis=1)
    bcubed_p = float(per_face_p.mean())
    bcubed_r = float(per_face_r.mean())

    contamination, n_checked = _contamination(predicted, shared, contaminants)

    return ClusteringMetrics(
        pairwise_precision=pairwise_p,
        pairwise_recall=pairwise_r,
        pairwise_f1=_f1(pairwise_p, pairwise_r),
        bcubed_precision=bcubed_p,
        bcubed_recall=bcubed_r,
        bcubed_f1=_f1(bcubed_p, bcubed_r),
        nmi=float(normalized_mutual_info_score(true_codes, pred_codes)),
        ari=float(adjusted_rand_score(true_codes, pred_codes)),
        n_clusters_pred=len(set(raw_pred.tolist()) - {NOISE}),
        n_clusters_true=len(set(true_codes.tolist())),
        n_scored=len(shared),
        pct_noise=float((raw_pred == NOISE).mean()),
        contamination=contamination,
        n_contaminants_checked=n_checked,
    )


def _contamination(
    predicted: dict[int, int],
    identity_faces: list[int],
    contaminants: dict[int, str] | None,
) -> tuple[float, int]:
    """Share of junk faces the system placed into a cluster holding real identities.

    Precision and recall say nothing about this: a poster filed into someone's album is
    invisible to them, because the poster is not identity-labelled. It is still a defect a
    user would notice immediately.
    """
    if not contaminants:
        return 0.0, 0

    person_clusters = {predicted[f] for f in identity_faces if predicted.get(f, NOISE) != NOISE}
    checked = [f for f in contaminants if f in predicted]
    if not checked:
        return 0.0, 0

    landed = sum(1 for f in checked if predicted[f] in person_clusters)
    return landed / len(checked), len(checked)


def score_by_slice(
    truth: dict[int, str],
    predicted: dict[int, int],
    slices: dict[int, str],
    *,
    min_faces: int = 10,
) -> list[SliceMetrics]:
    """Average per-face BCubed within each slice.

    A single aggregate number says the system is good or bad. This says *where* it is bad,
    which is the only form that tells you what to fix next.
    """
    shared = sorted(set(truth) & set(predicted) & set(slices))
    if len(shared) < 2:
        return []

    true_codes = _codes([truth[f] for f in shared])
    pred_codes = _expand_noise(np.array([predicted[f] for f in shared], dtype=np.int64))

    same_true = true_codes[:, None] == true_codes[None, :]
    same_pred = pred_codes[:, None] == pred_codes[None, :]
    correct = (same_true & same_pred).sum(axis=1).astype(np.float64)
    per_face_p = correct / same_pred.sum(axis=1)
    per_face_r = correct / same_true.sum(axis=1)

    out: list[SliceMetrics] = []
    for name in sorted({slices[f] for f in shared}):
        mask = np.array([slices[f] == name for f in shared])
        if int(mask.sum()) < min_faces:
            continue
        precision = float(per_face_p[mask].mean())
        recall = float(per_face_r[mask].mean())
        out.append(
            SliceMetrics(
                name=name,
                n_faces=int(mask.sum()),
                precision=precision,
                recall=recall,
                f1=_f1(precision, recall),
            )
        )
    return out
