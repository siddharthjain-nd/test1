"""Evaluation: scoring a clustering against the human-labelled gold set.

Nothing in this package is ever imported by the pipeline. Gold-set labels are evaluation
ground truth only and must never influence what the system does (PLAN.md decision 7).
"""

from faceindex.eval.goldset import GoldSet, load_gold_set
from faceindex.eval.metrics import ClusteringMetrics, SliceMetrics, score, score_by_slice
from faceindex.eval.split import Split, load_split, make_split, write_split

__all__ = [
    "ClusteringMetrics",
    "GoldSet",
    "SliceMetrics",
    "Split",
    "load_gold_set",
    "load_split",
    "make_split",
    "score",
    "score_by_slice",
    "write_split",
]
