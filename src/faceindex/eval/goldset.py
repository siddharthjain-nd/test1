"""Reading the frozen gold set.

``data/gold/labels.csv`` is the portable artifact -- a few hundred kilobytes that travels
between machines alongside the crops. It is deliberately read from the CSV rather than the
database, because the CSV is what gets frozen: once experiments begin, results are only
comparable if the ground truth stops moving.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from faceindex.eval.metrics import IDENTITY_LABEL

# Attributes carried per face so results can be sliced. "F1 is 0.92 overall but 0.61 on
# profile faces" is actionable; a single aggregate is not.
SLICE_COLUMNS = (
    "size_bucket",
    "pose_bucket",
    "era_bucket",
    "quality_bucket",
    "kind",
    "filtered_bucket",
    "group_bucket",
)


@dataclass
class GoldSet:
    """Human ground truth. Never an input to the pipeline, only a yardstick for it."""

    path: Path
    identities: dict[int, str] = field(default_factory=dict)
    contaminants: dict[int, str] = field(default_factory=dict)
    excluded: dict[int, str] = field(default_factory=dict)
    attributes: dict[int, dict[str, str]] = field(default_factory=dict)

    @property
    def n_people(self) -> int:
        return len({person for person in self.identities.values()})

    def slice_values(self, column: str) -> dict[int, str]:
        """``face_id -> bucket`` for one attribute, over identity-labelled faces only."""
        return {
            face_id: attrs[column]
            for face_id, attrs in self.attributes.items()
            if face_id in self.identities and attrs.get(column)
        }

    def summary(self) -> str:
        return (
            f"{len(self.identities):,} identity faces across {self.n_people} people · "
            f"{len(self.contaminants):,} stranger/non-face · {len(self.excluded):,} unsure"
        )


def load_gold_set(path: Path) -> GoldSet:
    """Parse labels.csv into the three groups the metrics treat differently."""
    if not path.exists():
        raise FileNotFoundError(
            f"No gold set at {path}. Run scripts/export_gold_set.py to write it."
        )

    gold = GoldSet(path=path)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            face_id = int(row["face_id"])
            label = row["label"]

            if label == IDENTITY_LABEL and row.get("person_id"):
                gold.identities[face_id] = row["person_id"]
            elif label in ("not_of_interest", "non_face"):
                # Kept apart from identity scoring, but needed for the contamination figure.
                gold.contaminants[face_id] = label
            else:
                # 'unsure', and anything malformed. Excluded, as promised at labelling time.
                gold.excluded[face_id] = label

            gold.attributes[face_id] = {column: row.get(column, "") for column in SLICE_COLUMNS}

    return gold
