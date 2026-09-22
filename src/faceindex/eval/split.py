"""Splitting the gold set by identity, so tuning cannot quietly grade its own homework.

Every threshold in the project gets chosen by trying values and keeping whichever scores
best. Do that against all 124 identities and the number you report is the number you
optimised -- it will look good whatever the system is actually worth.

So a fifth of the people are set aside and never consulted while tuning. They answer the
only question that matters afterwards: *does this work for people it was not tuned on?*

Split by **identity, not by face and not by time**. Splitting by face would put the same
person on both sides, which leaks the answer. Splitting by time would recreate the era blind
spot the gold set was built to avoid.

The assignment lives in its own file so ``labels.csv`` stays frozen, and it is deterministic
so that results from different days remain comparable.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

TUNE = "tune"
HOLDOUT = "holdout"


@dataclass
class Split:
    """Which identities may be used for tuning, and which are reserved."""

    assignment: dict[str, str]

    def side(self, person: str) -> str:
        return self.assignment.get(person, TUNE)

    def people(self, side: str) -> set[str]:
        return {p for p, s in self.assignment.items() if s == side}

    def filter_identities(self, identities: dict[int, str], side: str) -> dict[int, str]:
        """Keep only the faces belonging to one side of the split."""
        if side == "all":
            return dict(identities)
        wanted = self.people(side)
        return {face: person for face, person in identities.items() if person in wanted}


def make_split(
    identities: dict[int, str],
    spans: dict[str, tuple[int, int]] | None = None,
    *,
    holdout_fraction: float = 0.2,
) -> Split:
    """Assign identities to tune/holdout, balanced so both sides look alike.

    A naive random split can hand every frequently-photographed person to one side, leaving
    the other made of people with two faces each -- and a score computed on that says nothing.
    People are therefore ordered by how many faces they have and whether they span years,
    then dealt out alternately, so both sides get a similar mix of big, small and long-lived
    identities.

    Deterministic: same gold set in, same split out, on any machine.
    """
    counts: dict[str, int] = defaultdict(int)
    for person in identities.values():
        counts[person] += 1

    spans = spans or {}

    def sort_key(person: str) -> tuple[int, int, str]:
        low, high = spans.get(person, (0, 0))
        # Long-lived people first, then by how many faces, then by name for determinism.
        return (-(high - low), -counts[person], person)

    ordered = sorted(counts, key=sort_key)

    every = max(2, round(1 / holdout_fraction))
    assignment = {
        person: (HOLDOUT if index % every == every - 1 else TUNE)
        for index, person in enumerate(ordered)
    }
    return Split(assignment=assignment)


def write_split(path: Path, split: Split) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["person_id", "split"])
        for person in sorted(split.assignment):
            writer.writerow([person, split.assignment[person]])


def load_split(path: Path) -> Split:
    if not path.exists():
        raise FileNotFoundError(
            f"No split at {path}. Run scripts/make_holdout.py to create and freeze it."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return Split(assignment={r["person_id"]: r["split"] for r in rows})
