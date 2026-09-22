#!/usr/bin/env python3
"""Reserve a fifth of the people, so tuning cannot grade its own homework.

Every threshold from here on gets picked by trying values and keeping the best-scoring one.
Do that against all your identities and the number you end up reporting is the number you
optimised -- it flatters by construction, and you would have no way to tell.

These people are set aside and never consulted while tuning. At the end they answer the one
question that matters: *does this work on people it was not tuned on?*

You do no labelling for this. It only decides which existing labels are used when.

Run it once, then leave it alone -- changing the split later makes old results incomparable.

Usage
    python scripts/make_holdout.py            # show the proposed split
    python scripts/make_holdout.py --write    # freeze it
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from faceindex import paths
from faceindex.eval import load_gold_set
from faceindex.eval.split import HOLDOUT, TUNE, make_split, write_split

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--write", action="store_true", help="Freeze the split to disk")
    args = parser.parse_args()

    gold_path = args.gold or (paths.gold_dir() / "labels.csv")
    out_path = args.out or (paths.gold_dir() / "split.csv")

    gold = load_gold_set(gold_path)
    console.print(f"[bold]Gold set :[/bold] {gold.summary()}\n")

    if out_path.exists() and not args.write:
        console.print(
            f"[yellow]A split already exists at {out_path}.[/yellow] "
            f"Leave it alone -- changing it makes earlier results incomparable."
        )

    years: dict[str, list[int]] = {}
    for face_id, person in gold.identities.items():
        taken = gold.attributes.get(face_id, {}).get("era_bucket", "")
        if taken:
            years.setdefault(person, [])
    spans = {person: (0, 0) for person in years}

    split = make_split(gold.identities, spans, holdout_fraction=args.fraction)

    counts = Counter(gold.identities.values())
    table = Table(title="Split by identity", header_style="bold")
    table.add_column("Side")
    table.add_column("People", justify="right")
    table.add_column("Faces", justify="right")
    table.add_column("Largest person", justify="right")

    for side in (TUNE, HOLDOUT):
        people = split.people(side)
        faces = sum(counts[p] for p in people)
        biggest = max((counts[p] for p in people), default=0)
        table.add_row(side, f"{len(people)}", f"{faces:,}", f"{biggest}")
    console.print(table)

    held = sorted(split.people(HOLDOUT), key=lambda p: -counts[p])
    console.print("\nReserved: " + ", ".join(f"{p} ({counts[p]})" for p in held[:12]))
    if len(held) > 12:
        console.print(f"…and {len(held) - 12} more")

    if not args.write:
        console.print("\nRe-run with [bold]--write[/bold] to freeze it.")
        return 0

    write_split(out_path, split)
    console.print(f"\n[green]Frozen to {out_path}[/green]")
    console.print(
        "Tune against [bold]--split tune[/bold]. Look at [bold]--split holdout[/bold] once, "
        "at the end, to report the honest number."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
