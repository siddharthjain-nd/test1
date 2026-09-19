#!/usr/bin/env python3
"""Remove gold-set labels so they can be redone.

The in-browser Ctrl+Z only reaches the most recent action. This handles the rest: a bulk
mislabel noticed later, a pile judged before you understood the keys, or a decision to
start the pass over.

It touches only ``gold_labels``. Candidates, crops and embeddings are untouched, so
anything cleared here simply reappears in the labelling queue.

Usage
    python scripts/reset_labels.py --label unsure        # undo every "unsure" verdict
    python scripts/reset_labels.py --cluster 2398        # redo one pile
    python scripts/reset_labels.py --person cousin_a     # release one person's faces
    python scripts/reset_labels.py --last 51             # the 51 most recently labelled
    python scripts/reset_labels.py --all                 # start the whole pass again
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from faceindex import paths, store

console = Console()


def summarise(conn: object) -> int:
    rows = conn.execute(  # type: ignore[attr-defined]
        "SELECT label, COUNT(*) AS n FROM gold_labels GROUP BY label ORDER BY n DESC"
    ).fetchall()
    total = sum(int(r["n"]) for r in rows)

    if not total:
        console.print("[yellow]No labels recorded.[/yellow]")
        return 0

    table = Table(title="Current labels", header_style="bold")
    table.add_column("Label")
    table.add_column("Faces", justify="right")
    for row in rows:
        table.add_row(str(row["label"]), f"{int(row['n']):,}")
    table.add_row("[bold]total[/bold]", f"[bold]{total:,}[/bold]")
    console.print(table)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="Remove every label")
    parser.add_argument(
        "--label", help="Remove one verdict: person|not_of_interest|non_face|unsure"
    )
    parser.add_argument("--cluster", type=int, help="Remove labels for one bootstrap cluster")
    parser.add_argument("--person", help="Remove labels naming this person")
    parser.add_argument("--last", type=int, help="Remove the N most recently labelled faces")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    selectors = [args.all, args.label, args.cluster is not None, args.person, args.last]
    if sum(1 for s in selectors if s) != 1:
        console.print(
            "[red]Choose exactly one of --all, --label, --cluster, --person, --last.[/red]"
        )
        with store.open_index(db_path, read_only=True) as conn:
            summarise(conn)
        return 1

    if args.all:
        where, params, described = "1=1", (), "every label"
    elif args.label:
        where, params, described = "label = ?", (args.label,), f'every "{args.label}" verdict'
    elif args.cluster is not None:
        where = "face_id IN (SELECT face_id FROM gold_candidates WHERE bootstrap_cluster = ?)"
        params, described = (args.cluster,), f"cluster {args.cluster}"
    elif args.person:
        where, params, described = "person_id = ?", (args.person,), f'person "{args.person}"'
    else:
        where = "face_id IN (SELECT face_id FROM gold_labels ORDER BY labelled_at DESC LIMIT ?)"
        params, described = (args.last,), f"the {args.last} most recent labels"

    with store.open_index(db_path) as conn:
        affected = conn.execute(
            f"SELECT COUNT(*) AS n FROM gold_labels WHERE {where}", params
        ).fetchone()["n"]

        if not affected:
            console.print(f"[yellow]Nothing matches {described}.[/yellow]")
            summarise(conn)
            return 0

        console.print(f"About to remove [bold]{affected:,}[/bold] label(s) — {described}.")
        console.print("Crops and embeddings are untouched; these faces return to the queue.")

        if not args.yes:
            answer = input("Type 'yes' to confirm: ").strip().lower()
            if answer != "yes":
                console.print("Cancelled.")
                return 1

        conn.execute(f"DELETE FROM gold_labels WHERE {where}", params)
        conn.commit()
        console.print(f"[green]Removed {affected:,} label(s).[/green]\n")
        summarise(conn)

    console.print("\nResume with: python scripts/label_gold_set.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
