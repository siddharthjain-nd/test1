#!/usr/bin/env python3
"""Remove gold-set labels so they can be redone.

The in-browser Ctrl+Z only reaches the most recent action. This handles the rest: a bulk
mislabel noticed later, a pile judged before you understood the keys, or a decision to
start the pass over.

It touches only ``gold_labels``. Candidates, crops and embeddings are untouched, so
anything cleared here simply reappears in the labelling queue.

Usage
    python scripts/reset_labels.py --people              # who is named, and near-duplicates
    python scripts/reset_labels.py --merge Mom mom       # one person typed two ways
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


def _distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def list_people(conn: object) -> int:
    """Everyone named, with near-duplicate spellings flagged.

    One person typed two ways is the worst error this task can produce: it splits a human in
    the answer key, and the scorer then penalises the system for grouping them correctly.
    """
    rows = conn.execute(  # type: ignore[attr-defined]
        "SELECT person_id, COUNT(*) AS n FROM gold_labels "
        "WHERE person_id IS NOT NULL GROUP BY person_id ORDER BY n DESC"
    ).fetchall()
    if not rows:
        console.print("[yellow]Nobody named yet.[/yellow]")
        return 0

    table = Table(title="People named so far", header_style="bold")
    table.add_column("Person")
    table.add_column("Faces", justify="right")
    for row in rows:
        table.add_row(str(row["person_id"]), f"{int(row['n']):,}")
    console.print(table)

    names = [str(r["person_id"]) for r in rows]
    suspects = [
        (a, b)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if _distance(a.lower().strip(), b.lower().strip()) <= 2
    ]
    if suspects:
        console.print("\n[yellow]Possible duplicates — same person typed two ways?[/yellow]")
        for a, b in suspects:
            console.print(f"  • [bold]{a}[/bold] vs [bold]{b}[/bold]")
        console.print(
            "\nIf either pair is one person, join them:\n"
            "  python scripts/reset_labels.py --merge <from> <to>"
        )
    else:
        console.print("\n[green]No near-duplicate names.[/green]")
    return 0


def merge_people(conn: object, source: str, target: str, *, assume_yes: bool) -> int:
    """Move one person's labels onto another name. Nothing is deleted.

    Covers both a plain rename (the target does not exist yet) and joining two names that
    turned out to be the same human -- the same operation either way.
    """
    if source == target:
        console.print("[red]Those are the same name.[/red]")
        return 1

    counts = {
        name: int(
            conn.execute(  # type: ignore[attr-defined]
                "SELECT COUNT(*) AS n FROM gold_labels WHERE person_id = ?", (name,)
            ).fetchone()["n"]
        )
        for name in (source, target)
    }
    if not counts[source]:
        console.print(f'[red]No labels for "{source}".[/red] Run --people to see the names.')
        return 1

    if counts[target]:
        console.print(
            f"Joining [bold]{source}[/bold] ({counts[source]} faces) and "
            f"[bold]{target}[/bold] ({counts[target]} faces) into one person "
            f"-> {counts[source] + counts[target]} faces."
        )
    else:
        console.print(
            f"Renaming [bold]{source}[/bold] to [bold]{target}[/bold] "
            f"({counts[source]} faces). No existing person by that name."
        )

    if not assume_yes and input("Type 'yes' to confirm: ").strip().lower() != "yes":
        console.print("Cancelled.")
        return 1

    conn.execute(  # type: ignore[attr-defined]
        "UPDATE gold_labels SET person_id = ? WHERE person_id = ?", (target, source)
    )
    conn.commit()  # type: ignore[attr-defined]
    console.print(f"[green]Done. {target} now has {counts[source] + counts[target]} faces.[/green]")
    return 0


def clear_faces(conn: object, face_ids: list[int], *, assume_yes: bool) -> int:
    """Unlabel individual faces so they return to the queue.

    For the case where one face landed on the wrong person. Clearing the whole identity
    would throw away dozens of correct judgements to fix a single wrong one.
    """
    rows = conn.execute(  # type: ignore[attr-defined]
        f"SELECT face_id, label, person_id FROM gold_labels "
        f"WHERE face_id IN ({','.join('?' for _ in face_ids)})",
        face_ids,
    ).fetchall()

    if not rows:
        console.print("[yellow]None of those faces are labelled.[/yellow]")
        return 0

    table = Table(title="About to unlabel", header_style="bold")
    table.add_column("Face", justify="right")
    table.add_column("Current label")
    for row in rows:
        table.add_row(str(row["face_id"]), str(row["person_id"] or row["label"]))
    console.print(table)

    if not assume_yes and input("Type 'yes' to confirm: ").strip().lower() != "yes":
        console.print("Cancelled.")
        return 1

    conn.executemany(  # type: ignore[attr-defined]
        "DELETE FROM gold_labels WHERE face_id = ?", [(i,) for i in face_ids]
    )
    conn.commit()  # type: ignore[attr-defined]
    console.print(f"[green]Unlabelled {len(rows)} face(s); they return to the queue.[/green]")
    return 0


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
    parser.add_argument(
        "--people", action="store_true", help="List everyone named so far and flag near-duplicates"
    )
    parser.add_argument(
        "--merge",
        nargs=2,
        metavar=("FROM", "TO"),
        help="Move one person's faces onto another name. Renames, or joins a split identity.",
    )
    parser.add_argument(
        "--rename",
        nargs=2,
        metavar=("FROM", "TO"),
        help="Same as --merge; clearer when you simply gave someone the wrong name.",
    )
    parser.add_argument(
        "--face",
        type=int,
        nargs="+",
        metavar="ID",
        help="Unlabel these face ids, e.g. one face put on the wrong person.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    if args.people:
        with store.open_index(db_path, read_only=True) as conn:
            return list_people(conn)

    move = args.merge or args.rename
    if move:
        with store.open_index(db_path) as conn:
            return merge_people(conn, move[0], move[1], assume_yes=args.yes)

    if args.face:
        with store.open_index(db_path) as conn:
            return clear_faces(conn, args.face, assume_yes=args.yes)

    selectors: list[object] = [
        args.all,
        args.label,
        args.cluster is not None,
        args.person,
        args.last,
    ]
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
