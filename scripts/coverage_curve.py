#!/usr/bin/env python3
"""How many piles must a person name before most of their library is accounted for?

The review order only pays off if the useful piles are concentrated at the top. If covering
a library meant naming eight thousand groups, the design would be wrong and no amount of
ranking would save it -- nobody names eight thousand of anything.

So this measures the curve directly: work down the ranking and report what share of faces
has been covered after the first 10, 25, 50, 100 piles and so on. It reads the stored index
and writes nothing.

Faces are counted three ways, because they are not equally worth covering:
  all        -- every clustered face
  readable   -- eye distance of at least 30 px, the library median
  clear      -- at least 50 px, comfortably recognisable

Usage
    python scripts/coverage_curve.py
    python scripts/coverage_curve.py --run r50-components-0.51
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from faceindex import paths, review, store

console = Console()

STEPS = (10, 25, 50, 100, 200, 300, 500, 1000, 2000)
READABLE_PX = 30.0
CLEAR_PX = 50.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--run", default=None)
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    with store.open_index(db_path, read_only=True) as conn:
        chosen = (
            next((r for r in review.runs(conn) if r["run_id"] == args.run), None)
            if args.run
            else review.latest_run(conn)
        )
        if chosen is None:
            console.print("[red]No review index. Run scripts/build_review_index.py first.[/red]")
            return 1
        run_id = str(chosen["run_id"])

        # Faces per pile, split by how recognisable they are, in review order.
        rows = conn.execute(
            "SELECT p.pile_id, p.n_faces, "
            "  SUM(CASE WHEN f.interocular_px >= ? THEN 1 ELSE 0 END) AS readable, "
            "  SUM(CASE WHEN f.interocular_px >= ? THEN 1 ELSE 0 END) AS clear "
            "FROM review_piles p "
            "JOIN review_members m ON m.run_id = p.run_id AND m.pile_id = p.pile_id "
            "JOIN faces f ON f.id = m.face_id "
            "WHERE p.run_id = ? "
            "GROUP BY p.pile_id, p.n_faces, p.score "
            "ORDER BY p.score DESC, p.n_faces DESC, p.pile_id ASC",
            (READABLE_PX, CLEAR_PX, run_id),
        ).fetchall()

        totals = conn.execute(
            "SELECT COUNT(*) AS all_faces, "
            "  SUM(CASE WHEN f.interocular_px >= ? THEN 1 ELSE 0 END) AS readable, "
            "  SUM(CASE WHEN f.interocular_px >= ? THEN 1 ELSE 0 END) AS clear "
            "FROM review_members m JOIN faces f ON f.id = m.face_id WHERE m.run_id = ?",
            (READABLE_PX, CLEAR_PX, run_id),
        ).fetchone()

    if not rows:
        console.print("[red]This index has no piles.[/red]")
        return 1

    n_piles = len(rows)
    total_all = int(totals["all_faces"] or 0)
    total_readable = int(totals["readable"] or 0)
    total_clear = int(totals["clear"] or 0)

    console.print(
        f"[bold]{n_piles:,} piles[/bold] over {total_all:,} faces "
        f"({int(chosen['n_lone']):,} lone faces are not piles and are excluded).\n"
        f"[dim]{total_readable:,} faces are at least {READABLE_PX:.0f}px between the eyes, "
        f"{total_clear:,} at least {CLEAR_PX:.0f}px.[/dim]\n"
    )

    table = Table(title="Name this many piles, cover this much", header_style="bold")
    table.add_column("piles named", justify="right")
    table.add_column("faces", justify="right")
    table.add_column("of all", justify="right")
    table.add_column("of readable", justify="right")
    table.add_column("of clear", justify="right")

    running = readable = clear = 0
    index = 0
    marks = [s for s in STEPS if s <= n_piles] + [n_piles]
    for mark in marks:
        while index < mark:
            running += int(rows[index]["n_faces"])
            readable += int(rows[index]["readable"] or 0)
            clear += int(rows[index]["clear"] or 0)
            index += 1
        table.add_row(
            f"{mark:,}" + (" (all)" if mark == n_piles else ""),
            f"{running:,}",
            f"{100 * running / max(total_all, 1):.0f}%",
            f"{100 * readable / max(total_readable, 1):.0f}%",
            f"{100 * clear / max(total_clear, 1):.0f}%",
        )
    console.print(table)

    # How many piles to reach the usual coverage marks, counted on clear faces: those are
    # the ones a person would actually expect to find in somebody's album.
    def piles_for(target: float) -> int | None:
        seen = 0
        for position, row in enumerate(rows, start=1):
            seen += int(row["clear"] or 0)
            if seen >= target * max(total_clear, 1):
                return position
        return None

    console.print()
    marks_table = Table(title="Piles needed to cover the clear faces", header_style="bold")
    marks_table.add_column("coverage")
    marks_table.add_column("piles to name", justify="right")
    for target in (0.5, 0.7, 0.8, 0.9, 0.95):
        needed = piles_for(target)
        marks_table.add_row(f"{target:.0%}", "never" if needed is None else f"{needed:,}")
    console.print(marks_table)

    half = piles_for(0.5)
    eighty = piles_for(0.8)
    console.print()
    if half is None:
        console.print("[bold red]VERDICT: coverage never reaches half.[/bold red]")
    elif eighty is not None and eighty <= 400:
        console.print(
            f"[bold green]VERDICT: feasible.[/bold green] {half:,} piles covers half the clear "
            f"faces and {eighty:,} covers 80% — one or two sittings, not eight thousand. "
            f"The long tail below that is mostly strangers and background faces nobody names."
        )
    elif eighty is not None and eighty <= 1500:
        console.print(
            f"[bold yellow]VERDICT: heavy but possible.[/bold yellow] {eighty:,} piles for 80% "
            f"is several sittings. Worth reducing before this ships — by grouping harder, "
            f"or by filtering faces too poor to be worth naming."
        )
    else:
        console.print(
            f"[bold red]VERDICT: not feasible as it stands.[/bold red] Reaching 80% needs "
            f"{eighty:,} piles named. No one will do that. The clustering has to join more "
            f"per pile, or poor faces have to be kept out of the queue entirely."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
