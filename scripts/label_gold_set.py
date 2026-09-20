#!/usr/bin/env python3
"""Serve the gold-set labelling UI on localhost.

This is the only manual step in Phase 1. Everything before it was automated so that the
human does nothing but confirm: accept a clean cluster with one key, click out the few
faces that do not belong, judge the leftovers individually.

Label carefully, once. A bad gold set makes every downstream number wrong in a way that
cannot be detected later, and the numbers stop being comparable if you keep editing it
after experiments begin (PLAN.md risk register).

The four labels
    person_N          same arbitrary id for the same human; names are irrelevant
    not_of_interest   a real face, but a stranger -- the clusterer must be free to call
                      them noise, or every wedding invents a dozen phantom people
    non_face          no face present in the scene: a pattern, statue, poster, or a face
                      inside a framed photograph hanging on the wall
    unsure            not readable from the face. Excluded from metrics. Never guess.

Judge from the face alone (decision 32)
    Mentally crop away everything but the face. If you are identifying someone by their
    earrings, hair, clothing or by who else is in the shot, mark it `unsure`.

    Your labels never reach the model -- there is no training here -- so this cannot teach
    it the wrong thing. The damage is to the yardstick. The system only ever receives face
    pixels, so a face whose identity is not in those pixels is unwinnable, and unwinnable
    cases sink the profile slice for a reason no amount of work can ever fix. You would
    then spend real effort chasing a number that cannot move.

    Apply it mechanically. A rule you follow identically at minute 5 and minute 90 beats a
    better rule you apply by feel.

Usage
    python scripts/label_gold_set.py
    python scripts/label_gold_set.py --port 9000
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from faceindex import labelui, paths, store

console = Console()


def print_queue(conn: object) -> int:
    """How much work is left, and how much of it is the slow kind.

    Big clusters are one keystroke each. Small ones are often not a person at all -- faces
    of poor quality embed into a similar mush, so the clusterer groups them by *being bad*
    rather than by identity -- and those have to be judged face by face. Knowing the split
    up front stops the slow tail from reading as something having gone wrong.
    """
    rows = conn.execute(  # type: ignore[attr-defined]
        """
        SELECT c.bootstrap_cluster AS cid, COUNT(*) AS n
        FROM gold_candidates c
        LEFT JOIN gold_labels g ON g.face_id = c.face_id
        WHERE g.face_id IS NULL
        GROUP BY c.bootstrap_cluster
        """
    ).fetchall()

    if not rows:
        console.print("[green]Nothing left to label.[/green]")
        return 0

    noise = sum(int(r["n"]) for r in rows if int(r["cid"]) == -1)
    clusters = [int(r["n"]) for r in rows if int(r["cid"]) != -1]
    singles = sum(n for n in clusters if n == 1)

    bands = (("10+ faces", 10, 10**9), ("5-9", 5, 10), ("2-4", 2, 5))
    table = Table(title="What is left", header_style="bold")
    table.add_column("Sheet")
    table.add_column("Sheets", justify="right")
    table.add_column("Faces", justify="right")
    table.add_column("Effort")

    for name, low, high in bands:
        sized = [n for n in clusters if low <= n < high]
        if sized:
            effort = "one keystroke each" if low >= 5 else "usually one, sometimes split"
            table.add_row(name, f"{len(sized):,}", f"{sum(sized):,}", effort)

    if singles:
        table.add_row("single-face piles", f"{singles:,}", f"{singles:,}", "judged individually")
    if noise:
        table.add_row("noise bucket", "—", f"{noise:,}", "judged individually")

    console.print(table)

    slow = singles + noise
    total = sum(clusters) + noise
    console.print(
        f"\n{total:,} faces left. Roughly [bold]{slow:,}[/bold] ({100 * slow / total:.0f}%) are "
        f"the face-by-face kind — singles and the noise bucket, which is not pre-grouped on "
        f"purpose."
    )
    console.print(
        "Small mixed piles are expected: a cluster of three blurry faces is usually three "
        "different people who merely look equally unreadable. Judge them individually, or "
        "press [bold]s[/bold] to skip and come back."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Localhost only by default. The data behind this server is biometric.",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--queue", action="store_true", help="What is left to label, and in what shape"
    )
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    if args.queue:
        with store.open_index(db_path, read_only=True) as conn:
            return print_queue(conn)

    with store.open_index(db_path, read_only=True) as conn:
        pending = conn.execute("SELECT COUNT(*) AS n FROM gold_candidates").fetchone()["n"]
    if not pending:
        console.print("[red]No gold candidates. Run scripts/sample_gold_set.py first.[/red]")
        return 1

    if args.host != "127.0.0.1":
        console.print(
            f"[yellow]Warning: binding to {args.host}, not localhost. This server exposes "
            f"face crops, which are biometric data. There is no authentication.[/yellow]"
        )

    server = labelui.serve(db_path, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}/"

    console.print(f"\n[bold]Labelling UI:[/bold] {url}")
    console.print(f"[bold]Candidates  :[/bold] {pending:,}")
    console.print(
        "\n[bold]Keys[/bold]  click=pull out/select · Enter=assign person · "
        "n=stranger · x=not a face · u=unsure · s=skip · t=toggle crop · Cmd/Ctrl+Z=undo"
    )
    console.print("\nCtrl-C to stop. Progress is saved continuously.\n")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\nStopping…")
    finally:
        server.shutdown()
        server.server_close()

    console.print(
        "[green]Session ended.[/green] "
        "When labelling is complete: python scripts/export_gold_set.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
