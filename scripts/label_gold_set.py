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

from faceindex import labelui, paths, store

console = Console()


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
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

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
