#!/usr/bin/env python3
"""Serve the face review tool on localhost.

Five screens. Review names piles in ranked order; Merge confirms that two groups are one
person, which is the highest-value action because splitting is the measured dominant error;
People repairs a person; Not filed holds what was set aside or never matched; Browse order
shows the whole ranking.

Nothing is ever deleted. Decisions are recorded about faces, so re-clustering with a better
model costs no human work.

Usage
    python scripts/review_faces.py
    python scripts/review_faces.py --port 8800 --run r50-components-0.51
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console

from faceindex import paths, reviewui

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--run", default=None, help="Which review run; default is the newest")
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    try:
        server = reviewui.serve(db_path, host=args.host, port=args.port, run_id=args.run)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    except OSError as exc:
        console.print(f"[red]Cannot listen on {args.host}:{args.port} — {exc}[/red]")
        console.print("Another copy may already be running. Try --port 8767.")
        return 1

    if getattr(server, "review_store", None) is not None and server.review_store.stale:
        console.print(
            "\n[yellow]This index was built before merge suggestions existed, so the Merge "
            "screen has nothing to compare.[/yellow] Rebuild it with "
            "[bold]python scripts/build_review_index.py --model w600k_r50.onnx "
            "--similarity 0.51[/bold]. Naming already done is kept: decisions are stored "
            "against faces, not piles."
        )

    console.print(f"\n[bold green]Open http://{args.host}:{args.port}[/bold green]")
    console.print(
        "[dim]Review: type a name and press Enter, or s to skip, j if it is not a person, "
        "u to undo. Merge: y same person, n different. Every action is undoable, and undo "
        "never reaches the identities carried in from the gold set.[/dim]"
    )
    console.print("[dim]Ctrl-C to stop.[/dim]\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
