#!/usr/bin/env python3
"""Serve the read-only review browser on localhost.

Read-only by design (v0): it exists to test whether the ranking puts real people first,
before any editing is built on top of an order that might be wrong.

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

    console.print(f"\n[bold green]Open http://{args.host}:{args.port}[/bold green]")
    console.print(
        "[dim]Nothing here writes to the database. Page through the first fifty piles and "
        "judge one thing: are they people you recognise? Keys: n next, p previous, t wider "
        "crop. 'worst' jumps to the bottom of the order, which is worth a look too.[/dim]"
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
