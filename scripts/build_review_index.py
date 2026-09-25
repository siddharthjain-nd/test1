#!/usr/bin/env python3
"""Cluster the pool and store the order the review UI will present piles in.

Run once per (model, threshold). Re-running the same settings replaces its own rows; a
different model or threshold is stored beside it, so two can be compared without a rebuild.

Usage
    python scripts/build_review_index.py --model w600k_r50.onnx --similarity 0.51
    python scripts/build_review_index.py --model w600k_r50.onnx --similarity 0.51 --preview 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from faceindex import cluster, paths, review, store

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--model", default="w600k_r50.onnx")
    parser.add_argument(
        "--algorithm", default="components", choices=["components", "chinese_whispers", "hdbscan"]
    )
    parser.add_argument("--similarity", type=float, default=0.51)
    parser.add_argument("--epsilon", type=float, default=0.95, help="hdbscan only")
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--preview", type=int, default=10, help="Top N piles to print")
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    started = time.perf_counter()
    stage_started = [time.perf_counter()]

    def say(message: str) -> None:
        elapsed = time.perf_counter() - stage_started[0]
        if elapsed > 0.5:
            console.print(f"          [dim]{elapsed:.0f}s[/dim]")
        stage_started[0] = time.perf_counter()
        console.print(f"[bold]•[/bold] {message}…")

    with store.open_index(db_path) as conn:
        available = dict(cluster.available_models(conn))
        if args.model not in available:
            console.print(
                f"[red]No embeddings for {args.model}.[/red] Stored: {sorted(available) or 'none'}"
            )
            return 1
        try:
            summary = review.build_index(
                conn,
                model=args.model,
                algorithm=args.algorithm,
                threshold=args.similarity,
                epsilon=args.epsilon,
                neighbors=args.neighbors,
                jobs=args.jobs,
                on_stage=say,
            )
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return 1

        top = review.list_piles(conn, summary.run_id, limit=max(args.preview, 0))
        for pile in top:
            pile["faces"] = review.pile_faces(conn, summary.run_id, int(pile["pile_id"]), limit=1)
        n_known = len(review.known_people(conn))

    console.print(f"          [dim]{time.perf_counter() - stage_started[0]:.0f}s[/dim]\n")

    overview = Table(title=f"Review index — {summary.run_id}", header_style="bold")
    overview.add_column("")
    overview.add_column("", justify="right")
    overview.add_row("Faces clustered", f"{summary.n_faces:,}")
    overview.add_row("Piles (2+ faces)", f"{summary.n_piles:,}")
    overview.add_row("Lone faces set aside", f"{summary.n_lone:,}")
    overview.add_row("People already named", f"{n_known}")
    overview.add_row("Built in", f"{time.perf_counter() - started:.0f}s")
    console.print(overview)

    if top:
        preview = Table(title=f"Top {len(top)} by review order", header_style="bold")
        for column in ("rank", "faces", "eye px", "coherence", "score"):
            preview.add_column(column, justify="right" if column != "rank" else "left")
        for rank, pile in enumerate(top, start=1):
            eye = pile["median_eye"]
            coherence = pile["coherence"]
            preview.add_row(
                f"#{rank}",
                f"{int(pile['n_faces']):,}",
                "?" if eye is None else f"{float(eye):.0f}",
                "?" if coherence is None else f"{float(coherence):.2f}",
                f"{float(pile['score']):.2f}",
            )
        console.print(preview)

    console.print()
    if summary.n_piles == 0:
        console.print(
            "[bold red]FAILED: no piles.[/bold red] Every face linked to nothing, so the "
            "threshold is far too strict for this model. Nothing to review."
        )
        return 1

    lone_share = summary.n_lone / max(summary.n_faces, 1)
    console.print(
        f"[bold green]Built.[/bold green] {summary.n_piles:,} piles to review, "
        f"{summary.n_lone:,} lone faces ({lone_share:.0%} of the pool) set aside."
    )
    console.print(
        "\n[bold]Next:[/bold] python scripts/review_faces.py\n"
        "[dim]Then look at the first fifty piles. If they are recognisable people, the "
        "order works and v1 can be built on it. If they are blurs, the score is wrong and "
        "we fix it before anything depends on it — which is the entire reason this "
        "version has no buttons.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
