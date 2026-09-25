#!/usr/bin/env python3
"""Where does connected-components chaining start on the *full* face pool?

The gold set is ~1,700 faces; the library is ~64,000. Connected components is
single-linkage, so ONE bad edge fuses two whole people. The number of chances for
that edge grows with the number of pairs, not the number of faces -- 38x the faces
is roughly 1,400x the pairs. So the threshold that peaked on the gold set may sit on
the wrong side of the cliff at full scale.

Chaining needs no labels to detect: it shows up as a mega-cluster. If one cluster
holds thousands of faces, several people have been welded together, and no amount of
good BCubed on the tune split makes that acceptable.

This builds the neighbour graph ONCE and re-thresholds it, which is what makes
sweeping nine values affordable.

Usage
    python scripts/diagnose_threshold.py --model w600k_r50.onnx
    python scripts/diagnose_threshold.py --model w600k_r50.onnx --thresholds 0.49,0.51,0.53
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph

from faceindex import cluster, paths, store

console = Console()

DEFAULT_THRESHOLDS = "0.45,0.47,0.49,0.51,0.53,0.55,0.58,0.60,0.65"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--model", default=None, help="Default: the model with most faces")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Trial run on N faces")
    args = parser.parse_args()

    thresholds = sorted(float(t) for t in args.thresholds.split(","))

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    with store.open_index(db_path, read_only=True) as conn:
        models = cluster.available_models(conn)
        if not models:
            console.print("[red]No embeddings stored. Run scripts/embed_faces.py first.[/red]")
            return 1
        model = args.model or models[0][0]
        if model not in {m for m, _ in models}:
            console.print(f"[red]No embeddings for {model}.[/red] Stored: {[m for m, _ in models]}")
            return 1

        console.print(f"[bold]Stage 1/3[/bold] loading embeddings for {model}…")
        face_ids, matrix = cluster.load_embeddings(conn, model=model, limit=args.limit)

    if len(face_ids) < 2:
        console.print("[red]Not enough embedded faces.[/red]")
        return 1

    n = len(face_ids)
    jobs = args.jobs if args.jobs is not None else cluster.default_jobs()
    console.print(f"          {n:,} faces x {matrix.shape[1]}-d, {jobs} worker(s)\n")

    console.print(
        f"[bold]Stage 2/3[/bold] building the {args.neighbors}-neighbour graph "
        f"(the slow part; done once, then reused for every threshold)…"
    )
    started = time.perf_counter()
    graph = kneighbors_graph(
        matrix.astype(np.float32),
        n_neighbors=min(args.neighbors, n - 1),
        mode="distance",
        include_self=False,
        n_jobs=jobs,
    )
    graph.data = 1.0 - (graph.data**2) / 2.0  # squared Euclidean -> cosine, unit vectors
    graph = graph.maximum(graph.T).tocsr()
    console.print(
        f"          done in {time.perf_counter() - started:.0f}s, "
        f"{graph.nnz:,} edges, {graph.data.nbytes / 1e6:.0f} MB\n"
    )

    console.print(f"[bold]Stage 3/3[/bold] thresholding: {len(thresholds)} values\n")

    table = Table(title=f"Connected components on the full pool — {model}", header_style="bold")
    table.add_column("cos")
    table.add_column("clusters", justify="right")
    table.add_column("largest", justify="right")
    table.add_column("% of pool", justify="right")
    table.add_column("2nd", justify="right")
    table.add_column("3rd", justify="right")
    table.add_column("singletons", justify="right")
    table.add_column("verdict")

    for i, threshold in enumerate(thresholds, start=1):
        step = time.perf_counter()
        kept = graph.copy()
        kept.data[kept.data < threshold] = 0.0
        kept.eliminate_zeros()
        _, labels = connected_components(kept, directed=False)
        counts = np.sort(np.bincount(labels))[::-1]

        largest = int(counts[0])
        share = largest / n
        singletons = int((counts == 1).sum())

        # A real person tops out in the low hundreds of photos in a personal library.
        # A cluster holding several percent of everything is welded people, not a person.
        if share >= 0.10:
            verdict = "[red]chained[/red]"
        elif share >= 0.03:
            verdict = "[yellow]suspect[/yellow]"
        else:
            verdict = "[green]clean[/green]"

        table.add_row(
            f"{threshold:.2f}",
            f"{len(counts):,}",
            f"{largest:,}",
            f"{100 * share:.1f}%",
            f"{int(counts[1]):,}" if len(counts) > 1 else "-",
            f"{int(counts[2]):,}" if len(counts) > 2 else "-",
            f"{singletons:,}",
            verdict,
        )
        console.print(
            f"  [{i}/{len(thresholds)}] cos {threshold:.2f}: largest {largest:,} "
            f"({100 * share:.1f}%)  [dim]{time.perf_counter() - step:.0f}s[/dim]"
        )

    console.print()
    console.print(table)
    console.print(
        "\n[dim]No labels are used here. 'chained' means one cluster holds 10%+ of every "
        "face in the library, which no single person plausibly does — so that threshold "
        "has welded people together, whatever it scored on the gold set.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
