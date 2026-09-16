#!/usr/bin/env python3
"""Rough clustering of the face pool, purely to make labelling fast.

This output is never a result. It pre-groups crops so a human can confirm a whole cluster
with one keystroke instead of sorting 2,000 loose faces by hand (Register C6). Do not tune
it -- time spent improving bootstrap quality is time not spent labelling, and the labels
are the only durable artifact of Phase 1.

Usage
    python scripts/bootstrap_cluster.py
    python scripts/bootstrap_cluster.py --min-cluster-size 4
    python scripts/bootstrap_cluster.py --pca 128     # only if the full run is too slow
    python scripts/bootstrap_cluster.py --stats
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from faceindex import cluster, paths, store

console = Console()


def print_stats(conn: object) -> None:
    sizes = cluster.cluster_sizes(conn)  # type: ignore[arg-type]
    if not sizes:
        console.print("[yellow]No bootstrap clustering has been run yet.[/yellow]")
        return

    total = sum(n for _, n in sizes)
    noise = next((n for cid, n in sizes if cid == -1), 0)
    real = [(cid, n) for cid, n in sizes if cid != -1]

    table = Table(title="Bootstrap clustering", header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Faces clustered", f"{total:,}")
    table.add_row("Clusters found", f"{len(real):,}")
    table.add_row("Noise", f"{noise:,} ({100 * noise / total:.1f}%)")
    if real:
        table.add_row("Largest cluster", f"{real[0][1]:,} faces")
        table.add_row("Median cluster", f"{real[len(real) // 2][1]:,} faces")
    console.print(table)

    if real:
        top = Table(title="Ten largest clusters", header_style="bold")
        top.add_column("Cluster", justify="right")
        top.add_column("Faces", justify="right")
        for cid, n in real[:10]:
            top.add_row(str(cid), f"{n:,}")
        console.print(top)

    if noise / total > 0.5:
        console.print(
            "[yellow]Over half the pool is noise.[/yellow] Expected for a bootstrap pass "
            "over a corpus that is 34% tiny faces. The sampler reserves a share of this "
            "bucket for review, so it is not lost."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument(
        "--pca",
        type=int,
        default=None,
        help="Reduce to N dimensions before clustering. Off by default: measure first.",
    )
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--stats", action="store_true", help="Report and exit")
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    if args.stats:
        with store.open_index(db_path, read_only=True) as conn:
            print_stats(conn)
        return 0

    config = cluster.ClusterConfig(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        pca_components=args.pca,
        random_seed=args.seed,
    )

    with store.open_index(db_path) as conn:
        console.print("Loading embeddings…")
        face_ids, matrix = cluster.load_embeddings(conn)

        if not face_ids:
            console.print("[red]No embeddings found. Run scripts/embed_faces.py first.[/red]")
            return 1

        console.print(f"[bold]Faces    :[/bold] {len(face_ids):,} x {matrix.shape[1]}-d")
        console.print(f"[bold]Min size :[/bold] {args.min_cluster_size}")
        console.print(f"[bold]PCA      :[/bold] {args.pca or 'off (full dimension)'}\n")

        started = time.perf_counter()
        result = cluster.bootstrap_cluster(matrix, config)
        elapsed = time.perf_counter() - started

        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        cluster.write_clusters(conn, face_ids, result, run_id)

        console.print(
            f"Clustered {len(face_ids):,} faces in {elapsed:.1f}s: "
            f"{result.n_clusters:,} clusters, {result.noise_fraction:.1%} noise.\n"
        )
        print_stats(conn)

    console.print(
        "\n[green]Bootstrap clustering done.[/green] Next: python scripts/sample_gold_set.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
