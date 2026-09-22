#!/usr/bin/env python3
"""Cluster the whole face pool, score it against the gold set, record one row.

This is the point of everything before it. Each run appends a line to
``data/results/results.csv`` so configurations can be compared rather than argued about.

Two things worth understanding about how the score is produced.

**The full pool is clustered; only the labelled faces are scored.** Clustering just the
2,000 labelled faces would be artificially easy -- fewer distractors means fewer chances to
confuse similar-looking people, so the number would be inflated and meaningless. The other
61,000 faces are part of the problem being measured.

**The first number is meant to be mediocre.** It is a baseline, not a result. Its job is to
be the thing every later change has to beat.

Usage
    python scripts/run_experiment.py                          # baseline
    python scripts/run_experiment.py --min-cluster-size 5 --label mcs5
    python scripts/run_experiment.py --sweep 2,3,4,5,6,8      # one run per value
    python scripts/run_experiment.py --results                # show the table
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from faceindex import cluster, paths, store
from faceindex.eval import load_gold_set, score, score_by_slice
from faceindex.eval.goldset import SLICE_COLUMNS
from faceindex.eval.split import load_split

console = Console()

RESULT_COLUMNS = (
    "experiment_id",
    "timestamp",
    "label",
    "split",
    "embedder",
    "min_cluster_size",
    "min_samples",
    "pca",
    "n_faces_clustered",
    "n_scored",
    "n_people_true",
    "pairwise_p",
    "pairwise_r",
    "pairwise_f1",
    "bcubed_p",
    "bcubed_r",
    "bcubed_f1",
    "nmi",
    "ari",
    "n_clusters_pred",
    "n_clusters_true",
    "pct_noise",
    "contamination",
    "t_cluster_s",
    "platform",
)


def show_results(path: Path) -> int:
    if not path.exists():
        console.print(f"[yellow]No results yet at {path}.[/yellow]")
        return 1

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        console.print("[yellow]Results file is empty.[/yellow]")
        return 1

    table = Table(title="Experiments", header_style="bold")
    for column in ("label", "min_cluster_size", "pairwise_f1", "bcubed_f1", "ari"):
        table.add_column(column.replace("_", " "), justify="right" if "_" in column else "left")
    table.add_column("clusters", justify="right")
    table.add_column("noise", justify="right")

    best = max(rows, key=lambda r: float(r["bcubed_f1"]))
    for row in rows:
        mark = " [green]*[/green]" if row is best else ""
        table.add_row(
            row["label"] + mark,
            row["min_cluster_size"],
            row["pairwise_f1"],
            f"[bold]{row['bcubed_f1']}[/bold]",
            row["ari"],
            row["n_clusters_pred"],
            f"{100 * float(row['pct_noise']):.0f}%",
        )
    console.print(table)
    console.print(
        "[dim]* best BCubed F1 so far. BCubed is the headline: pairwise is dominated by "
        "whoever appears most often, so one person with hundreds of faces can carry it.[/dim]"
    )
    return 0


def run_once(
    conn: object,
    gold: object,
    *,
    min_cluster_size: int,
    min_samples: int | None,
    pca: int | None,
    seed: int,
) -> tuple[object, object, int, float]:
    face_ids, matrix = cluster.load_embeddings(conn)  # type: ignore[arg-type]
    if not face_ids:
        raise SystemExit("No embeddings. Run scripts/embed_faces.py first.")

    config = cluster.ClusterConfig(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        pca_components=pca,
        random_seed=seed,
    )
    started = time.perf_counter()
    result = cluster.bootstrap_cluster(matrix, config)
    elapsed = time.perf_counter() - started

    predicted = {int(f): int(c) for f, c in zip(face_ids, result.labels, strict=True)}
    metrics = score(
        gold.identities,  # type: ignore[attr-defined]
        predicted,
        contaminants=gold.contaminants,  # type: ignore[attr-defined]
    )
    return metrics, predicted, len(face_ids), elapsed


def print_report(metrics: object, slices: dict[str, list[object]]) -> None:
    table = Table(title="Score", header_style="bold")
    table.add_column("Metric")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")

    m = metrics  # type: ignore[assignment]
    table.add_row(
        "Pairwise",
        f"{m.pairwise_precision:.4f}",  # type: ignore[attr-defined]
        f"{m.pairwise_recall:.4f}",  # type: ignore[attr-defined]
        f"{m.pairwise_f1:.4f}",  # type: ignore[attr-defined]
    )
    table.add_row(
        "[bold]BCubed[/bold]",
        f"{m.bcubed_precision:.4f}",  # type: ignore[attr-defined]
        f"{m.bcubed_recall:.4f}",  # type: ignore[attr-defined]
        f"[bold]{m.bcubed_f1:.4f}[/bold]",  # type: ignore[attr-defined]
    )
    console.print(table)

    extra = Table(header_style="bold")
    extra.add_column("Metric")
    extra.add_column("Value", justify="right")
    extra.add_row("NMI", f"{m.nmi:.4f}")  # type: ignore[attr-defined]
    extra.add_row("ARI", f"{m.ari:.4f}")  # type: ignore[attr-defined]
    extra.add_row(
        "Clusters found vs real people",
        f"{m.n_clusters_pred:,} vs {m.n_clusters_true}",  # type: ignore[attr-defined]
    )
    extra.add_row("Faces scored", f"{m.n_scored:,}")  # type: ignore[attr-defined]
    extra.add_row("Left ungrouped", f"{m.pct_noise:.1%}")  # type: ignore[attr-defined]
    if m.n_contaminants_checked:  # type: ignore[attr-defined]
        extra.add_row(
            "Junk inside person albums",
            f"{m.contamination:.1%} of {m.n_contaminants_checked:,}",  # type: ignore[attr-defined]
        )
    console.print(extra)

    for column, entries in slices.items():
        if not entries:
            continue
        sliced = Table(
            title=f"By {column.replace('_bucket', '').replace('_', ' ')}", header_style="bold"
        )
        sliced.add_column("Bucket")
        sliced.add_column("Faces", justify="right")
        sliced.add_column("BCubed F1", justify="right")
        worst = min(entries, key=lambda s: s.f1)  # type: ignore[attr-defined]
        for entry in entries:
            colour = "red" if entry is worst else "white"  # type: ignore[attr-defined]
            sliced.add_row(
                entry.name,  # type: ignore[attr-defined]
                f"{entry.n_faces:,}",  # type: ignore[attr-defined]
                f"[{colour}]{entry.f1:.4f}[/{colour}]",  # type: ignore[attr-defined]
            )
        console.print(sliced)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--results", action="store_true", help="Print the table and exit")
    parser.add_argument("--label", default=None, help="Name for this run in the table")
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--pca", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument(
        "--sweep", default=None, help="Comma-separated min-cluster-size values, one run each"
    )
    parser.add_argument(
        "--split",
        choices=("tune", "holdout", "all"),
        default="tune",
        help="Which identities to score. Tune against 'tune'; look at 'holdout' once, at the "
        "end. Scoring the people you tuned on reports the number you optimised.",
    )
    args = parser.parse_args()

    results_path = paths.results_dir() / "results.csv"
    if args.results:
        return show_results(results_path)

    db_path = args.db or paths.index_db_path()
    gold_path = args.gold or (paths.gold_dir() / "labels.csv")
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    gold = load_gold_set(gold_path)
    console.print(f"[bold]Gold set :[/bold] {gold.summary()}")

    split_path = paths.gold_dir() / "split.csv"
    if args.split != "all":
        if not split_path.exists():
            console.print(
                f"[red]No held-out split at {split_path}.[/red]\n"
                f"Run scripts/make_holdout.py first, or pass --split all to score everyone "
                f"(which reports the number you tuned on)."
            )
            return 1
        split = load_split(split_path)
        gold.identities = split.filter_identities(gold.identities, args.split)
        console.print(
            f"[bold]Split    :[/bold] {args.split} — "
            f"{len({p for p in gold.identities.values()})} people, "
            f"{len(gold.identities):,} faces"
        )
        if args.split == "holdout":
            console.print(
                "[yellow]Scoring the reserved people. Use this to report, not to tune — "
                "every look costs a little of its independence.[/yellow]"
            )

    sizes = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.min_cluster_size]

    with store.open_index(db_path, read_only=True) as conn:
        for size in sizes:
            label = args.label or (f"mcs{size}" if len(sizes) > 1 else "baseline")
            console.print(f"\n[bold]Run      :[/bold] {label} (min cluster size {size})")

            metrics, predicted, n_faces, elapsed = run_once(
                conn,
                gold,
                min_cluster_size=size,
                min_samples=args.min_samples,
                pca=args.pca,
                seed=args.seed,
            )
            console.print(f"Clustered {n_faces:,} faces in {elapsed:.0f}s.\n")

            slices = {
                column: score_by_slice(gold.identities, predicted, gold.slice_values(column))
                for column in SLICE_COLUMNS
            }
            print_report(metrics, slices)

            run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{label}"
            row = {
                "experiment_id": run_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "label": label,
                "split": args.split,
                "embedder": "w600k_mbf.onnx",
                "min_cluster_size": size,
                "min_samples": args.min_samples or "",
                "pca": args.pca or "",
                "n_faces_clustered": n_faces,
                "n_people_true": gold.n_people,
                "t_cluster_s": round(elapsed, 1),
                "platform": f"{platform.system()}-{platform.machine()}",
                **metrics.as_row(),
            }
            results_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not results_path.exists()
            with results_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(RESULT_COLUMNS))
                if write_header:
                    writer.writeheader()
                writer.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})

    console.print(f"\n[green]Recorded in {results_path}[/green]")
    console.print("Compare runs with: python scripts/run_experiment.py --results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
