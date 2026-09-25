#!/usr/bin/env python3
"""How much does a score wobble when it is computed over only a handful of people?

The test set holds a few dozen identities. A BCubed F1 over that many people is partly
luck: draw a different few dozen and the figure moves. Without knowing how far it moves,
a test result cannot be read at all -- 0.93 against a dev score of 0.956 might be a real
regression or might be nothing.

This measures the wobble *before* the test set is opened, using only dev identities. It
repeatedly scores a random subset of dev people the same size as the test set, and reports
the range those scores span. That range is the tolerance to judge the test result against.

Clusters once, then re-scores subsets, so the cost is one clustering run regardless of how
many samples are drawn.

Usage
    python scripts/estimate_error_bar.py --model w600k_r50.onnx --algorithm components --similarity 0.53
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from faceindex import cluster, paths, store
from faceindex.eval import load_gold_set, score
from faceindex.eval.split import HOLDOUT, TUNE, load_split
from faceindex.progress import run_with_progress

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--algorithm", default="components", choices=["hdbscan", "components", "chinese_whispers"]
    )
    parser.add_argument("--similarity", type=float, default=0.53)
    parser.add_argument("--epsilon", type=float, default=0.95)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--samples", type=int, default=2000, help="How many subsets to draw")
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    gold = load_gold_set(paths.gold_dir() / "labels.csv")
    split = load_split(paths.gold_dir() / "split.csv")

    dev_truth = split.filter_identities(gold.identities, TUNE)
    dev_people = sorted({p for p in dev_truth.values()})
    n_holdout_people = len(split.people(HOLDOUT) & {p for p in gold.identities.values()})

    if n_holdout_people < 2:
        console.print("[red]Fewer than two identities reserved; nothing to estimate.[/red]")
        return 1
    if n_holdout_people >= len(dev_people):
        console.print(
            f"[red]The test set ({n_holdout_people} people) is not smaller than the dev set "
            f"({len(dev_people)}), so a subset of dev cannot stand in for it.[/red]"
        )
        return 1

    console.print(
        f"[bold]Stage 1/3[/bold] clustering once with {args.model} / {args.algorithm} / "
        f"{'epsilon' if args.algorithm == 'hdbscan' else 'similarity'} "
        f"{args.epsilon if args.algorithm == 'hdbscan' else args.similarity}"
    )
    with store.open_index(db_path, read_only=True) as conn:
        face_ids, matrix = cluster.load_embeddings(conn, model=args.model)
    if not face_ids:
        console.print(f"[red]No embeddings for {args.model}.[/red]")
        return 1

    config = cluster.ClusterConfig(
        min_cluster_size=args.min_cluster_size,
        algorithm=args.algorithm,
        similarity_threshold=args.similarity,
        selection_epsilon=args.epsilon,
        n_neighbors=args.neighbors,
        random_seed=args.seed,
        n_jobs=args.jobs if args.jobs is not None else cluster.default_jobs(),
    )
    result = run_with_progress(
        lambda: cluster.bootstrap_cluster(matrix, config),
        f"Clustering {len(face_ids):,} faces",
        estimate_seconds=None,
        console=console,
    )
    predicted = {int(f): int(c) for f, c in zip(face_ids, result.labels, strict=True)}

    console.print("\n[bold]Stage 2/3[/bold] scoring the whole dev set for reference")
    full = score(dev_truth, predicted)
    console.print(
        f"          dev BCubed F1 {full.bcubed_f1:.4f} "
        f"over {len(dev_people)} people / {len(dev_truth):,} faces\n"
    )

    console.print(
        f"[bold]Stage 3/3[/bold] scoring {args.samples:,} random draws of "
        f"{n_holdout_people} dev people (the test set's size)"
    )
    by_person: dict[str, list[int]] = {}
    for face, person in dev_truth.items():
        by_person.setdefault(person, []).append(face)

    rng = np.random.default_rng(args.seed)
    scores: list[float] = []
    for _ in tqdm(range(args.samples), unit="draw", smoothing=0.05):
        chosen = rng.choice(len(dev_people), size=n_holdout_people, replace=False)
        subset = {
            face: dev_people[int(i)]
            for i in chosen
            for face in by_person[dev_people[int(i)]]
        }
        scores.append(score(subset, predicted).bcubed_f1)

    values = np.array(scores)
    low, median, high = (float(np.percentile(values, q)) for q in (5, 50, 95))
    spread = high - low

    table = Table(title=f"Spread of a {n_holdout_people}-person score", header_style="bold")
    table.add_column("")
    table.add_column("BCubed F1", justify="right")
    table.add_row("whole dev set", f"{full.bcubed_f1:.4f}")
    table.add_row("subset: 5th percentile", f"{low:.4f}")
    table.add_row("subset: median", f"{median:.4f}")
    table.add_row("subset: 95th percentile", f"{high:.4f}")
    table.add_row("subset: worst seen", f"{values.min():.4f}")
    console.print()
    console.print(table)

    console.print(
        f"\n[bold]Verdict.[/bold] Scoring only {n_holdout_people} people makes the figure "
        f"move by about [bold]{spread:.3f}[/bold] purely by which people get drawn."
    )
    console.print(
        f"\nSo when the test set is run, read it like this:\n"
        f"  • [green]F1 at or above {low:.3f}[/green] — normal. The system works on new "
        f"people; the gap from {full.bcubed_f1:.4f} is the small-sample wobble, not a fault.\n"
        f"  • [yellow]F1 between {values.min():.3f} and {low:.3f}[/yellow] — weak but not "
        f"damning. Possible, though unlucky, if nothing is wrong. Look at which people did "
        f"badly before concluding anything.\n"
        f"  • [red]F1 below {values.min():.3f}[/red] — worse than any draw of dev people "
        f"managed. That is a real failure to generalise, not luck. Stop and diagnose.\n"
    )
    console.print(
        "[dim]Write these three numbers down now. Deciding what counts as a pass after "
        "seeing the result is how a test set stops meaning anything.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
