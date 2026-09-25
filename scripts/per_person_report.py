#!/usr/bin/env python3
"""Which individual people the clustering got wrong, and whether the damage is concentrated.

Bucket slices ("people with 15-39 photos") stop meaning anything on a small split: a bucket
can hold three people, so one bad identity sinks it and the bucket name gets blamed for
something that has nothing to do with it.

This drops to the level that is actually real -- one person at a time. It then asks the
question that decides what to do next: is the shortfall spread across everybody, or carried
by a couple of identities? Spread means a systematic weakness worth fixing. Concentrated
means a few hard people and small-sample luck, and the aggregate score should not be
over-read.

Usage
    python scripts/per_person_report.py --model w600k_r50.onnx --similarity 0.51 --split holdout
    python scripts/per_person_report.py --model w600k_r50.onnx --similarity 0.51 --split tune
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from rich.console import Console
from rich.table import Table

from faceindex import cluster, paths, store
from faceindex.eval import load_gold_set
from faceindex.eval.metrics import _codes, _expand_noise, _f1
from faceindex.eval.split import load_split
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
    parser.add_argument("--similarity", type=float, default=0.51)
    parser.add_argument("--epsilon", type=float, default=0.95)
    parser.add_argument("--split", default="holdout", choices=["tune", "holdout", "all"])
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--worst", type=int, default=12, help="How many people to list")
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    gold = load_gold_set(paths.gold_dir() / "labels.csv")
    split = load_split(paths.gold_dir() / "split.csv")
    truth = split.filter_identities(gold.identities, args.split)
    if len(truth) < 2:
        console.print(f"[red]Nothing labelled on the {args.split} side.[/red]")
        return 1

    console.print(f"[bold]Stage 1/2[/bold] clustering with {args.model} / {args.algorithm}")
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
        n_jobs=args.jobs if args.jobs is not None else cluster.default_jobs(),
    )
    result = run_with_progress(
        lambda: cluster.bootstrap_cluster(matrix, config),
        f"Clustering {len(face_ids):,} faces",
        estimate_seconds=None,
        console=console,
    )
    predicted = {int(f): int(c) for f, c in zip(face_ids, result.labels, strict=True)}

    console.print("\n[bold]Stage 2/2[/bold] scoring each person separately")
    shared = sorted(set(truth) & set(predicted))
    true_codes = _codes([truth[f] for f in shared])
    pred_codes = _expand_noise(np.array([predicted[f] for f in shared], dtype=np.int64))

    same_true = true_codes[:, None] == true_codes[None, :]
    same_pred = pred_codes[:, None] == pred_codes[None, :]
    correct = (same_true & same_pred).sum(axis=1).astype(np.float64)
    per_face_p = correct / same_pred.sum(axis=1)
    per_face_r = correct / same_true.sum(axis=1)

    people = sorted({truth[f] for f in shared})
    stats = []
    for person in people:
        mask = np.array([truth[f] == person for f in shared])
        precision = float(per_face_p[mask].mean())
        recall = float(per_face_r[mask].mean())
        piles = len({int(pred_codes[i]) for i in np.flatnonzero(mask)})
        stats.append(
            {
                "person": person,
                "faces": int(mask.sum()),
                "piles": piles,
                "p": precision,
                "r": recall,
                "f1": _f1(precision, recall),
                "mask": mask,
            }
        )

    overall_p, overall_r = float(per_face_p.mean()), float(per_face_r.mean())
    overall = _f1(overall_p, overall_r)

    stats.sort(key=lambda s: s["f1"])
    table = Table(
        title=f"Worst {min(args.worst, len(stats))} people on the {args.split} split "
        f"({len(people)} people, {len(shared)} faces)",
        header_style="bold",
    )
    for column, justify in (
        ("person", "left"),
        ("faces", "right"),
        ("piles", "right"),
        ("BCubed P", "right"),
        ("BCubed R", "right"),
        ("BCubed F1", "right"),
        ("what went wrong", "left"),
    ):
        table.add_column(column, justify=justify)

    for entry in stats[: args.worst]:
        if entry["p"] < 0.9 and entry["r"] < 0.9:
            what = "[red]mixed with others and split[/red]"
        elif entry["p"] < 0.9:
            what = "[red]strangers in their album[/red]"
        elif entry["piles"] > 1:
            what = f"[yellow]split across {entry['piles']} piles[/yellow]"
        else:
            what = "[green]clean[/green]"
        table.add_row(
            str(entry["person"]),
            str(entry["faces"]),
            str(entry["piles"]),
            f"{entry['p']:.3f}",
            f"{entry['r']:.3f}",
            f"{entry['f1']:.3f}",
            what,
        )
    console.print()
    console.print(table)

    # Drop the worst people one at a time and watch the aggregate recover. A score carried
    # by two identities is a different situation from one that is uniformly mediocre, and
    # only this distinguishes them.
    console.print()
    recovery = Table(title="If the worst people are set aside", header_style="bold")
    recovery.add_column("excluding")
    recovery.add_column("people left", justify="right")
    recovery.add_column("BCubed F1", justify="right")
    recovery.add_row("nobody (the reported score)", str(len(people)), f"{overall:.4f}")

    dropped: np.ndarray = np.zeros(len(shared), dtype=bool)
    recovered: list[float] = []
    for k in range(1, min(4, len(stats))):
        dropped = dropped | stats[k - 1]["mask"]
        keep = ~dropped
        value = _f1(float(per_face_p[keep].mean()), float(per_face_r[keep].mean()))
        recovered.append(value)
        names = ", ".join(str(s["person"]) for s in stats[:k])
        recovery.add_row(names, str(len(people) - k), f"{value:.4f}")
    console.print(recovery)

    console.print()
    split_only = [s for s in stats if s["p"] >= 0.9 and s["piles"] > 1]
    mixed = [s for s in stats if s["p"] < 0.9]

    # Whether the shortfall is concentrated is decided by comparing the TYPICAL person to
    # the aggregate. If the median person is near-perfect while the average is not, a few
    # identities are carrying the loss. If the median person is also mediocre, everybody is
    # a bit wrong and that is systematic. Recovery from dropping the worst few is supporting
    # evidence, not the test -- on a high score there is nothing to recover and the
    # recovery rule alone would call a clean result "systematic".
    median_f1 = float(np.median([s_["f1"] for s_ in stats]))
    weak = [s_ for s_ in stats if s_["f1"] < 0.9]
    lift = (recovered[-1] - overall) if recovered else 0.0

    console.print(
        f"[dim]typical (median) person scores {median_f1:.4f}; the aggregate is "
        f"{overall:.4f}; dropping the worst {len(recovered)} lifts it by {lift:+.4f}; "
        f"{len(weak)} of {len(people)} people are below 0.9.[/dim]\n"
    )

    if not weak:
        console.print(
            f"[bold green]VERDICT: clean.[/bold green] Every one of the {len(people)} people "
            f"scores above 0.9, and the typical person is at {median_f1:.4f}. There is no "
            f"weak group to chase here."
        )
    elif median_f1 - overall > 0.03:
        # Concentrated means the typical person is clearly better than the average, i.e. a
        # minority is dragging the mean down. "Dropping the worst few lifts the score" is
        # NOT sufficient on its own -- that is true even when everybody is equally bad --
        # so it is reported as evidence but does not decide the verdict.
        console.print(
            f"[bold yellow]VERDICT: concentrated.[/bold yellow] The typical person scores "
            f"{median_f1:.4f} but the aggregate is {overall:.4f}, and setting aside "
            f"{len(recovered)} of {len(people)} people lifts it to "
            f"{(recovered[-1] if recovered else overall):.4f}. A handful of hard identities carry the loss — it is not a "
            f"weakness spread across everybody. Do not read the aggregate as a verdict on "
            f"the method, and do not retune on it."
        )
    else:
        console.print(
            f"[bold red]VERDICT: spread out.[/bold red] The typical person scores "
            f"{median_f1:.4f}, close to the aggregate {overall:.4f}, and dropping the worst "
            f"{len(recovered)} barely moves it. Most people are a little wrong, which is a "
            f"systematic weakness and worth fixing."
        )

    console.print(
        f"\n{len(mixed)} of {len(people)} people have strangers in their album "
        f"(precision below 0.9) — the expensive error.\n"
        f"{len(split_only)} are clean but split across several piles — the cheap error, "
        f"one merge each in the review UI."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
