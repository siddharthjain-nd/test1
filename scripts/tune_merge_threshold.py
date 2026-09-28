#!/usr/bin/env python3
"""What similarity should a merge suggestion need? Measured, not guessed.

The merge screen currently offers any person/pile pair above cosine 0.30. That number was
picked by me on the reasoning that "a suggestion only has to be worth a glance", which is
not evidence. If it is too low the queue fills with pairs that are obviously different
people and the screen wastes the one thing it is meant to save: attention.

The gold set can settle it. Piles that contain labelled faces have a known identity, so for
every pair of such piles we know whether they are really the same person, and we know how
similar their centroids are. That gives a precision/recall curve over the threshold and a
defensible place to put it.

Only tuning identities are used; the reserved ones stay out, as everywhere else.

Usage
    python scripts/tune_merge_threshold.py
    python scripts/tune_merge_threshold.py --run r50-components-0.51
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from rich.console import Console
from rich.table import Table

from faceindex import paths, review, store
from faceindex.eval import load_gold_set
from faceindex.eval.split import HOLDOUT, load_split

console = Console()

BANDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--run", default=None, help="Default: the newest review index")
    parser.add_argument(
        "--purity",
        type=float,
        default=0.8,
        help="A pile counts as one person's only if this share of its labelled faces agree",
    )
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    gold = load_gold_set(paths.gold_dir() / "labels.csv")
    reserved = load_split(paths.gold_dir() / "split.csv").people(HOLDOUT)
    truth = {f: p for f, p in gold.identities.items() if p not in reserved}

    with store.open_index(db_path, read_only=True) as conn:
        chosen = (
            next((r for r in review.runs(conn) if r["run_id"] == args.run), None)
            if args.run
            else review.latest_run(conn)
        )
        if chosen is None:
            console.print("[red]No review index. Run scripts/build_review_index.py first.[/red]")
            return 1
        run_id = str(chosen["run_id"])

        rows = conn.execute(
            "SELECT pile_id, face_id FROM review_members WHERE run_id = ? AND pile_id >= 0",
            (run_id,),
        ).fetchall()
        pile_ids, matrix = review._pile_centroids(conn, run_id)

    if not pile_ids:
        console.print("[red]This index has no pile centroids. Rebuild it.[/red]")
        return 1

    # Which identity, if any, each pile belongs to, judged only on its labelled faces.
    labelled: dict[int, Counter] = {}
    for row in rows:
        person = truth.get(int(row["face_id"]))
        if person:
            labelled.setdefault(int(row["pile_id"]), Counter())[person] += 1

    identity: dict[int, str] = {}
    for pile_id, counts in labelled.items():
        person, n = counts.most_common(1)[0]
        if n / sum(counts.values()) >= args.purity:
            identity[pile_id] = person

    index = {pile_id: position for position, pile_id in enumerate(pile_ids)}
    judged = sorted(p for p in identity if p in index)
    if len(judged) < 2:
        console.print(
            f"[red]Only {len(judged)} pile(s) carry enough labelled faces to judge.[/red]"
        )
        return 1

    positions = [index[p] for p in judged]
    block = matrix[positions]
    similarity = block @ block.T
    same = np.array([[identity[a] == identity[b] for b in judged] for a in judged])
    upper = np.triu(np.ones_like(same, dtype=bool), k=1)

    values = similarity[upper]
    is_same = same[upper]
    n_same = int(is_same.sum())

    console.print(
        f"[bold]{len(judged)} piles[/bold] carry enough labelled faces to judge, giving "
        f"{len(values):,} pairs — {n_same} of them genuinely the same person.\n"
    )
    if not n_same:
        console.print(
            "[yellow]No two judged piles hold the same person.[/yellow] Nothing to tune "
            "against: either the clustering split nobody, or too few piles carry labels."
        )
        return 0

    table = Table(title="If suggestions needed this similarity", header_style="bold")
    for column in ("cosine", "suggested", "of those, right", "same-people found", "wasted looks"):
        table.add_column(column, justify="right" if column != "cosine" else "left")

    best: tuple[float, float] = (0.0, 0.0)
    for cut in BANDS:
        offered = values >= cut
        n_offered = int(offered.sum())
        if not n_offered:
            table.add_row(f"{cut:.2f}", "0", "—", "0%", "0")
            continue
        right = int((offered & is_same).sum())
        precision = right / n_offered
        recall = right / n_same
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if f1 > best[1]:
            best = (cut, f1)
        table.add_row(
            f"{cut:.2f}",
            f"{n_offered:,}",
            f"{precision:.0%}",
            f"{recall:.0%}",
            f"{n_offered - right:,}",
        )
    console.print(table)

    current = 0.30
    offered_now = int((values >= current).sum())
    right_now = int(((values >= current) & is_same).sum())
    console.print()
    console.print(
        f"[bold]At the current 0.30:[/bold] {offered_now:,} pairs offered, {right_now} of them "
        f"the same person — "
        f"{'[red]' if offered_now and right_now / offered_now < 0.3 else '[green]'}"
        f"{(right_now / offered_now if offered_now else 0):.0%} useful[/]."
    )
    at_best = int((values >= best[0]).sum())
    direction = (
        "Raising" if best[0] > current else "Lowering" if best[0] < current else "Leaving"
    )
    change = (
        f"cuts the queue from {offered_now:,} pairs to {at_best:,}"
        if at_best < offered_now
        else f"grows the queue from {offered_now:,} pairs to {at_best:,}"
        if at_best > offered_now
        else f"keeps the queue at {at_best:,} pairs"
    )
    console.print(
        f"[bold green]Best balance at cosine {best[0]:.2f}.[/bold green] {direction} the floor "
        f"to there {change}, while finding "
        f"{int(((values >= best[0]) & is_same).sum()) / n_same:.0%} of the splits.\n"
    )
    console.print(
        "[dim]'wasted looks' is pairs you would be shown that are different people. "
        "Read the table for the trade you want: a lower cut finds more splits and costs "
        "more of your attention.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
