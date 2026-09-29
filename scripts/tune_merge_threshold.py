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

from faceindex import embed, paths, review, store
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
        "--method",
        default="both",
        choices=["centroid", "closest", "both"],
        help="How two piles are compared. 'centroid' is the average face of each (what the "
        "merge screen uses today). 'closest' is the single most similar pair of faces "
        "between them, which averaging destroys.",
    )
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

    same = np.array([[identity[a] == identity[b] for b in judged] for a in judged])
    upper = np.triu(np.ones_like(same, dtype=bool), k=1)
    is_same = same[upper]
    n_same = int(is_same.sum())

    console.print(
        f"[bold]{len(judged)} piles[/bold] carry enough labelled faces to judge, giving "
        f"{int(upper.sum()):,} pairs \u2014 {n_same} of them genuinely the same person.\n"
    )
    if not n_same:
        console.print(
            "[yellow]No two judged piles hold the same person.[/yellow] Nothing to tune "
            "against: either the clustering split nobody, or too few piles carry labels."
        )
        return 0

    scores: dict[str, np.ndarray] = {}
    if args.method in ("centroid", "both"):
        block = matrix[[index[p] for p in judged]]
        scores["average face"] = (block @ block.T)[upper]

    if args.method in ("closest", "both"):
        console.print("[dim]Loading face vectors for the closest-pair comparison\u2026[/dim]")
        members: dict[int, list[int]] = {}
        for row in rows:
            pile_id = int(row["pile_id"])
            if pile_id in identity:
                members.setdefault(pile_id, []).append(int(row["face_id"]))
        wanted = sorted({f for p in judged for f in members.get(p, [])})
        with store.open_index(db_path, read_only=True) as conn:
            marks = ",".join("?" * len(wanted))
            found = conn.execute(
                f"SELECT face_id, embedding FROM face_embeddings WHERE face_id IN ({marks}) "
                "AND model = (SELECT model FROM review_runs WHERE run_id = ?) ORDER BY face_id",
                (*wanted, run_id),
            ).fetchall()
        order = {int(r["face_id"]): i for i, r in enumerate(found)}
        faces = embed.l2_normalise(
            embed.load_matrix([bytes(r["embedding"]) for r in found]).astype(np.float32)
        )
        console.print(f"[dim]  {len(found):,} faces across {len(judged)} piles[/dim]\n")

        closest = np.zeros((len(judged), len(judged)), dtype=np.float32)
        blocks = [
            faces[[order[f] for f in members.get(p, []) if f in order]] for p in judged
        ]
        for i in range(len(judged)):
            for j in range(i + 1, len(judged)):
                if len(blocks[i]) and len(blocks[j]):
                    closest[i, j] = closest[j, i] = float((blocks[i] @ blocks[j].T).max())
        scores["closest pair"] = closest[upper]

    recommended: dict[str, tuple[float, float, int, int]] = {}
    for name, values in scores.items():
        table = Table(
            title=f"If suggestions compared the {name}", header_style="bold"
        )
        for column in ("score", "questions", "right", "splits found", "wasted"):
            table.add_column(column, justify="right" if column != "score" else "left")

        best: tuple[float, float] = (0.0, -1.0)
        for cut in BANDS:
            offered = values >= cut
            n_offered = int(offered.sum())
            if not n_offered:
                table.add_row(f"{cut:.2f}", "0", "\u2014", "0%", "0")
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

        # Cheapest row that still finds something: fewest questions answered per merge
        # gained. Percentages mislead here -- two questions for one merge beats forty-one
        # for five, even though the percentage is similar.
        cheapest: tuple[float, float, int, int] | None = None
        for cut in BANDS:
            offered = values >= cut
            n_offered = int(offered.sum())
            right = int((offered & is_same).sum())
            if right == 0:
                continue
            cost = n_offered / right
            if cheapest is None or cost < cheapest[1]:
                cheapest = (cut, cost, n_offered, right)
        if cheapest is not None:
            recommended[name] = cheapest

    console.print()
    for name, (cut, cost, offered, right) in recommended.items():
        verdict = (
            "[green]cheap \u2014 keep it[/green]"
            if cost <= 4
            else "[yellow]tolerable[/yellow]"
            if cost <= 10
            else "[red]too costly[/red]"
        )
        console.print(
            f"[bold]{name}[/bold]: best at {cut:.2f} \u2014 {offered} questions for {right} "
            f"merge(s), so [bold]{cost:.1f} questions per merge[/bold] \u2014 {verdict}"
        )

    winner = min(recommended.items(), key=lambda kv: kv[1][1], default=None)
    console.print()
    if winner is None:
        console.print(
            "[bold red]VERDICT: nothing to find.[/bold red] No setting of either comparison "
            "turns up a single real merge, so the screen has no work to do here."
        )
    elif winner[1][1] > 10:
        console.print(
            f"[bold red]VERDICT: drop the merge screen.[/bold red] The best on offer is "
            f"{winner[1][1]:.0f} questions per merge gained ({winner[0]} at "
            f"{winner[1][0]:.2f}). That is not worth a human's attention."
        )
    else:
        console.print(
            f"[bold green]VERDICT: use the {winner[0]} at {winner[1][0]:.2f}.[/bold green] "
            f"About {winner[1][1]:.0f} question(s) per merge gained \u2014 "
            f"{winner[1][2]} questions in the labelled sample, {winner[1][3]} of them real."
        )
        console.print(
            "[dim]It will not find every split: a strict setting trades coverage for your "
            "time. The ones it misses stay as two piles, which you can still merge by hand "
            "when you notice.[/dim]"
        )

    console.print(
        "[dim]'wasted looks' is pairs you would be shown that are different people. "
        "Read the table for the trade you want: a lower cut finds more splits and costs "
        "more of your attention.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
