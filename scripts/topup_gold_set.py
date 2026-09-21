#!/usr/bin/env python3
"""Add more faces of people you have already named, from the era they are missing.

The cross-era requirement is the one thing the sample cannot arrange for itself. A person
photographed at 20 and at 35 is the hardest case the system faces, and their old and new
photos land in different clusters precisely *because* of that difficulty -- so no amount of
resampling reliably finds them. It only becomes visible once a human has put one name on
both, which is after labelling, not before.

So this works backwards from the finished labels. For each person, it takes the faces you
named, builds an average of them, and searches the unlabelled pool for faces that look like
that person **in the era they are currently missing**. Those become new candidates for you
to confirm or reject.

It never assigns a label. Every face it finds still has to be judged by eye, because a
similarity score is a suggestion and the gold set is ground truth.

Usage
    python scripts/topup_gold_set.py                    # show what it would add
    python scripts/topup_gold_set.py --write            # add them to the queue
    python scripts/topup_gold_set.py --per-person 12 --min-similarity 0.35
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from rich.console import Console
from rich.table import Table

from faceindex import embed, paths, sampling, store

console = Console()

ERAS = ("oldest", "recent")


def browse_era(
    db_path: Path,
    candidates: list[sampling.Candidate],
    *,
    era: str,
    in_queue: set[int],
    piles: int,
    per_pile: int,
    write: bool,
) -> int:
    """Queue the biggest unlabelled piles from one era, so they can be recognised by eye.

    Searching outward from a person fails for exactly the people worth finding: their old
    photos look different from the average of their recent ones, which is what cross-age
    drift *means*. Browsing avoids the problem entirely -- a pile of one person from 2010 is
    obvious to someone who knows them, whatever a similarity score thinks. Typing a name
    that already exists is what creates the cross-era link.
    """
    by_cluster: dict[int, list[sampling.Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.cluster_id != -1 and candidate.face_id not in in_queue:
            by_cluster[candidate.cluster_id].append(candidate)

    # A pile counts as belonging to this era when most of it was taken then.
    scored: list[tuple[int, int, float, list[sampling.Candidate]]] = []
    for cluster_id, members in by_cluster.items():
        in_era = [c for c in members if c.strata["era"] == era]
        share = len(in_era) / len(members)
        if share >= 0.5 and len(in_era) >= 2:
            scored.append((cluster_id, len(in_era), share, in_era))

    if not scored:
        console.print(f"[yellow]No unqueued piles are mostly from the {era} era.[/yellow]")
        return 0

    scored.sort(key=lambda s: -s[1])
    chosen = scored[:piles]

    table = Table(title=f"Biggest unlabelled piles from the {era} era", header_style="bold")
    table.add_column("Pile", justify="right")
    table.add_column("Faces in era", justify="right")
    table.add_column("Share of pile", justify="right")
    for cluster_id, n, share, _ in chosen[:15]:
        table.add_row(str(cluster_id), str(n), f"{share:.0%}")
    console.print(table)

    queued = [(c, cid) for cid, _n, _s, members in chosen for c in members[:per_pile]]
    console.print(
        f"\n{len(queued):,} faces across {len(chosen)} piles.\n"
        f"[dim]In the labelling UI these arrive as normal piles. When you recognise someone "
        f"you have already named, type that same name — that is what makes them a cross-era "
        f"identity. Anyone new, or anyone you cannot place, gets n or u as usual.[/dim]"
    )

    if not write:
        console.print("\nRe-run with [bold]--write[/bold] to add them.")
        return 0

    now = datetime.now(UTC).isoformat()
    run_id = datetime.now(UTC).strftime(f"browse-{era}-%Y%m%dT%H%M%SZ")
    with store.open_index(db_path) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO gold_candidates "
            "(face_id, stratum, bootstrap_cluster, reserved_for, sample_run, created_at) "
            "VALUES (?,?,?,?,?,?)",
            [
                (
                    c.face_id,
                    json.dumps(c.strata, sort_keys=True),
                    cid,
                    f"cross_era_browse:{era}",
                    run_id,
                    now,
                )
                for c, cid in queued
            ],
        )
        conn.commit()

    console.print(
        f"\n[green]Added {len(queued):,} candidates.[/green] Run: python scripts/label_gold_set.py"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--beauty-marker", default="You cam perfect")
    parser.add_argument(
        "--per-person", type=int, default=10, help="Most faces to add per person per era"
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.35,
        help="Deliberately generous. You confirm every one by eye, and cross-age pairs score "
        "low by their very nature -- a strict cut-off would exclude exactly what is wanted.",
    )
    parser.add_argument(
        "--browse-era",
        choices=ERAS,
        help="Instead of searching from each person, queue the biggest unlabelled piles from "
        "this era so you can recognise who they are.",
    )
    parser.add_argument("--piles", type=int, default=25, help="How many piles to queue")
    parser.add_argument("--write", action="store_true", help="Add them to the labelling queue")
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    with store.open_index(db_path) as conn:
        console.print("Loading…")
        candidates = sampling.load_candidates(conn, beauty_marker=args.beauty_marker)
        if not candidates:
            console.print("[red]No embedded faces found.[/red]")
            return 1
        era_of = {c.face_id: c.strata["era"] for c in candidates}
        strata_of = {c.face_id: c.strata for c in candidates}
        cluster_of = {c.face_id: c.cluster_id for c in candidates}

        labelled = conn.execute(
            "SELECT face_id, person_id FROM gold_labels WHERE label = 'person' "
            "AND person_id IS NOT NULL"
        ).fetchall()
        if not labelled:
            console.print("[red]Nobody named yet. Label first.[/red]")
            return 1

        in_queue = {int(r["face_id"]) for r in conn.execute("SELECT face_id FROM gold_candidates")}

        rows = conn.execute(
            "SELECT face_id, embedding FROM face_embeddings WHERE embed_version = ? "
            "ORDER BY face_id",
            (embed.EMBED_VERSION,),
        ).fetchall()

    face_ids = np.array([int(r["face_id"]) for r in rows])
    matrix = embed.load_matrix([bytes(r["embedding"]) for r in rows])
    position = {int(f): i for i, f in enumerate(face_ids)}

    if args.browse_era:
        return browse_era(
            db_path,
            candidates,
            era=args.browse_era,
            in_queue=in_queue,
            piles=args.piles,
            per_pile=args.per_person,
            write=args.write,
        )

    # ---- who is missing which era? --------------------------------------------------
    faces_by_person: dict[str, list[int]] = defaultdict(list)
    for row in labelled:
        faces_by_person[str(row["person_id"])].append(int(row["face_id"]))

    eras_by_person = {
        person: {era_of.get(f, "?") for f in faces} for person, faces in faces_by_person.items()
    }
    covered = [p for p, seen in eras_by_person.items() if {"oldest", "recent"} <= seen]

    console.print(
        f"[bold]{len(faces_by_person)}[/bold] people named · "
        f"[bold]{len(covered)}[/bold] already span both eras (PLAN.md wants >=10)\n"
    )

    # Only bother with people who have a real chance: they need enough labelled faces to
    # build a stable average, and they must be missing exactly one of the two eras.
    wanted: list[tuple[str, str]] = []
    for person, faces in faces_by_person.items():
        if len(faces) < 2:
            continue
        seen = eras_by_person[person]
        for era in ERAS:
            if era not in seen and seen & set(ERAS):
                wanted.append((person, era))

    if not wanted:
        console.print(
            "[yellow]Nobody is one era short of spanning.[/yellow] Either everyone already "
            "spans, or the people you named appear only in the middle years."
        )
        return 0

    # ---- search the pool ------------------------------------------------------------
    era_index = {
        era: np.array([position[f] for f in face_ids.tolist() if era_of.get(f) == era])
        for era in ERAS
    }

    found: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for person, era in wanted:
        members = [position[f] for f in faces_by_person[person] if f in position]
        if not members:
            continue
        centroid = embed.l2_normalise(matrix[members].mean(axis=0, keepdims=True))[0]

        pool_index = era_index[era]
        if not len(pool_index):
            continue
        sims = matrix[pool_index] @ centroid

        order = np.argsort(-sims)
        picks: list[tuple[int, float]] = []
        for j in order:
            if sims[j] < args.min_similarity:
                break
            fid = int(face_ids[pool_index[j]])
            if fid in in_queue:
                continue
            picks.append((fid, float(sims[j])))
            if len(picks) >= args.per_person:
                break
        if picks:
            found[(person, era)] = picks

    if not found:
        console.print(
            f"[yellow]No plausible matches above {args.min_similarity:.2f}.[/yellow]\n"
            f"That is itself a finding: your library may genuinely hold few people "
            f"photographed across both eras. If so, lower the >=10 requirement in PLAN.md "
            f"to match reality and record why -- do not manufacture pairs that are not there."
        )
        return 0

    table = Table(title="Candidates to confirm", header_style="bold")
    table.add_column("Person")
    table.add_column("Missing era")
    table.add_column("Found", justify="right")
    table.add_column("Best match", justify="right")
    for (person, era), picks in sorted(found.items(), key=lambda kv: -kv[1][0][1]):
        table.add_row(person, era, str(len(picks)), f"{picks[0][1]:.3f}")
    console.print(table)

    total = sum(len(p) for p in found.values())
    console.print(
        f"\n{total:,} faces across {len(found)} person/era pairs. If even 4 of them are "
        f"confirmed, you clear the >=10 requirement."
    )
    console.print(
        "[dim]These are suggestions from similarity alone. Expect some to be other people — "
        "reject those. A wrong confirmation here is worse than a missing one.[/dim]"
    )

    if not args.write:
        console.print("\nRe-run with [bold]--write[/bold] to add them to the labelling queue.")
        return 0

    now = datetime.now(UTC).isoformat()
    run_id = datetime.now(UTC).strftime("topup-%Y%m%dT%H%M%SZ")
    with store.open_index(db_path) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO gold_candidates "
            "(face_id, stratum, bootstrap_cluster, reserved_for, sample_run, created_at) "
            "VALUES (?,?,?,?,?,?)",
            [
                (
                    fid,
                    json.dumps(strata_of.get(fid, {}), sort_keys=True),
                    cluster_of.get(fid, -1),
                    f"cross_era:{person}",
                    run_id,
                    now,
                )
                for (person, _era), picks in found.items()
                for fid, _sim in picks
            ],
        )
        conn.commit()

    console.print(
        f"\n[green]Added {total:,} candidates.[/green] Run: python scripts/label_gold_set.py"
    )
    console.print(
        "They appear in the leftovers view. Name the ones that really are that person, "
        "and press [bold]n[/bold] or [bold]u[/bold] on the rest."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
