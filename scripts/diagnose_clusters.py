#!/usr/bin/env python3
"""How tight are the bootstrap clusters, and which ones are junk?

A density-based clusterer does not promise that every face in a cluster resembles every
other one. It promises a connected dense region -- so A joins B, B joins C, C joins D, and
A never gets compared to D at all. Five small steps can cross a lot of ground.

The other half of the story is degenerate embeddings. A 15px face upscaled to 112x112 is
mostly invented pixels, so its vector collapses toward a low-variance region that every
other unreadable face also collapses toward. Identity, age and sex all wash out. That
region is *dense*, which is exactly what HDBSCAN looks for, so it forms confident clusters
whose members share nothing but being unreadable.

This measures both effects: how tight each cluster actually is, and whether looseness
tracks face quality. Loose clusters full of tiny faces confirm the story; loose clusters
of large sharp faces would mean something else is wrong.

Usage
    python scripts/diagnose_clusters.py
    python scripts/diagnose_clusters.py --sampled-only   # only piles you will be shown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from rich.console import Console
from rich.table import Table

from faceindex import embed, paths, store

console = Console()

# Tightness bands. A pair of photographs of one person normally exceeds ~0.5 with this
# model; unrelated faces sit near 0. Anything under 0.3 cannot plausibly be one person.
BANDS = (
    ("tight   >=0.55", 0.55, 2.0, "almost certainly one person"),
    ("fair  0.40-0.55", 0.40, 0.55, "probably one person"),
    ("loose 0.30-0.40", 0.30, 0.40, "doubtful"),
    ("junk    <0.30", -2.0, 0.30, "not one person"),
)


def compare_noise(conn: object) -> None:
    """Are the unclustered faces junk, or perfectly good photographs?

    This decides how much of the gold set they should occupy, and the two answers point in
    opposite directions. If noise is unreadable crops, sampling it heavily wastes labelling
    effort on faces nobody can judge. If noise is clear photographs of people who simply
    appear once or twice, then it is the *hardest and most valuable* material available --
    and leaving it out produces a gold set made only of faces the system already handles,
    which scores well and teaches nothing.
    """
    table = Table(title="Unclustered faces vs clustered faces", header_style="bold")
    table.add_column("Group")
    table.add_column("Faces", justify="right")
    table.add_column("Median size", justify="right")
    table.add_column("Readable >=40px", justify="right")
    table.add_column("Median blur", justify="right")
    table.add_column("Median confidence", justify="right")

    stats = {}
    for name, predicate in (("clustered", "b.cluster_id != -1"), ("noise", "b.cluster_id = -1")):
        rows = conn.execute(  # type: ignore[attr-defined]
            f"SELECT f.interocular_px AS iod, f.blur AS blur, f.det_score AS score "
            f"FROM bootstrap_clusters b JOIN faces f ON f.id = b.face_id WHERE {predicate}"
        ).fetchall()
        if not rows:
            continue
        iod = np.array([float(r["iod"] or 0.0) for r in rows])
        blur = np.array([float(r["blur"] or 0.0) for r in rows])
        score = np.array([float(r["score"]) for r in rows])
        readable = float((iod >= 40).mean())
        stats[name] = (len(rows), float(np.median(iod)), readable)
        table.add_row(
            name,
            f"{len(rows):,}",
            f"{np.median(iod):.0f} px",
            f"{readable:5.1%}",
            f"{np.median(blur):.0f}",
            f"{np.median(score):.3f}",
        )
    console.print(table)

    if "noise" in stats and "clustered" in stats:
        noise_readable = stats["noise"][2]
        clustered_readable = stats["clustered"][2]
        if noise_readable >= clustered_readable * 0.7:
            console.print(
                f"\n[green]The noise bucket is not junk.[/green] {noise_readable:.0%} of "
                f"unclustered faces are 40px or larger, against {clustered_readable:.0%} of "
                f"clustered ones. These are mostly real, readable photographs of people who "
                f"appear too rarely to form a group, or whose photo is atypical for them.\n"
                f"They are the hardest and most valuable material the gold set can hold, and "
                f"cutting them to a token share would leave a test made of faces the system "
                f"already handles."
            )
        else:
            console.print(
                f"\n[yellow]The noise bucket is mostly low quality.[/yellow] Only "
                f"{noise_readable:.0%} of unclustered faces reach 40px, against "
                f"{clustered_readable:.0%} of clustered ones. Sampling it heavily spends "
                f"labelling effort on faces nobody can judge."
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument(
        "--sampled-only",
        action="store_true",
        help="Restrict to clusters present in the gold sample -- the ones you will see.",
    )
    parser.add_argument("--min-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    with store.open_index(db_path, read_only=True) as conn:
        where = ""
        if args.sampled_only:
            where = "AND b.cluster_id IN (SELECT bootstrap_cluster FROM gold_candidates) "

        rows = conn.execute(
            f"""
            SELECT b.cluster_id AS cid, e.embedding AS emb, f.interocular_px AS iod
            FROM bootstrap_clusters b
            JOIN face_embeddings e ON e.face_id = b.face_id
            JOIN faces f ON f.id = b.face_id
            WHERE b.cluster_id != -1 {where}
            ORDER BY b.cluster_id
            """
        ).fetchall()

        baseline_rows = conn.execute(
            "SELECT embedding FROM face_embeddings WHERE embed_version = ? LIMIT 4000",
            (embed.EMBED_VERSION,),
        ).fetchall()

    if not rows:
        console.print("[red]No clusters found. Run bootstrap_cluster.py first.[/red]")
        return 1

    with store.open_index(db_path, read_only=True) as conn:
        compare_noise(conn)
    console.print()

    # ---- baseline: what do two unrelated faces score? ------------------------------
    base = embed.load_matrix([bytes(r["embedding"]) for r in baseline_rows])
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(base), size=(4000, 2))
    pick = pick[pick[:, 0] != pick[:, 1]]
    random_sim = np.sum(base[pick[:, 0]] * base[pick[:, 1]], axis=1)
    console.print(
        f"[bold]Baseline[/bold]  two unrelated faces score "
        f"{float(random_sim.mean()):.3f} on average "
        f"(95th percentile {float(np.percentile(random_sim, 95)):.3f}).\n"
    )

    # ---- per-cluster tightness ------------------------------------------------------
    by_cluster: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        by_cluster.setdefault(int(row["cid"]), []).append(index)

    matrix = embed.load_matrix([bytes(r["emb"]) for r in rows])
    iods = np.array([float(r["iod"] or 0.0) for r in rows])

    stats: list[tuple[int, int, float, float]] = []
    for cid, members in by_cluster.items():
        if len(members) < args.min_size:
            continue
        block = matrix[members]
        sim = block @ block.T
        n = len(members)
        # Mean of the off-diagonal: how alike the members are to each other.
        mean_sim = float((sim.sum() - n) / (n * (n - 1))) if n > 1 else 1.0
        stats.append((cid, n, mean_sim, float(np.median(iods[members]))))

    if not stats:
        console.print("[yellow]No clusters big enough to measure.[/yellow]")
        return 1

    table = Table(title="Cluster tightness", header_style="bold")
    table.add_column("Band")
    table.add_column("Clusters", justify="right")
    table.add_column("Share", justify="right")
    table.add_column("Faces", justify="right")
    table.add_column("Median size", justify="right")
    table.add_column("Reading")

    for name, low, high, reading in BANDS:
        band = [s for s in stats if low <= s[2] < high]
        if not band:
            continue
        table.add_row(
            name,
            f"{len(band):,}",
            f"{100 * len(band) / len(stats):5.1f}%",
            f"{sum(s[1] for s in band):,}",
            f"{np.median([s[3] for s in band]):.0f} px",
            reading,
        )
    console.print(table)

    # ---- does looseness track cluster size? ----------------------------------------
    size_table = Table(title="Tightness by how many faces the cluster holds", header_style="bold")
    size_table.add_column("Cluster size")
    size_table.add_column("Clusters", justify="right")
    size_table.add_column("Mean tightness", justify="right")
    size_table.add_column("Junk share", justify="right")

    for label, low, high in (("2-4", 2, 5), ("5-9", 5, 10), ("10-29", 10, 30), ("30+", 30, 10**9)):
        band = [s for s in stats if low <= s[1] < high]
        if not band:
            continue
        junk = sum(1 for s in band if s[2] < 0.30) / len(band)
        colour = "red" if junk > 0.5 else "yellow" if junk > 0.2 else "green"
        size_table.add_row(
            label,
            f"{len(band):,}",
            f"{np.mean([s[2] for s in band]):.3f}",
            f"[{colour}]{junk:5.1%}[/{colour}]",
        )
    console.print(size_table)

    junk_total = [s for s in stats if s[2] < 0.30]
    tight_total = [s for s in stats if s[2] >= 0.55]
    console.print(
        f"\n[bold]{len(tight_total):,}[/bold] clusters are tight enough to accept in one "
        f"keystroke; [bold]{len(junk_total):,}[/bold] are below the level two photographs of "
        f"one person reach, and will need judging face by face."
    )

    if junk_total:
        junk_px = np.median([s[3] for s in junk_total])
        tight_px = np.median([s[3] for s in tight_total]) if tight_total else float("nan")
        console.print(
            f"Median face size: [bold]{junk_px:.0f}px[/bold] in the junk clusters vs "
            f"[bold]{tight_px:.0f}px[/bold] in the tight ones."
        )
        if junk_px < tight_px * 0.75:
            console.print(
                "\n[green]That gap is the explanation.[/green] The loose clusters are made of "
                "small faces, whose embeddings carry almost no identity and collapse into one "
                "dense region. HDBSCAN groups that region because it is dense, not because the "
                "faces match. Nothing is broken — it is the bootstrap doing what a weak model "
                "on unreadable faces must do, and it is why Phase 3 gates on face size."
            )
        else:
            console.print(
                "\n[yellow]Face size does not explain the loose clusters.[/yellow] Worth "
                "checking alignment on a sample before trusting the pool."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
