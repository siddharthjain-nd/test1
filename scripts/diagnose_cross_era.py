#!/usr/bin/env python3
"""Is there enough cross-era material in the pool, and can clustering see it?

The sampler asserts that >=10 bootstrap clusters span the oldest and newest eras. That
assertion may be unsatisfiable by construction: cross-age drift is exactly what stops one
person's old and new photos from clustering together, so requiring a cluster to span eras
asks the bootstrap to have already solved the problem the gold set exists to measure.

This separates the two possibilities:

  supply  -- how many people plausibly appear in both eras at all
  sight   -- how many of those the bootstrap clustering can actually see

Cluster membership answers "sight". Nearest-neighbour similarity across the era boundary
answers "supply", because two photos of one person 14 years apart may sit well below the
clustering threshold while still being far more similar than two strangers.

Usage
    python scripts/diagnose_cross_era.py
    python scripts/diagnose_cross_era.py --sample 8000   # cap the cost of the NN search
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from rich.console import Console
from rich.table import Table

from faceindex import embed, paths, sampling, store

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--beauty-marker", default="You cam perfect")
    parser.add_argument(
        "--sample",
        type=int,
        default=8000,
        help="Max faces per era for the neighbour search. Keeps peak memory bounded.",
    )
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    with store.open_index(db_path, read_only=True) as conn:
        console.print("Loading candidates…")
        candidates = sampling.load_candidates(conn, beauty_marker=args.beauty_marker)
        if not candidates:
            console.print("[red]No embedded faces found.[/red]")
            return 1

        by_id = {c.face_id: c for c in candidates}

        # ---- Era supply ------------------------------------------------------------
        eras = Table(title="Faces per era in the whole pool", header_style="bold")
        eras.add_column("Era")
        eras.add_column("Faces", justify="right")
        eras.add_column("Share", justify="right")
        counts: dict[str, int] = {}
        for candidate in candidates:
            era = candidate.strata["era"]
            counts[era] = counts.get(era, 0) + 1
        for era in ("oldest", "middle", "recent", "undated"):
            n = counts.get(era, 0)
            eras.add_row(era, f"{n:,}", f"{100 * n / len(candidates):5.1f}%")
        console.print(eras)

        # ---- What clustering can see ------------------------------------------------
        spanning = {c.cluster_id for c in candidates if c.cross_era and c.cluster_id != -1}
        clustered = {c.cluster_id for c in candidates if c.cluster_id != -1}
        console.print(
            f"\n[bold]Sight[/bold]  bootstrap clusters spanning oldest+recent: "
            f"[bold]{len(spanning):,}[/bold] of {len(clustered):,} clusters "
            f"({100 * len(spanning) / max(1, len(clustered)):.1f}%)"
        )

        # ---- What actually exists ---------------------------------------------------
        rows = conn.execute(
            "SELECT face_id, embedding FROM face_embeddings WHERE embed_version = ? "
            "ORDER BY face_id",
            (embed.EMBED_VERSION,),
        ).fetchall()

    face_ids = np.array([int(r["face_id"]) for r in rows])
    matrix = embed.load_matrix([bytes(r["embedding"]) for r in rows])

    era_of = np.array([by_id[int(f)].strata["era"] if int(f) in by_id else "?" for f in face_ids])
    rng = np.random.default_rng(args.seed)

    def take(era: str) -> np.ndarray:
        index = np.flatnonzero(era_of == era)
        if len(index) > args.sample:
            index = np.sort(rng.choice(index, args.sample, replace=False))
        return index

    old_index, new_index = take("oldest"), take("recent")
    if not len(old_index) or not len(new_index):
        console.print("[yellow]One of the eras is empty; nothing to compare.[/yellow]")
        return 1

    console.print(
        f"\nSearching {len(old_index):,} oldest-era faces against {len(new_index):,} "
        f"recent-era faces for their best match…"
    )

    new_matrix = matrix[new_index]
    best = np.full(len(old_index), -1.0, dtype=np.float32)
    # Chunked so the similarity block never exceeds a few hundred MB on the 8 GB box.
    for start in range(0, len(old_index), 512):
        block = matrix[old_index[start : start + 512]]
        best[start : start + 512] = (block @ new_matrix.T).max(axis=1)

    console.print(
        "\n[bold]Supply[/bold]  best cross-era similarity per oldest-era face.\n"
        "Random strangers sit near 0.0; the same person usually exceeds ~0.4 even years apart."
    )
    supply = Table(header_style="bold")
    supply.add_column("Similarity >=", justify="right")
    supply.add_column("Oldest-era faces with a match", justify="right")
    supply.add_column("Share", justify="right")
    for threshold in (0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7):
        n = int((best >= threshold).sum())
        supply.add_row(f"{threshold:.2f}", f"{n:,}", f"{100 * n / len(best):5.1f}%")
    console.print(supply)

    console.print(
        f"\nmedian best match {float(np.median(best)):.3f} · "
        f"90th percentile {float(np.percentile(best, 90)):.3f} · "
        f"max {float(best.max()):.3f}"
    )

    strong = int((best >= 0.4).sum())
    if strong >= 200:
        console.print(
            f"\n[green]Cross-era material exists.[/green] {strong:,} oldest-era faces have a "
            f"plausible recent-era counterpart, but clustering only sees {len(spanning)} spanning "
            f"clusters. The shortfall is the bootstrap's blindness to cross-age drift, not the "
            f"library's. Requiring spanning clusters at sampling time asks the clusterer to have "
            f"already solved the problem the gold set exists to measure -- move the check to "
            f"export, where human labels make it meaningful."
        )
    else:
        console.print(
            f"\n[yellow]Little cross-era material: only {strong:,} oldest-era faces have a "
            f"plausible recent counterpart.[/yellow] If that holds up, the library genuinely has "
            f"few people spanning both eras, and the >=10 requirement should be lowered to match "
            f"reality rather than forced."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
