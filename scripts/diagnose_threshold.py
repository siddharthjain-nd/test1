#!/usr/bin/env python3
"""Where does connected-components chaining start on the *full* face pool?

The gold set is ~1,700 faces; the library is ~64,000. Connected components is
single-linkage, so ONE bad edge fuses two whole people. The number of chances for
that edge grows with the number of pairs, not the number of faces -- 38x the faces
is roughly 1,400x the pairs. So the threshold that peaked on the gold set may sit on
the wrong side of the cliff at full scale.

Welding needs no labels to detect, and it must be detected *relatively*. An absolute
rule ("a pile over 5% of the library is bad") does not transfer between galleries: in
a family album the most-photographed person legitimately owns a large share, while in
a 500,000-photo archive they own almost none. What does transfer is the *jump*. Loosen
the setting one notch and a real person's pile grows a little; the moment two people
weld, it doubles or worse. The jump is the signal, not the size.

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
# A "biggest pile" smaller than this means nothing has really grouped yet, so growth
# ratios against it are meaningless.
MIN_MEANINGFUL_PILE = 5


def row_largest(rows: list[dict[str, object]], index: int) -> int:
    return int(rows[index]["largest"])


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

    # Strictest first. Loosening one notch should grow a real person's pile gently; a
    # sudden multiplication is two piles becoming one.
    rows: list[dict[str, object]] = []
    for i, threshold in enumerate(reversed(thresholds), start=1):
        step = time.perf_counter()
        kept = graph.copy()
        kept.data[kept.data < threshold] = 0.0
        kept.eliminate_zeros()
        _, labels = connected_components(kept, directed=False)
        counts = np.sort(np.bincount(labels))[::-1]
        rows.append(
            {
                "threshold": threshold,
                "clusters": len(counts),
                "largest": int(counts[0]),
                "grouped": int(n - (counts == 1).sum()),
            }
        )
        console.print(
            f"  [{i}/{len(thresholds)}] cos {threshold:.2f}: largest pile "
            f"{int(counts[0]):,}  [dim]{time.perf_counter() - step:.0f}s[/dim]"
        )

    table = Table(
        title=f"Connected components across the whole pool — {model}", header_style="bold"
    )
    table.add_column("cos")
    table.add_column("piles", justify="right")
    table.add_column("biggest pile", justify="right")
    table.add_column("grew by", justify="right")
    table.add_column("reading")

    # One rule: a person's pile has a ceiling -- they only own so many photos -- so as the
    # setting loosens the biggest pile grows, then plateaus. The moment two people weld it
    # jumps. The single largest jump is therefore the weld, and the setting one notch
    # stricter is the loosest safe choice. Reading the biggest jump rather than the first
    # one over some fixed factor keeps this free of a magic number.
    factors: list[float | None] = [None]
    for index in range(1, len(rows)):
        previous = int(rows[index - 1]["largest"])
        factors.append(
            int(row_largest(rows, index)) / previous
            if previous >= MIN_MEANINGFUL_PILE
            else None
        )

    comparable = [(f, i) for i, f in enumerate(factors) if f is not None]
    weld_index = max(comparable)[1] if comparable and max(comparable)[0] >= 2.0 else None

    for index, row in enumerate(rows):
        factor = factors[index]
        if factor is None:
            growth, reading = ("", "[dim]strictest tried[/dim]" if not index else "")
            if index:
                growth, reading = (
                    f"x{int(row['largest']) / max(int(rows[index-1]['largest']), 1):.1f}",
                    "[dim]too strict to judge[/dim]",
                )
        elif index == weld_index:
            growth, reading = f"x{factor:.1f}", "[red]<-- welded here[/red]"
        elif factor >= 1.3:
            growth, reading = f"x{factor:.1f}", "[yellow]growing[/yellow]"
        else:
            growth, reading = f"x{factor:.1f}", "[green]steady[/green]"
        table.add_row(
            f"{float(row['threshold']):.2f}",
            f"{int(row['clusters']):,}",
            f"{int(row['largest']):,}",
            growth,
            reading,
        )

    console.print()
    console.print(table)

    if weld_index is None:
        console.print(
            "\n[yellow]No weld found in this range.[/yellow] Every step grew gently, so the "
            "edge is looser than anything tried. Re-run reaching lower, e.g. "
            "--thresholds 0.35,0.38,0.41,0.44,0.47"
        )
        return 0

    safe = float(rows[weld_index - 1]["threshold"])
    console.print(
        f"\n[bold]Weld at cos {float(rows[weld_index]['threshold']):.2f} — "
        f"biggest pile jumped x{factors[weld_index]:.1f}.[/bold]"
    )
    console.print(
        f"[bold green]Loosest safe setting: cos {safe:.2f}[/bold green]\n"
        "[dim]Looser pulls more of each person into one pile, so the loosest value that has "
        "not yet welded two people together is the best trade on offer.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
