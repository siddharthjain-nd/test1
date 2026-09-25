#!/usr/bin/env python3
"""What is actually inside the biggest pile?

diagnose_threshold.py reports how big the largest group gets. It cannot say whether that
group is one very photographed person or a heap of unrelated junk, and the two demand
opposite responses. This looks inside.

Two questions decide it:

1. Do gold-labelled faces inside the pile belong to *one* person or to many? Many means the
   pile has welded people together. This is decisive where labels exist.
2. How does the pile compare to the rest of the library on face size, sharpness and detector
   confidence? A pile made of tiny blurry faces is the classic failure: an embedder maps
   unreadable faces to nearly the same vector, so they all collapse together regardless of
   who they are.

Only dev identities are consulted. Reserved identities are dropped.

Usage
    python scripts/inspect_pile.py --model w600k_r50.onnx --similarity 0.53
    python scripts/inspect_pile.py --model w600k_r50.onnx --similarity 0.53 --contact-sheet
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
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph

from faceindex import cluster, paths, store
from faceindex.eval.goldset import load_gold_set
from faceindex.eval.split import HOLDOUT, load_split

console = Console()

QUALITY_COLUMNS = ("interocular_px", "blur", "det_score", "relative_size")


def describe(values: list[float]) -> tuple[float, float, float]:
    array = np.array([v for v in values if v is not None], dtype=np.float64)
    if not len(array):
        return (float("nan"),) * 3
    return tuple(float(np.percentile(array, q)) for q in (25, 50, 75))  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--similarity",
        required=True,
        help="One value, or several comma-separated. The graph is built once and reused, "
        "so sweeping costs almost nothing over a single value.",
    )
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--top", type=int, default=5, help="How many large piles to list")
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="Write a grid of sample crops from the biggest pile so it can be eyeballed",
    )
    parser.add_argument("--sheet-size", type=int, default=64)
    parser.add_argument("--include-holdout", action="store_true")
    args = parser.parse_args()

    thresholds = sorted((float(t) for t in str(args.similarity).split(",")), reverse=True)

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    with store.open_index(db_path, read_only=True) as conn:
        console.print(f"[bold]Stage 1/3[/bold] loading {args.model}…")
        face_ids, matrix = cluster.load_embeddings(conn, model=args.model)
        if not face_ids:
            console.print(f"[red]No embeddings for {args.model}.[/red]")
            return 1

        gold_path = paths.gold_dir() / "labels.csv"
        split_path = paths.gold_dir() / "split.csv"
        truth: dict[int, str] = {}
        if gold_path.exists() and split_path.exists():
            gold = load_gold_set(gold_path)
            reserved = load_split(split_path).people(HOLDOUT)
            truth = {f: p for f, p in gold.identities.items() if p not in reserved}
            if not args.include_holdout:
                drop = {f for f, p in gold.identities.items() if p in reserved}
                keep = [i for i, f in enumerate(face_ids) if f not in drop]
                face_ids = [face_ids[i] for i in keep]
                matrix = matrix[keep]
                console.print(f"          dropped {len(drop):,} reserved face(s)")

        console.print(f"          {len(face_ids):,} faces\n")

        console.print("[bold]Stage 2/3[/bold] building the neighbour graph once…")
        jobs = args.jobs if args.jobs is not None else cluster.default_jobs()
        graph = kneighbors_graph(
            matrix.astype(np.float32),
            n_neighbors=min(args.neighbors, len(face_ids) - 1),
            mode="distance",
            include_self=False,
            n_jobs=jobs,
        )
        graph.data = 1.0 - (graph.data**2) / 2.0
        graph = graph.maximum(graph.T).tocsr()
        console.print(f"          {graph.nnz:,} edges\n")

        console.print("[bold]Stage 3/3[/bold] reading face quality…")
        rows = {
            int(r["id"]): r
            for r in conn.execute(
                "SELECT id, interocular_px, blur, det_score, relative_size, crop_path FROM faces"
            )
        }

    pool_stats = {
        column: describe([rows[f][column] for f in face_ids if f in rows])
        for column in QUALITY_COLUMNS
    }

    summary = Table(title="The biggest pile at each setting", header_style="bold")
    summary.add_column("cos")
    summary.add_column("faces", justify="right")
    summary.add_column("% pool", justify="right")
    summary.add_column("labelled", justify="right")
    summary.add_column("people in it", justify="right")
    summary.add_column("median eye px", justify="right")
    summary.add_column("reading")

    pool_eye = pool_stats["interocular_px"][1]
    findings: list[tuple[float, str, int, list[int]]] = []

    for threshold in thresholds:
        kept = graph.copy()
        kept.data[kept.data < threshold] = 0.0
        kept.eliminate_zeros()
        _, labels = connected_components(kept, directed=False)
        counts = Counter(labels.tolist())
        top = counts.most_common(1)[0][0]
        members = [f for f, lab in zip(face_ids, labels, strict=True) if lab == top]
        labelled = [f for f in members if f in truth]
        people = Counter(truth[f] for f in labelled)
        median_eye = describe([rows[f]["interocular_px"] for f in members if f in rows])[1]

        if len(people) > 1:
            reading, kind = "[red]several people welded[/red]", "welded"
        elif len(people) == 1 and len(labelled) >= 3:
            reading, kind = "[green]one person[/green]", "person"
        elif pool_eye and median_eye == median_eye and median_eye < 0.6 * pool_eye:
            reading, kind = "[red]junk — unreadable faces[/red]", "junk"
        else:
            reading, kind = "[yellow]unlabelled, can't tell[/yellow]", "unknown"

        findings.append((threshold, kind, len(members), members))
        summary.add_row(
            f"{threshold:.2f}",
            f"{len(members):,}",
            f"{100 * len(members) / len(face_ids):.1f}%",
            f"{len(labelled)}",
            f"{len(people)}",
            f"{median_eye:.3g}",
            reading,
        )

    console.print()
    console.print(summary)

    # Is the pile at the loosest setting simply the strictest one plus accretion, or a
    # different blob? If it is the same core throughout, no threshold in this range touches
    # it and the cause lies elsewhere.
    if len(findings) > 1:
        core = set(findings[0][3])
        widest = set(findings[-1][3])
        shared = len(core & widest) / max(len(core), 1)
        console.print(
            f"\n[dim]{100 * shared:.0f}% of the strictest pile is still inside the loosest "
            f"one — {'the same core throughout' if shared > 0.9 else 'they are different piles'}"
            f".[/dim]"
        )

    strictest_threshold, _kind, strictest_size, strictest_members = findings[0]
    quality = Table(
        title=f"Pile at cos {strictest_threshold:.2f} versus the whole library",
        header_style="bold",
    )
    quality.add_column("measure")
    quality.add_column("pile (25/50/75)", justify="right")
    quality.add_column("library (25/50/75)", justify="right")
    for column in QUALITY_COLUMNS:
        pile_q = describe([rows[f][column] for f in strictest_members if f in rows])
        pool_q = pool_stats[column]
        quality.add_row(
            column,
            " / ".join(f"{v:.3g}" for v in pile_q),
            " / ".join(f"{v:.3g}" for v in pool_q),
        )
    console.print()
    console.print(quality)

    if args.contact_sheet:
        for threshold, _, _, members in (findings[0], findings[-1]) if len(findings) > 1 else findings:
            written = write_sheet(members, rows, args.sheet_size, threshold, 0)
            if written:
                console.print(f"Contact sheet for cos {threshold:.2f}: [bold]{written}[/bold]")

    console.print()
    kinds = {k for _, k, _, _ in findings}
    share = 100 * strictest_size / len(face_ids)

    if "welded" in kinds:
        worst = next(t for t, k, _, _ in findings if k == "welded")
        console.print(
            f"[bold red]VERDICT: broken at cos {worst:.2f} and looser.[/bold red] The biggest "
            f"pile there holds labelled faces from more than one person, so that setting "
            f"merges people. Use a setting above it."
        )
        return 1
    if "junk" in kinds:
        console.print(
            f"[bold red]VERDICT: junk pile.[/bold red] The biggest pile survives even the "
            f"strictest setting tried ({strictest_size:,} faces, {share:.1f}% of the library) "
            f"and its faces are far smaller than a typical face here. These are crops the "
            f"model cannot tell apart, so they collapse together whatever the threshold is. "
            f"Filter them out before clustering; changing the threshold will not help."
        )
        return 1
    if kinds == {"person"}:
        console.print(
            f"[bold green]VERDICT: fine at every setting tried.[/bold green] The biggest pile "
            f"is {strictest_size:,} faces ({share:.1f}%) and every labelled face in it is the "
            f"same person. Large, but legitimately one person."
        )
        return 0
    console.print(
        f"[bold yellow]VERDICT: unclear — the gold set cannot settle this.[/bold yellow] The "
        f"biggest pile is {strictest_size:,} faces ({share:.1f}%) even at the strictest "
        f"setting, but almost none of those faces are labelled. Open the contact sheet and "
        f"look: all one person means the setting is fine, a mixture of different people or a "
        f"wall of blurry crops means it is not."
    )
    return 0


def write_sheet(
    members: list[int], rows: dict[int, object], n: int, similarity: float, pile: int
) -> Path | None:
    import cv2

    paths_found = [
        rows[f]["crop_path"]  # type: ignore[index]
        for f in members
        if f in rows and rows[f]["crop_path"]  # type: ignore[index]
    ]
    if not paths_found:
        console.print("[yellow]No crop paths recorded; cannot build a contact sheet.[/yellow]")
        return None

    rng = np.random.default_rng(0)
    picked = rng.choice(len(paths_found), size=min(n, len(paths_found)), replace=False)
    tiles = []
    for index in picked:
        image = cv2.imread(str(paths_found[int(index)]))
        if image is not None:
            tiles.append(cv2.resize(image, (112, 112)))
    if not tiles:
        console.print("[yellow]No crops could be read; cannot build a contact sheet.[/yellow]")
        return None

    cols = int(np.ceil(np.sqrt(len(tiles))))
    grid_rows = int(np.ceil(len(tiles) / cols))
    sheet = np.zeros((grid_rows * 112, cols * 112, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        r, c = divmod(index, cols)
        sheet[r * 112 : (r + 1) * 112, c * 112 : (c + 1) * 112] = tile

    out = paths.results_dir() / f"pile{pile}_cos{similarity}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
