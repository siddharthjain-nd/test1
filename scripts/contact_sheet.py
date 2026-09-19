#!/usr/bin/env python3
"""Render a grid of face crops to a single image, for eyeballing the pool.

Statistics say 34% of detected faces are tiny. Only looking at them reveals whether they
are genuine small faces in crowd shots or detector false positives, and the answer changes
the quality-gating thresholds and the gold-set strata.

It also answers the other question statistics cannot: **did the clustering work?** One row
per cluster shows at a glance whether each pile is one person or a mixture, which is the
only honest check available before any labelling has happened.

Usage
    python scripts/contact_sheet.py --bucket tiny --out /tmp/tiny.jpg
    python scripts/contact_sheet.py --bucket profile --context
    python scripts/contact_sheet.py --min-iod 0 --max-iod 12 --count 144

    python scripts/contact_sheet.py --cluster-overview   # one row per big cluster
    python scripts/contact_sheet.py --cluster 2398       # one cluster in detail
    python scripts/contact_sheet.py --noise              # what failed to cluster
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw
from rich.console import Console

from faceindex import facepool, paths, store

console = Console()

# name -> (min_iod, max_iod, min_abs_yaw, max_abs_yaw)
BUCKETS: dict[str, tuple[float, float, float, float]] = {
    "tiny": (0.0, 20.0, 0.0, 91.0),
    "small": (20.0, 40.0, 0.0, 91.0),
    "medium": (40.0, 80.0, 0.0, 91.0),
    "large": (80.0, 1e9, 0.0, 91.0),
    "frontal": (0.0, 1e9, 0.0, 15.0),
    "profile": (0.0, 1e9, 45.0, 91.0),
    "blurry": (0.0, 1e9, 0.0, 91.0),
    "all": (0.0, 1e9, 0.0, 91.0),
}


def render(
    rows: list[object],
    *,
    out: Path,
    columns: int,
    cell: int,
    context: bool,
    label: str = "size",
) -> None:
    """Paste crops into a grid, captioned so the numbers behind each face are visible."""
    label_height = 14
    row_count = (len(rows) + columns - 1) // columns

    sheet = Image.new("RGB", (columns * cell, row_count * (cell + label_height)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(rows):
        source = row["context_path"] if context else row["crop_path"]  # type: ignore[index]
        x = (index % columns) * cell
        y = (index // columns) * (cell + label_height)
        try:
            with Image.open(str(source)) as image:
                sheet.paste(image.convert("RGB").resize((cell, cell)), (x, y))
        except (OSError, ValueError):
            draw.rectangle([x, y, x + cell, y + cell], fill=(80, 0, 0))

        if label == "date":
            # When judging a cluster, *when* the photo was taken matters far more than how
            # many pixels wide the face is: one person across years is the hard case.
            caption = str(row["taken_at"] or "undated")[:7]  # type: ignore[index]
        else:
            caption = (
                f"{row['interocular_px']:.0f}px "  # type: ignore[index]
                f"y{row['yaw_deg']:+.0f} b{row['blur']:.0f}"  # type: ignore[index]
            )
        draw.text((x + 2, y + cell + 1), caption, fill=(200, 200, 200))

    sheet.save(out, "JPEG", quality=88)


def cluster_overview(conn: object, args: argparse.Namespace) -> int:
    """One row per cluster, largest first. The fastest honest read on clustering quality."""
    top = [
        int(r["cluster_id"])  # type: ignore[index]
        for r in conn.execute(  # type: ignore[attr-defined]
            "SELECT cluster_id, COUNT(*) AS n FROM bootstrap_clusters "
            "WHERE cluster_id != -1 GROUP BY cluster_id ORDER BY n DESC LIMIT ?",
            (args.clusters,),
        )
    ]
    if not top:
        console.print("[yellow]No clusters. Run scripts/bootstrap_cluster.py first.[/yellow]")
        return 1

    picked: list[object] = []
    for cluster_id in top:
        rows = conn.execute(  # type: ignore[attr-defined]
            "SELECT f.crop_path, f.context_path, f.interocular_px, f.yaw_deg, f.blur, "
            "p.taken_at FROM bootstrap_clusters b JOIN faces f ON f.id = b.face_id "
            "JOIN photos p ON p.id = f.photo_id WHERE b.cluster_id = ? "
            "ORDER BY p.taken_at",
            (cluster_id,),
        ).fetchall()
        # Spread the picks across the cluster's whole time range rather than taking the
        # first N, so a cluster that drifts across years reveals it.
        if len(rows) > args.columns:
            step = len(rows) / args.columns
            rows = [rows[int(i * step)] for i in range(args.columns)]
        picked.extend(rows)
        picked.extend([None] * (args.columns - len(rows)))

    usable = [r for r in picked if r is not None]
    render(
        usable,
        out=args.out,
        columns=args.columns,
        cell=args.cell,
        context=args.context,
        label="date",
    )
    console.print(f"[green]Wrote {args.out}[/green] — {len(top)} clusters, one per row.")
    console.print("Captions are capture dates. Each row should be ONE person.")
    console.print(
        "\n[bold]What to look for[/bold]\n"
        "  • A row that is clearly one person       -> clustering is working\n"
        "  • A row mixing two or more people        -> clusters are over-merged\n"
        "  • The same person across several rows    -> expected; 4,023 clusters for ~40 people\n"
        "    means people are split, which is normal for a bootstrap and is what labelling fixes\n"
        "  • Dates spanning years within one row    -> a genuine cross-era identity, the most\n"
        "    valuable thing the gold set can contain"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--bucket", choices=sorted(BUCKETS), default="tiny")
    parser.add_argument("--cluster", type=int, default=None, help="Render one bootstrap cluster")
    parser.add_argument(
        "--cluster-overview", action="store_true", help="One row per cluster, largest first"
    )
    parser.add_argument("--clusters", type=int, default=12, help="Rows in the overview")
    parser.add_argument("--noise", action="store_true", help="Faces that failed to cluster")
    parser.add_argument("--min-iod", type=float, default=None)
    parser.add_argument("--max-iod", type=float, default=None)
    parser.add_argument("--kind", default=None, help="Restrict to a photo kind, e.g. forwarded")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument("--cell", type=int, default=112)
    parser.add_argument("--context", action="store_true", help="Use the wider labelling crops")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("contact_sheet.jpg"))
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    with store.open_index(db_path, read_only=True) as conn:
        if args.cluster_overview:
            return cluster_overview(conn, args)

        base = (
            "SELECT f.crop_path, f.context_path, f.interocular_px, f.yaw_deg, f.blur, "
            "f.det_score, p.kind, p.taken_at FROM faces f JOIN photos p ON p.id = f.photo_id "
        )

        if args.cluster is not None or args.noise:
            cluster_id = -1 if args.noise else args.cluster
            sql = (
                base + "JOIN bootstrap_clusters b ON b.face_id = f.id "
                "WHERE b.cluster_id = ? ORDER BY p.taken_at"
            )
            rows = conn.execute(sql, (cluster_id,)).fetchall()
            label = "date"
            described = "noise bucket" if args.noise else f"cluster {args.cluster}"
        else:
            min_iod, max_iod, min_yaw, max_yaw = BUCKETS[args.bucket]
            if args.min_iod is not None:
                min_iod = args.min_iod
            if args.max_iod is not None:
                max_iod = args.max_iod

            sql = (
                base + "WHERE f.pool_version = ? AND f.interocular_px >= ? "
                "AND f.interocular_px < ? AND ABS(f.yaw_deg) >= ? AND ABS(f.yaw_deg) < ?"
            )
            params: list[object] = [facepool.POOL_VERSION, min_iod, max_iod, min_yaw, max_yaw]
            if args.kind:
                sql += " AND p.kind = ?"
                params.append(args.kind)
            if args.bucket == "blurry":
                sql += " AND f.blur < 40"
            rows = conn.execute(sql, params).fetchall()
            label = "size"
            described = f"bucket={args.bucket}"

    if not rows:
        console.print("[yellow]No faces matched.[/yellow]")
        return 1

    total = len(rows)
    if args.cluster is not None and not args.noise:
        # Show a cluster in time order and complete, up to the cap: the question is whether
        # it holds together, and shuffling would hide drift across years.
        rows = rows[: args.count]
    else:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.count]

    render(
        rows,
        out=args.out,
        columns=args.columns,
        cell=args.cell,
        context=args.context,
        label=label,
    )

    console.print(f"[green]Wrote {args.out}[/green] ({len(rows)} of {total:,} faces, {described})")

    if label == "date":
        console.print("Captions are capture dates.")
        if args.noise:
            console.print(
                "\n[bold]What to look for:[/bold] the noise bucket is 40.9% of the pool. "
                "If it is mostly tiny or junk detections, that is healthy. If it contains "
                "plenty of clear faces, clustering is too strict and labelling will be slower."
            )
        else:
            console.print(
                "\n[bold]What to look for:[/bold] is this ONE person? Mixed people means "
                "over-merging. Dates spanning years means a cross-era identity, which is the "
                "most valuable thing the gold set can hold."
            )
    else:
        console.print("Labels under each crop: inter-ocular px, yaw, blur.")
        console.print(
            "\n[bold]What to look for:[/bold] are these real faces, or detector noise "
            "(hands, patterns, background texture)? The answer sets the gating threshold."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
