#!/usr/bin/env python3
"""Render a grid of face crops to a single image, for eyeballing the pool.

Statistics say 34% of detected faces are tiny. Only looking at them reveals whether they
are genuine small faces in crowd shots or detector false positives, and the answer changes
the quality-gating thresholds and the gold-set strata.

Usage
    python scripts/contact_sheet.py --bucket tiny --out /tmp/tiny.jpg
    python scripts/contact_sheet.py --bucket profile --context
    python scripts/contact_sheet.py --min-iod 0 --max-iod 12 --count 144
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--bucket", choices=sorted(BUCKETS), default="tiny")
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

    min_iod, max_iod, min_yaw, max_yaw = BUCKETS[args.bucket]
    if args.min_iod is not None:
        min_iod = args.min_iod
    if args.max_iod is not None:
        max_iod = args.max_iod

    sql = (
        "SELECT f.crop_path, f.context_path, f.interocular_px, f.yaw_deg, f.blur, "
        "f.det_score, p.kind FROM faces f JOIN photos p ON p.id = f.photo_id "
        "WHERE f.pool_version = ? AND f.interocular_px >= ? AND f.interocular_px < ? "
        "AND ABS(f.yaw_deg) >= ? AND ABS(f.yaw_deg) < ?"
    )
    params: list[object] = [facepool.POOL_VERSION, min_iod, max_iod, min_yaw, max_yaw]
    if args.kind:
        sql += " AND p.kind = ?"
        params.append(args.kind)
    if args.bucket == "blurry":
        sql += " AND f.blur < 40"

    with store.open_index(db_path, read_only=True) as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        console.print("[yellow]No faces matched.[/yellow]")
        return 1

    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.count]

    columns = args.columns
    cell = args.cell
    label_height = 14
    row_count = (len(rows) + columns - 1) // columns

    sheet = Image.new("RGB", (columns * cell, row_count * (cell + label_height)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(rows):
        source = row["context_path"] if args.context else row["crop_path"]
        x = (index % columns) * cell
        y = (index // columns) * (cell + label_height)
        try:
            with Image.open(str(source)) as image:
                sheet.paste(image.convert("RGB").resize((cell, cell)), (x, y))
        except (OSError, ValueError):
            draw.rectangle([x, y, x + cell, y + cell], fill=(80, 0, 0))

        draw.text(
            (x + 2, y + cell + 1),
            f"{row['interocular_px']:.0f}px y{row['yaw_deg']:+.0f} b{row['blur']:.0f}",
            fill=(200, 200, 200),
        )

    sheet.save(args.out, "JPEG", quality=88)

    console.print(f"[green]Wrote {args.out}[/green] ({len(rows)} faces, bucket={args.bucket})")
    console.print("Labels under each crop: inter-ocular px, yaw, blur.")
    console.print(
        "\n[bold]What to look for:[/bold] are these real faces, or detector noise "
        "(hands, patterns, background texture)? The answer sets the gating threshold."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
