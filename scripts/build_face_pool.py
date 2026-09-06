#!/usr/bin/env python3
"""Build the face pool: decode, detect, align, and store crops plus attributes.

The one expensive pass that must read the original photos. Everything downstream works
from the 112x112 crops this produces, so a face missed here is missed permanently --
which is why it defaults to the large detector rather than the fast one.

Resumable at photo granularity. Interrupt it and rerun the identical command.

Usage
    python scripts/build_face_pool.py
    python scripts/build_face_pool.py --limit 200          # trial run first
    python scripts/build_face_pool.py --detector buffalo_sc/det_500m.onnx
    python scripts/build_face_pool.py --stats              # report without processing
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from faceindex import facepool, paths, store

console = Console()


def print_stats(conn: object) -> None:
    rows = conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) AS photos, COALESCE(SUM(n_faces),0) AS faces, "
        "SUM(error IS NOT NULL) AS errors FROM photo_pool_status WHERE pool_version = ?",
        (facepool.POOL_VERSION,),
    ).fetchone()

    table = Table(title="Face pool", header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Photos processed", f"{rows['photos']:,}")
    table.add_row("Faces detected", f"{rows['faces']:,}")
    table.add_row("Photos with errors", f"{rows['errors'] or 0:,}")
    if rows["photos"]:
        table.add_row("Faces per photo", f"{rows['faces'] / rows['photos']:.2f}")
    console.print(table)

    if not rows["faces"]:
        return

    buckets = Table(title="Face size (inter-ocular distance, decoded pixels)", header_style="bold")
    buckets.add_column("Bucket")
    buckets.add_column("Faces", justify="right")
    buckets.add_column("Share", justify="right")
    for label, lo, hi in (
        ("tiny  <20", 0, 20),
        ("small 20-40", 20, 40),
        ("med   40-80", 40, 80),
        ("large >80", 80, 10**9),
    ):
        n = conn.execute(  # type: ignore[attr-defined]
            "SELECT COUNT(*) AS n FROM faces WHERE pool_version = ? "
            "AND interocular_px >= ? AND interocular_px < ?",
            (facepool.POOL_VERSION, lo, hi),
        ).fetchone()["n"]
        buckets.add_row(label, f"{n:,}", f"{100 * n / rows['faces']:5.1f}%")
    console.print(buckets)

    pose = Table(title="Pose", header_style="bold")
    pose.add_column("Bucket")
    pose.add_column("Faces", justify="right")
    pose.add_column("Share", justify="right")
    for label, lo, hi in (("frontal <15", 0, 15), ("semi 15-45", 15, 45), ("profile >45", 45, 91)):
        n = conn.execute(  # type: ignore[attr-defined]
            "SELECT COUNT(*) AS n FROM faces WHERE pool_version = ? "
            "AND ABS(yaw_deg) >= ? AND ABS(yaw_deg) < ?",
            (facepool.POOL_VERSION, lo, hi),
        ).fetchone()["n"]
        pose.add_row(label, f"{n:,}", f"{100 * n / rows['faces']:5.1f}%")
    console.print(pose)

    by_kind = Table(title="Faces by photo kind", header_style="bold")
    by_kind.add_column("Kind")
    by_kind.add_column("Photos", justify="right")
    by_kind.add_column("Faces", justify="right")
    by_kind.add_column("Faces/photo", justify="right")
    for row in conn.execute(  # type: ignore[attr-defined]
        "SELECT p.kind AS kind, COUNT(DISTINCT s.photo_id) AS photos, "
        "COALESCE(SUM(s.n_faces),0) AS faces FROM photo_pool_status s "
        "JOIN photos p ON p.id = s.photo_id WHERE s.pool_version = ? "
        "GROUP BY p.kind ORDER BY faces DESC",
        (facepool.POOL_VERSION,),
    ):
        ratio = row["faces"] / row["photos"] if row["photos"] else 0.0
        by_kind.add_row(row["kind"], f"{row['photos']:,}", f"{row['faces']:,}", f"{ratio:.2f}")
    console.print(by_kind)

    _print_size_bias(conn)


def _print_size_bias(conn: object) -> None:
    """Face size split by photo kind.

    Tests the risk that messaging-app recompression makes forwarded faces systematically
    smaller, so that quality gating on absolute pixels would silently discard the user's
    party photos while keeping their camera photos (PLAN.md risk register).
    """
    table = Table(title="Face size by photo kind (tests gating bias)", header_style="bold")
    table.add_column("Kind")
    table.add_column("Faces", justify="right")
    table.add_column("Median IOD", justify="right")
    table.add_column("Tiny <20px", justify="right")
    table.add_column("Usable >=40px", justify="right")

    kinds = conn.execute(  # type: ignore[attr-defined]
        "SELECT DISTINCT p.kind AS kind FROM faces f JOIN photos p ON p.id = f.photo_id "
        "WHERE f.pool_version = ?",
        (facepool.POOL_VERSION,),
    ).fetchall()

    for entry in kinds:
        kind = str(entry["kind"])
        sizes = [
            float(r["interocular_px"])
            for r in conn.execute(  # type: ignore[attr-defined]
                "SELECT f.interocular_px FROM faces f JOIN photos p ON p.id = f.photo_id "
                "WHERE f.pool_version = ? AND p.kind = ? ORDER BY f.interocular_px",
                (facepool.POOL_VERSION, kind),
            )
        ]
        if not sizes:
            continue
        median = sizes[len(sizes) // 2]
        tiny = sum(1 for s in sizes if s < 20) / len(sizes)
        usable = sum(1 for s in sizes if s >= 40) / len(sizes)
        table.add_row(
            kind,
            f"{len(sizes):,}",
            f"{median:.1f} px",
            f"{100 * tiny:5.1f}%",
            f"{100 * usable:5.1f}%",
        )

    console.print(table)


def main() -> int:
    default_workers = max(1, min(4, (os.cpu_count() or 2) // 2))

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument(
        "--detector",
        default="buffalo_l/det_10g.onnx",
        help="Relative to models/. Default is the large detector: a face missed here is "
        "missed permanently, and re-running requires the original photos again.",
    )
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--max-long-edge", type=int, default=2048)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N photos")
    parser.add_argument("--no-context", action="store_true", help="Skip human-labelling crops")
    parser.add_argument("--stats", action="store_true", help="Report progress and exit")
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}. Run scan_corpus.py first.[/red]")
        return 1

    if args.stats:
        with store.open_index(db_path, read_only=True) as conn:
            print_stats(conn)
        return 0

    detector_path = paths.models_dir() / args.detector
    if not detector_path.exists():
        console.print(f"[red]Detector not found: {detector_path}[/red]")
        console.print("Run: python scripts/download_models.py")
        return 1

    paths.ensure_dirs()
    config = facepool.PoolConfig(
        detector_path=detector_path,
        crops_dir=paths.crops_dir(),
        context_dir=paths.context_crops_dir(),
        max_long_edge=args.max_long_edge,
        det_input_size=args.det_size,
        score_threshold=args.score_threshold,
        threads_per_worker=args.threads_per_worker,
        save_context=not args.no_context,
    )

    with store.open_index(db_path) as conn:
        tasks = facepool.pending_photos(conn, limit=args.limit)
        if not tasks:
            console.print("[green]Nothing to do: every candidate is already processed.[/green]\n")
            print_stats(conn)
            return 0

        console.print(f"[bold]Detector :[/bold] {args.detector} @ {args.det_size}px")
        console.print(f"[bold]Decode   :[/bold] max {args.max_long_edge}px long edge")
        console.print(f"[bold]Workers  :[/bold] {args.workers} x {args.threads_per_worker} threads")
        console.print(f"[bold]Pending  :[/bold] {len(tasks):,} photos\n")

        started = time.perf_counter()
        total_faces = 0
        errors: list[str] = []

        progress = tqdm(total=len(tasks), unit="photo", smoothing=0.05)
        for result in facepool.run(conn, tasks, config, workers=args.workers):
            total_faces += len(result.faces)
            if result.error:
                errors.append(f"photo {result.photo_id}: {result.error}")
            progress.update(1)
            progress.set_postfix(faces=total_faces)
        progress.close()

        elapsed = time.perf_counter() - started
        rate = len(tasks) / elapsed if elapsed else 0.0
        console.print(
            f"\nProcessed {len(tasks):,} photos in {elapsed / 60:.1f} min "
            f"({rate:.1f} photo/s), found {total_faces:,} faces."
        )
        if errors:
            console.print(f"[yellow]{len(errors)} photo(s) errored. First few:[/yellow]")
            for line in errors[:5]:
                console.print(f"  {line}")

        print_stats(conn)

    console.print("\n[green]Face pool built.[/green] Record the numbers in DEVLOG.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
