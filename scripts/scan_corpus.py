#!/usr/bin/env python3
"""Scan a photo library: walk, classify, hash, dedup, extract EXIF.

Cheap by design. No pixel data is decoded and videos are rejected on extension before
any I/O, so runtime is bounded by directory traversal plus ~128 KB per candidate file.

Resumable: rerun after an interruption and only new or changed files are processed.

Usage
    python scripts/scan_corpus.py --root "/path/to/photo Timeline"
    python scripts/scan_corpus.py --root ... --no-dedup     # skip the SHA-256 pass
    python scripts/scan_corpus.py --report-only             # re-print from the database
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from tqdm import tqdm

from faceindex import ingest, report, store
from faceindex.paths import data_dir, index_db_path

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Photo library root. Quote it if the path contains spaces.",
    )
    parser.add_argument("--db", type=Path, default=None, help="Index path (default: data/index.db)")
    parser.add_argument(
        "--no-dedup", action="store_true", help="Skip the duplicate resolution pass"
    )
    parser.add_argument("--report-only", action="store_true", help="Print the report and exit")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Show what landed in the reject buckets, then exit",
    )
    parser.add_argument(
        "--inspect",
        metavar="SUBSTRING",
        help="Explain what happened to every file whose path contains SUBSTRING, then exit",
    )
    parser.add_argument(
        "--backfill-dates-only",
        action="store_true",
        help="Recover missing dates from filenames and folders without rescanning",
    )
    parser.add_argument(
        "--use-mtime",
        action="store_true",
        help="Last-resort date fallback to file modification time. Off by default: copying "
        "a library resets mtimes, and a wrong date is worse than no date.",
    )
    parser.add_argument("--follow-symlinks", action="store_true")
    args = parser.parse_args()

    db_path = args.db or index_db_path()

    if args.diagnose:
        if not db_path.exists():
            console.print(f"[red]No index at {db_path}. Run a scan first.[/red]")
            return 1
        with store.open_index(db_path, read_only=True) as conn:
            report.print_diagnostics(conn, console)
        return 0

    if args.inspect:
        if not db_path.exists():
            console.print(f"[red]No index at {db_path}. Run a scan first.[/red]")
            return 1
        with store.open_index(db_path, read_only=True) as conn:
            report.print_path_inspection(conn, args.inspect, console)
        return 0

    if args.backfill_dates_only:
        if not db_path.exists():
            console.print(f"[red]No index at {db_path}. Run a scan first.[/red]")
            return 1
        with store.open_index(db_path) as conn:
            counts = ingest.backfill_dates(conn, use_mtime=args.use_mtime)
            console.print(
                f"Recovered dates -- filename: {counts['filename']:,}, "
                f"folder: {counts['folder']:,}, mtime: {counts['mtime']:,}\n"
            )
            report.print_report(conn, console)
        return 0

    if args.report_only:
        if not db_path.exists():
            console.print(f"[red]No index at {db_path}. Run a scan first.[/red]")
            return 1
        with store.open_index(db_path, read_only=True) as conn:
            report.print_report(conn, console)
        return 0

    if args.root is None:
        parser.error("--root is required unless --report-only is given")

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        console.print(f"[red]Not a directory: {root}[/red]")
        return 1

    data_dir().mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Corpus:[/bold] {root}")
    console.print(f"[bold]Index :[/bold] {db_path}\n")

    with store.open_index(db_path) as conn:
        store.set_meta(conn, "corpus_root", str(root))

        console.print("Counting files...")
        total = sum(1 for _ in ingest.iter_files(root, follow_symlinks=args.follow_symlinks))
        console.print(f"{total:,} files found.\n")

        walker = tqdm(
            ingest.iter_files(root, follow_symlinks=args.follow_symlinks),
            total=total,
            unit="file",
            smoothing=0.05,
        )
        stats = ingest.scan(conn, root, follow_symlinks=args.follow_symlinks, progress=walker)

        console.print(
            f"\nProcessed {stats.processed:,}, resumed-skip {stats.skipped_resumed:,}, "
            f"seen {stats.seen:,}."
        )
        if stats.errors:
            console.print(f"[yellow]{len(stats.errors)} file(s) errored. First few:[/yellow]")
            for line in stats.errors[:5]:
                console.print(f"  {line}")

        if not args.no_dedup:
            console.print("\nResolving duplicates (SHA-256 on quick-hash collisions)...")
            count, freed = ingest.resolve_duplicates(conn)
            console.print(f"{count:,} duplicate file(s), {freed / 1e9:.1f} GB redundant.")

        console.print("\nRecovering missing dates from filenames and folders...")
        counts = ingest.backfill_dates(conn, use_mtime=args.use_mtime)
        console.print(
            f"filename: {counts['filename']:,}, folder: {counts['folder']:,}, "
            f"mtime: {counts['mtime']:,}\n"
        )

        report.print_report(conn, console)

    console.print(
        "\n[green]Scan complete.[/green] Record the composition numbers in DEVLOG.md, "
        "then run the face-pool build."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
