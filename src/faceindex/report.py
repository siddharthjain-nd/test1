"""Corpus composition report.

The numbers here directly drive the gold-set sampling strata in PLAN.md Phase 1:
how many face candidates exist, how the library splits across eras, and which cameras
appear. Until this runs, corpus size is a guess.
"""

from __future__ import annotations

import sqlite3

from rich.console import Console
from rich.table import Table

from faceindex.ingest import FACE_CANDIDATE_KINDS


def _rows(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
    return list(conn.execute(sql, params).fetchall())


def _candidate_filter() -> tuple[str, tuple[str, ...]]:
    placeholders = ",".join("?" for _ in FACE_CANDIDATE_KINDS)
    kinds = tuple(sorted(FACE_CANDIDATE_KINDS))
    return f"kind IN ({placeholders}) AND duplicate_of IS NULL", kinds


def print_report(conn: sqlite3.Connection, console: Console | None = None) -> None:
    console = console or Console()

    total = _rows(conn, "SELECT COUNT(*) AS n, COALESCE(SUM(size_bytes),0) AS b FROM photos")[0]

    kinds = Table(title="Corpus composition", header_style="bold")
    kinds.add_column("Kind")
    kinds.add_column("Files", justify="right")
    kinds.add_column("Size", justify="right")
    kinds.add_column("Share", justify="right")

    for row in _rows(
        conn,
        "SELECT kind, COUNT(*) AS n, COALESCE(SUM(size_bytes),0) AS b "
        "FROM photos GROUP BY kind ORDER BY n DESC",
    ):
        share = 100.0 * row["n"] / total["n"] if total["n"] else 0.0
        kinds.add_row(row["kind"], f"{row['n']:,}", f"{row['b'] / 1e9:.1f} GB", f"{share:5.1f}%")

    kinds.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total['n']:,}[/bold]",
        f"[bold]{total['b'] / 1e9:.1f} GB[/bold]",
        "",
    )
    console.print(kinds)

    where, params = _candidate_filter()

    duplicates = _rows(
        conn,
        "SELECT COUNT(*) AS n, COALESCE(SUM(size_bytes),0) AS b "
        "FROM photos WHERE duplicate_of IS NOT NULL",
    )[0]
    candidates = _rows(conn, f"SELECT COUNT(*) AS n FROM photos WHERE {where}", params)[0]

    summary = Table(title="Face-detection input", header_style="bold")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Duplicates found", f"{duplicates['n']:,}")
    summary.add_row("Duplicate bytes", f"{duplicates['b'] / 1e9:.1f} GB")
    summary.add_row("[bold]Unique face candidates[/bold]", f"[bold]{candidates['n']:,}[/bold]")
    console.print(summary)

    year_rows = _rows(
        conn,
        f"SELECT substr(taken_at, 1, 4) AS year, COUNT(*) AS n FROM photos "
        f"WHERE {where} AND taken_at IS NOT NULL GROUP BY year ORDER BY year",
        params,
    )
    undated = _rows(
        conn, f"SELECT COUNT(*) AS n FROM photos WHERE {where} AND taken_at IS NULL", params
    )[0]

    years = Table(title="Era distribution (EXIF capture date)", header_style="bold")
    years.add_column("Year")
    years.add_column("Photos", justify="right")
    years.add_column("", justify="left")

    peak = max((int(r["n"]) for r in year_rows), default=1)
    for row in year_rows:
        bar = "#" * max(1, round(40 * int(row["n"]) / peak))
        years.add_row(str(row["year"]), f"{row['n']:,}", bar)
    if undated["n"]:
        years.add_row("[yellow]no EXIF date[/yellow]", f"{undated['n']:,}", "")
    console.print(years)

    camera_rows = _rows(
        conn,
        f"SELECT COALESCE(camera_make, '?') AS mk, COALESCE(camera_model, '?') AS md, "
        f"COUNT(*) AS n FROM photos WHERE {where} AND camera_model IS NOT NULL "
        f"GROUP BY mk, md ORDER BY n DESC LIMIT 15",
        params,
    )
    if camera_rows:
        cameras = Table(title="Cameras", header_style="bold")
        cameras.add_column("Make")
        cameras.add_column("Model")
        cameras.add_column("Photos", justify="right")
        for row in camera_rows:
            cameras.add_row(str(row["mk"]), str(row["md"]), f"{row['n']:,}")
        console.print(cameras)

    _print_warnings(conn, console, where, params, year_rows, undated["n"], candidates["n"])


def _print_warnings(
    conn: sqlite3.Connection,
    console: Console,
    where: str,
    params: tuple[str, ...],
    year_rows: list[sqlite3.Row],
    undated: int,
    candidates: int,
) -> None:
    """Surface conditions that would silently invalidate the gold set."""
    warnings: list[str] = []

    if candidates == 0:
        warnings.append("No face candidates found. Check the corpus path and extensions.")

    if candidates and undated / candidates > 0.25:
        warnings.append(
            f"{undated:,} candidates ({100 * undated / candidates:.0f}%) have no EXIF date. "
            "Era stratification and the Phase 4 time prior both weaken. "
            "Consider falling back to file mtime."
        )

    if len(year_rows) < 3:
        warnings.append(
            "Fewer than 3 distinct years present. The gold set needs cross-era identities "
            "to measure age drift, which is the hardest failure mode (PLAN.md Phase 1)."
        )

    camera_count = _rows(
        conn,
        f"SELECT COUNT(DISTINCT camera_model) AS n FROM photos WHERE {where} "
        f"AND camera_model IS NOT NULL",
        params,
    )[0]["n"]
    if camera_count <= 1:
        warnings.append(
            "Only one camera model detected. If you changed phones, some photos may be "
            "missing from the corpus."
        )

    unreadable = _rows(conn, "SELECT COUNT(*) AS n FROM photos WHERE kind = 'unreadable'")[0]["n"]
    if unreadable:
        warnings.append(f"{unreadable:,} unreadable files. Inspect before assuming they are junk.")

    raw = _rows(conn, "SELECT COUNT(*) AS n FROM photos WHERE kind = 'raw'")[0]["n"]
    if raw:
        warnings.append(f"{raw:,} RAW files skipped (out of scope). They contain faces you lose.")

    if warnings:
        console.print("\n[bold yellow]Warnings[/bold yellow]")
        for warning in warnings:
            console.print(f"  - {warning}")
