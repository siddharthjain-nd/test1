"""Corpus composition report.

The numbers here directly drive the gold-set sampling strata in PLAN.md Phase 1:
how many face candidates exist, how the library splits across eras, and which cameras
appear. Until this runs, corpus size is a guess.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rich.console import Console
from rich.table import Table

from faceindex.ingest import FACE_CANDIDATE_KINDS, filename_exif_disagreements


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

    years = Table(title="Era distribution (capture date, any source)", header_style="bold")
    years.add_column("Year")
    years.add_column("Photos", justify="right")
    years.add_column("", justify="left")

    peak = max((int(r["n"]) for r in year_rows), default=1)
    for row in year_rows:
        bar = "#" * max(1, round(40 * int(row["n"]) / peak))
        years.add_row(str(row["year"]), f"{row['n']:,}", bar)
    if undated["n"]:
        years.add_row("[yellow]no date[/yellow]", f"{undated['n']:,}", "")
    console.print(years)

    source_rows = _rows(
        conn,
        f"SELECT COALESCE(taken_at_source, 'none') AS src, COUNT(*) AS n FROM photos "
        f"WHERE {where} GROUP BY src ORDER BY n DESC",
        params,
    )
    provenance = Table(title="Date provenance", header_style="bold")
    provenance.add_column("Source")
    provenance.add_column("Photos", justify="right")
    provenance.add_column("Share", justify="right")
    for row in source_rows:
        share = 100.0 * row["n"] / candidates["n"] if candidates["n"] else 0.0
        label = "[yellow]no date[/yellow]" if row["src"] == "none" else str(row["src"])
        provenance.add_row(label, f"{row['n']:,}", f"{share:5.1f}%")
    console.print(provenance)

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
            f"{undated:,} candidates ({100 * undated / candidates:.0f}%) still have no date after "
            "filename and folder recovery. Era stratification and the Phase 4 time prior both "
            "weaken. Rerun with --use-mtime if the drive's modification times are trustworthy."
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

    archives = _rows(
        conn,
        "SELECT rel_path FROM photos WHERE kind = 'archive' ORDER BY size_bytes DESC LIMIT 5",
    )
    if archives:
        listed = ", ".join(str(r["rel_path"])[:50] for r in archives)
        warnings.append(
            f"{len(archives)} archive(s) found and skipped. An archive inside a photo library "
            f"usually contains photos -- extract them into the corpus and rescan: {listed}"
        )

    disagreements, comparable = filename_exif_disagreements(conn)
    if comparable and disagreements / comparable > 0.10:
        warnings.append(
            f"{disagreements:,} of {comparable:,} photos ({100 * disagreements / comparable:.0f}%) "
            "have an EXIF capture date more than a week away from the date in their filename. "
            "One source is unreliable here; inspect a sample before trusting the era histogram."
        )

    if warnings:
        console.print("\n[bold yellow]Warnings[/bold yellow]")
        for warning in warnings:
            console.print(f"  - {warning}")


def print_diagnostics(conn: sqlite3.Connection, console: Console | None = None) -> None:
    """Explain what landed in the reject buckets, so nothing is discarded blindly."""
    console = console or Console()

    unsupported = Table(title="Unsupported extensions", header_style="bold")
    unsupported.add_column("Extension")
    unsupported.add_column("Files", justify="right")
    unsupported.add_column("Size", justify="right")
    unsupported.add_column("Example")
    for row in _rows(
        conn,
        "SELECT LOWER(reason) AS ext, COUNT(*) AS n, COALESCE(SUM(size_bytes),0) AS b, "
        "MIN(rel_path) AS example FROM photos WHERE kind = 'unsupported' "
        "GROUP BY ext ORDER BY n DESC LIMIT 25",
    ):
        unsupported.add_row(
            str(row["ext"]),
            f"{row['n']:,}",
            f"{row['b'] / 1e6:.0f} MB",
            str(row["example"])[:60],
        )
    console.print(unsupported)

    unreadable = Table(title="Unreadable files", header_style="bold")
    unreadable.add_column("Path")
    unreadable.add_column("Size", justify="right")
    unreadable.add_column("Reason")
    for row in _rows(
        conn,
        "SELECT rel_path, size_bytes, reason FROM photos WHERE kind = 'unreadable' LIMIT 40",
    ):
        unreadable.add_row(
            str(row["rel_path"])[:70],
            f"{row['size_bytes']:,}",
            str(row["reason"])[:60],
        )
    console.print(unreadable)

    formats = Table(title="Image formats among face candidates", header_style="bold")
    formats.add_column("Format")
    formats.add_column("Files", justify="right")
    where, params = _candidate_filter()
    for row in _rows(
        conn,
        f"SELECT COALESCE(image_format,'?') AS f, COUNT(*) AS n FROM photos "
        f"WHERE {where} GROUP BY f ORDER BY n DESC",
        params,
    ):
        formats.add_row(str(row["f"]), f"{row['n']:,}")
    console.print(formats)

    _print_top_folders(conn, console, where, params)


def _print_top_folders(
    conn: sqlite3.Connection,
    console: Console,
    where: str,
    params: tuple[str, ...],
    limit: int = 20,
) -> None:
    """Largest folders and how well dated they are.

    A big folder with few dates is where an era goes missing from the histogram.
    """
    counts: dict[str, list[int]] = {}
    for row in _rows(conn, f"SELECT rel_path, taken_at FROM photos WHERE {where}", params):
        parts = Path(str(row["rel_path"])).parts
        folder = str(Path(*parts[:-1])) if len(parts) > 1 else "."
        entry = counts.setdefault(folder, [0, 0])
        entry[0] += 1
        if row["taken_at"] is not None:
            entry[1] += 1

    table = Table(title=f"Largest folders (top {limit})", header_style="bold")
    table.add_column("Folder")
    table.add_column("Files", justify="right")
    table.add_column("Dated", justify="right")
    table.add_column("Undated", justify="right")

    for folder, (total, dated) in sorted(counts.items(), key=lambda kv: -kv[1][0])[:limit]:
        undated = total - dated
        marker = f"[red]{undated:,}[/red]" if undated > total / 2 else f"{undated:,}"
        table.add_row(folder[:60], f"{total:,}", f"{dated:,}", marker)
    console.print(table)


def print_path_inspection(
    conn: sqlite3.Connection, pattern: str, console: Console | None = None
) -> None:
    """Explain what happened to every file whose path contains ``pattern``.

    Answers "where did my 2024 photos go?" with facts instead of inference.
    """
    console = console or Console()
    like = f"%{pattern}%"

    total = _rows(conn, "SELECT COUNT(*) AS n FROM photos WHERE rel_path LIKE ?", (like,))[0]["n"]
    console.print(f"\n[bold]{total:,}[/bold] file(s) with a path containing [cyan]{pattern}[/cyan]")
    if not total:
        console.print("[yellow]Nothing matched. Try a shorter substring.[/yellow]")
        return

    kinds = Table(title="By kind", header_style="bold")
    kinds.add_column("Kind")
    kinds.add_column("Files", justify="right")
    kinds.add_column("Duplicates", justify="right")
    for row in _rows(
        conn,
        "SELECT kind, COUNT(*) AS n, SUM(duplicate_of IS NOT NULL) AS dup FROM photos "
        "WHERE rel_path LIKE ? GROUP BY kind ORDER BY n DESC",
        (like,),
    ):
        kinds.add_row(str(row["kind"]), f"{row['n']:,}", f"{row['dup'] or 0:,}")
    console.print(kinds)

    dates = Table(title="By year and date source", header_style="bold")
    dates.add_column("Year")
    dates.add_column("Source")
    dates.add_column("Files", justify="right")
    for row in _rows(
        conn,
        "SELECT COALESCE(substr(taken_at,1,4),'none') AS y, "
        "COALESCE(taken_at_source,'none') AS src, COUNT(*) AS n FROM photos "
        "WHERE rel_path LIKE ? GROUP BY y, src ORDER BY y, n DESC",
        (like,),
    ):
        label = "[yellow]no date[/yellow]" if row["y"] == "none" else str(row["y"])
        dates.add_row(label, str(row["src"]), f"{row['n']:,}")
    console.print(dates)

    samples = Table(title="Sample files", header_style="bold")
    samples.add_column("Path")
    samples.add_column("Kind")
    samples.add_column("Date")
    samples.add_column("Source")
    samples.add_column("Size", justify="right")
    for row in _rows(
        conn,
        "SELECT rel_path, kind, taken_at, taken_at_source, width, height FROM photos "
        "WHERE rel_path LIKE ? LIMIT 12",
        (like,),
    ):
        samples.add_row(
            str(row["rel_path"])[-55:],
            str(row["kind"]),
            str(row["taken_at"] or "-")[:19],
            str(row["taken_at_source"] or "-"),
            f"{row['width'] or '?'}x{row['height'] or '?'}",
        )
    console.print(samples)
