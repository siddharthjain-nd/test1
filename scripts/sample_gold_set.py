#!/usr/bin/env python3
"""Choose the ~1,800 faces a human will label, and prove the mix is what was asked for.

The composition report is not decoration. A gold set whose strata were assumed rather than
verified produces confident numbers about the wrong population, and the error is
undetectable afterwards -- every downstream conclusion inherits it silently. So the
sampler asserts its own output and exits non-zero when it misses.

Usage
    python scripts/sample_gold_set.py
    python scripts/sample_gold_set.py --target-size 1800
    python scripts/sample_gold_set.py --tolerance 0.07     # loosen if a stratum is starved
    python scripts/sample_gold_set.py --report-only        # re-print the last sample
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from faceindex import paths, sampling, store

console = Console()

# The retouching-app folder measured in this corpus: 2,163 images, ~12% of candidates.
# These filters alter face geometry, which is exactly what the embedding encodes.
DEFAULT_BEAUTY_MARKER = "You cam perfect"


def print_report(report: sampling.CompositionReport) -> None:
    for dimension in report.dimensions:
        table = Table(title=dimension.dimension, header_style="bold")
        table.add_column("Category")
        table.add_column("Faces", justify="right")
        table.add_column("Actual", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Delta", justify="right")

        for category, count, actual, target in dimension.rows:
            delta = actual - target
            colour = "green" if abs(delta) <= 0.05 else "yellow" if abs(delta) <= 0.10 else "red"
            table.add_row(
                category,
                f"{count:,}",
                f"{actual:6.1%}",
                f"{target:6.1%}",
                f"[{colour}]{delta:+6.1%}[/{colour}]",
            )
        console.print(table)

    summary = Table(title="Sample summary", header_style="bold")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Faces selected", f"{report.n_selected:,}")
    summary.add_row("Bootstrap clusters represented", f"{report.n_clusters:,}")
    # Informational only. The >=10 requirement is checked at export, against human labels.
    summary.add_row("Cross-era clusters (informational)", f"{report.n_cross_era_clusters:,}")
    summary.add_row("Reserved: noise review", f"{report.n_noise_review:,}")
    summary.add_row("Reserved: low-confidence detections", f"{report.n_detector_fp:,}")
    console.print(summary)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--target-size", type=int, default=1800)
    parser.add_argument("--per-cluster-per-day", type=int, default=3)
    parser.add_argument("--per-cluster-total", type=int, default=70)
    parser.add_argument("--noise-share", type=float, default=0.10)
    parser.add_argument("--detector-fp", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--beauty-marker", default=DEFAULT_BEAUTY_MARKER)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write the sample even if the composition report fails its assertions.",
    )
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    config = sampling.SampleConfig(
        target_size=args.target_size,
        per_cluster_per_day=args.per_cluster_per_day,
        per_cluster_total=args.per_cluster_total,
        noise_review_share=args.noise_share,
        detector_fp_count=args.detector_fp,
        random_seed=args.seed,
    )

    with store.open_index(db_path) as conn:
        if args.report_only:
            rows = conn.execute("SELECT COUNT(*) AS n FROM gold_candidates").fetchone()["n"]
            if not rows:
                console.print("[yellow]No sample stored yet.[/yellow]")
                return 1
            console.print(f"Stored sample: {rows:,} candidates.")
            return 0

        console.print("Loading candidates…")
        candidates = sampling.load_candidates(conn, beauty_marker=args.beauty_marker)
        if not candidates:
            console.print(
                "[red]No embedded faces found. Run embed_faces.py and "
                "bootstrap_cluster.py first.[/red]"
            )
            return 1

        capped = sampling.apply_caps(candidates, config)
        console.print(
            f"[bold]Pool     :[/bold] {len(candidates):,} faces "
            f"-> {len(capped):,} after per-person caps "
            f"({args.per_cluster_per_day}/day, {args.per_cluster_total} total)"
        )

        selected = sampling.select(capped, config)
        console.print(f"[bold]Selected :[/bold] {len(selected):,} faces\n")

        report = sampling.composition_report(selected, config, tolerance=args.tolerance)
        print_report(report)

        for warning in report.warnings:
            console.print(f"\n[yellow]Note:[/yellow] {warning}")

        if report.failures:
            console.print("\n[red]Composition report FAILED:[/red]")
            for failure in report.failures:
                console.print(f"  • {failure}")
            console.print(
                "\nA stratum usually misses because the corpus cannot supply it -- the pool "
                "is 33.8% tiny faces against a 10% target, for example. Either loosen "
                "--tolerance, adjust the targets in sampling.DEFAULT_TARGETS and log the "
                "decision in PLAN.md, or accept the skew with --force and record it."
            )
            if not args.force:
                return 2
            console.print("[yellow]--force given: writing the sample anyway.[/yellow]")
        else:
            console.print("\n[green]Composition report passed every target.[/green]")

        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        sampling.write_candidates(conn, selected, run_id)
        console.print(f"Wrote {len(selected):,} candidates (run {run_id}).")

    console.print("\n[green]Sample ready.[/green] Next: python scripts/label_gold_set.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
