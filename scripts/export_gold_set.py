#!/usr/bin/env python3
"""Freeze the labelled gold set to data/gold/labels.csv.

The CSV, not the database, is the portable artifact: ~200 KB that travels between machines
alongside the crops, and the thing the evaluation harness reads. It carries the auto-derived
attributes with each row so that results can be sliced by pose, size, era and quality --
"F1 is 0.92 overall but 0.61 on profile faces" is actionable, a single aggregate number is not.

Freeze it once labelling is done. Editing it after experiments begin makes the numbers
non-comparable, which quietly destroys the point of having a gold set at all.

Usage
    python scripts/export_gold_set.py
    python scripts/export_gold_set.py --out data/gold/labels.csv
    python scripts/export_gold_set.py --check     # validate without writing
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from faceindex import paths, store

console = Console()

COLUMNS = (
    "face_id",
    "label",
    "person_id",
    "occlusion",
    "photo_id",
    "rel_path",
    "taken_at",
    "kind",
    "camera_model",
    "det_score",
    "interocular_px",
    "relative_size",
    "yaw_deg",
    "roll_deg",
    "blur",
    "brightness",
    "n_faces_in_photo",
    "size_bucket",
    "pose_bucket",
    "era_bucket",
    "quality_bucket",
    "filtered_bucket",
    "group_bucket",
    "bootstrap_cluster",
    "reserved_for",
    "labelled_at",
)


def fetch_rows(conn: object) -> list[dict[str, object]]:
    rows = conn.execute(  # type: ignore[attr-defined]
        """
        SELECT
            g.face_id, g.label, g.person_id, g.occlusion, g.labelled_at,
            c.stratum, c.bootstrap_cluster, c.reserved_for,
            f.photo_id, f.det_score, f.interocular_px, f.relative_size,
            f.yaw_deg, f.roll_deg, f.blur, f.brightness,
            p.rel_path, p.taken_at, p.kind, p.camera_model,
            (SELECT COUNT(*) FROM faces x WHERE x.photo_id = f.photo_id) AS n_faces
        FROM gold_labels g
        JOIN gold_candidates c ON c.face_id = g.face_id
        JOIN faces f  ON f.id = g.face_id
        JOIN photos p ON p.id = f.photo_id
        ORDER BY g.face_id
        """
    ).fetchall()

    out: list[dict[str, object]] = []
    for row in rows:
        strata = json.loads(row["stratum"])
        out.append(
            {
                "face_id": row["face_id"],
                "label": row["label"],
                "person_id": row["person_id"] or "",
                "occlusion": row["occlusion"] or "",
                "photo_id": row["photo_id"],
                "rel_path": row["rel_path"],
                "taken_at": row["taken_at"] or "",
                "kind": row["kind"],
                "camera_model": row["camera_model"] or "",
                "det_score": round(float(row["det_score"]), 4),
                "interocular_px": round(float(row["interocular_px"] or 0.0), 2),
                "relative_size": round(float(row["relative_size"] or 0.0), 5),
                "yaw_deg": round(float(row["yaw_deg"] or 0.0), 2),
                "roll_deg": round(float(row["roll_deg"] or 0.0), 2),
                "blur": round(float(row["blur"] or 0.0), 4),
                "brightness": round(float(row["brightness"] or 0.0), 2),
                "n_faces_in_photo": row["n_faces"],
                "size_bucket": strata.get("size", ""),
                "pose_bucket": strata.get("pose", ""),
                "era_bucket": strata.get("era", ""),
                "quality_bucket": strata.get("quality", ""),
                "filtered_bucket": strata.get("filtered", ""),
                "group_bucket": strata.get("group", ""),
                "bootstrap_cluster": row["bootstrap_cluster"],
                "reserved_for": row["reserved_for"] or "",
                "labelled_at": row["labelled_at"],
            }
        )
    return out


def print_readability(rows: list[dict[str, object]], column: str, order: tuple[str, ...]) -> None:
    """How often a human could not read a face, sliced by an auto-derived attribute.

    This turns the `unsure` labels from a shrug into a measurement. Phase 3 has to choose a
    quality gate, and the honest upper bound on what a *machine* should attempt is what a
    *human* could read: if four in five of the worst-quality faces defeat you, a clusterer
    has no business forming identities from them. Those thresholds are otherwise guesswork.
    """
    present = [r for r in rows if r.get(column)]
    if not present:
        return

    table = Table(title=f"Human readability by {column.replace('_', ' ')}", header_style="bold")
    table.add_column(column.replace("_bucket", ""))
    table.add_column("Faces", justify="right")
    table.add_column("Named", justify="right")
    table.add_column("Stranger", justify="right")
    table.add_column("Not a face", justify="right")
    table.add_column("Unreadable", justify="right")

    buckets = [b for b in order if any(str(r[column]) == b for r in present)]
    for bucket in buckets:
        group = [r for r in present if str(r[column]) == bucket]
        counts = Counter(str(r["label"]) for r in group)
        unreadable = counts["unsure"] / len(group)
        colour = "red" if unreadable > 0.5 else "yellow" if unreadable > 0.25 else "green"
        table.add_row(
            bucket,
            f"{len(group):,}",
            f"{counts['person']:,}",
            f"{counts['not_of_interest']:,}",
            f"{counts['non_face']:,}",
            f"[{colour}]{unreadable:5.1%}[/{colour}]",
        )
    console.print(table)


def person_spans(rows: list[dict[str, object]]) -> dict[str, tuple[int, int]]:
    """Earliest and latest year of each named person's faces."""
    years: dict[str, list[int]] = {}
    for row in rows:
        if row["label"] == "person" and row["taken_at"]:
            years.setdefault(str(row["person_id"]), []).append(int(str(row["taken_at"])[:4]))
    return {person: (min(seen), max(seen)) for person, seen in years.items() if seen}


def print_spans(rows: list[dict[str, object]]) -> None:
    """How far apart each person's oldest and newest photo are.

    The era buckets exist to spread the *sample* across time; using membership of two of them
    as a measure of one person's time span was a category error. Someone photographed in 2010
    and 2022 spans twelve years and is an excellent cross-age case, yet failed that check
    because 2022 fell in the middle bucket, while 2016-to-2024 passed at eight years.

    Years between a person's first and last photo is the quantity actually of interest, so
    measure that directly and let the threshold follow the evidence.
    """
    spans = person_spans(rows)
    if not spans:
        return

    gaps = sorted(((hi - lo), person) for person, (lo, hi) in spans.items())

    table = Table(title="How far apart each person's photos are", header_style="bold")
    table.add_column("Time span")
    table.add_column("People", justify="right")
    table.add_column("Reading")

    for label, low, high, reading in (
        ("15+ years", 15, 10**9, "the hardest case there is"),
        ("10-14 years", 10, 15, "hard: real cross-age drift"),
        ("5-9 years", 5, 10, "moderate, and severe for a child"),
        ("1-4 years", 1, 5, "mild"),
        ("same year", 0, 1, "no ageing tested"),
    ):
        band = [p for gap, p in gaps if low <= gap < high]
        if band:
            table.add_row(label, str(len(band)), reading)
    console.print(table)

    widest = gaps[-5:][::-1]
    if widest:
        console.print(
            "Widest spans: "
            + " · ".join(
                f"[bold]{person}[/bold] {spans[person][0]}-{spans[person][1]} ({gap}y)"
                for gap, person in widest
            )
        )


def validate(rows: list[dict[str, object]]) -> list[str]:
    """Checks whose failure would silently invalidate every downstream number."""
    problems: list[str] = []
    labels = Counter(str(r["label"]) for r in rows)

    people = Counter(str(r["person_id"]) for r in rows if r["label"] == "person")
    singletons = [p for p, n in people.items() if n < 2]

    if len(people) < 25:
        problems.append(
            f"{len(people)} identities labelled; PLAN.md targets 25-35. Fewer identities "
            f"means fewer chances to confuse similar-looking people, so the score inflates."
        )
    if labels["person"] < 1000:
        problems.append(f"only {labels['person']} person-labelled faces; PLAN.md targets ~1,400.")
    if labels["not_of_interest"] < 100:
        problems.append(
            f"only {labels['not_of_interest']} strangers labelled; PLAN.md targets ~300. "
            f"Without them the clusterer is never tested on its right to call a face noise."
        )
    # Informational, not a defect. A one-face identity contributes no *positive* pairs, but it
    # still tests whether the system wrongly merges them into somebody else, and BCubed scores
    # them correctly when they are left alone. Only worth knowing about in bulk.
    if len(singletons) > len(people) * 0.5:
        problems.append(
            f"{len(singletons)} of {len(people)} identities have a single face. They still test "
            f"against wrong merges, but contribute no same-person pairs, so the positive side of "
            f"the score rests on the other {len(people) - len(singletons)}."
        )

    # A person photographed years apart is the highest-value thing the gold set can hold --
    # cross-age drift is the system's worst failure mode.
    #
    # Measured as the gap between that person's first and last photo, NOT as membership of
    # the oldest and most recent era buckets. Those buckets exist to spread the sample across
    # time; reusing them here rejected a 2010-to-2022 person at twelve years while accepting
    # a 2016-to-2024 one at eight.
    spans = person_spans(rows)
    hard = [p for p, (lo, hi) in spans.items() if hi - lo >= 10]
    moderate = [p for p, (lo, hi) in spans.items() if hi - lo >= 5]

    if len(moderate) < 10:
        problems.append(
            f"only {len(moderate)} identities span 5 years or more ({len(hard)} span 10+); "
            f"PLAN.md wants >=10 to make cross-age drift measurable at all. To fix: label more "
            f"faces of people you have already named, from years they are currently missing. "
            f"If the library genuinely lacks them, lower the requirement and record why -- do "
            f"not invent pairs that are not there."
        )
    elif len(hard) < 5:
        problems.append(
            f"{len(moderate)} identities span 5+ years, but only {len(hard)} span 10+. The "
            f"5-year cases test mild ageing; the 10-year ones are where clustering actually "
            f"breaks. Results on cross-age drift will be weakly supported."
        )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="Validate without writing")
    parser.add_argument(
        "--force", action="store_true", help="Write even if validation reports problems"
    )
    args = parser.parse_args()

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    with store.open_index(db_path, read_only=True) as conn:
        rows = fetch_rows(conn)

    if not rows:
        console.print("[red]No labels yet. Run scripts/label_gold_set.py.[/red]")
        return 1

    labels = Counter(str(r["label"]) for r in rows)
    people = {str(r["person_id"]) for r in rows if r["label"] == "person"}

    table = Table(title="Gold set", header_style="bold")
    table.add_column("Label")
    table.add_column("Faces", justify="right")
    table.add_column("Share", justify="right")
    for label, count in labels.most_common():
        table.add_row(label, f"{count:,}", f"{100 * count / len(rows):5.1f}%")
    table.add_row("[bold]total[/bold]", f"[bold]{len(rows):,}[/bold]", "")
    console.print(table)
    console.print(f"Distinct identities: [bold]{len(people)}[/bold]")

    print_readability(rows, "quality_bucket", ("good", "marginal", "bad"))
    print_readability(rows, "size_bucket", ("large", "medium", "small", "tiny"))
    print_spans(rows)

    problems = validate(rows)
    if problems:
        console.print("\n[yellow]Validation warnings:[/yellow]")
        for problem in problems:
            console.print(f"  • {problem}")
    else:
        console.print("\n[green]Validation passed.[/green]")

    if args.check:
        return 0 if not problems else 2

    if problems and not args.force:
        console.print(
            "\n[red]Not writing.[/red] Keep labelling, or pass --force to export anyway "
            "and record the shortfall in DEVLOG.md."
        )
        return 2

    out_path = args.out or (paths.gold_dir() / "labels.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)  # type: ignore[arg-type]

    size_kb = out_path.stat().st_size / 1024
    console.print(f"\n[green]Wrote {len(rows):,} rows to {out_path}[/green] ({size_kb:.0f} KB)")
    console.print(
        "\n[bold]Freeze it now.[/bold] Editing the gold set after experiments begin makes "
        "results non-comparable.\nNext: src/faceindex/eval/metrics.py and run_experiment.py."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
