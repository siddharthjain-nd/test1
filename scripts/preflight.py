#!/usr/bin/env python3
"""Everything that must be true before the test set is spent. One GO / NO-GO at the end.

The test set is read once. If it is read while something upstream is broken -- a half
finished embedding run, a split that overlaps, a results file whose columns have shifted --
the number that comes out is worthless and there is no second chance to notice.

Every check prints its own plain verdict as it runs. Exit code is 0 only on GO.

Usage
    python scripts/preflight.py --model w600k_r50.onnx --algorithm components --similarity 0.53
    python scripts/preflight.py --model w600k_r50.onnx --algorithm hdbscan --epsilon 0.95
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from rich.console import Console

from faceindex import cluster, embed, paths, store
from faceindex.eval.goldset import load_gold_set
from faceindex.eval.split import HOLDOUT, TUNE, load_split

console = Console()

PASS, WARN, FAIL = "pass", "warn", "fail"

# A score computed over very few people is mostly luck. Below this the test result cannot
# separate a real regression from noise, so it is not worth spending.
MIN_HOLDOUT_PEOPLE = 15
MIN_HOLDOUT_FACES = 150


class Report:
    """Collects verdicts and prints each one as it happens."""

    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str) -> None:
        self.results.append((status, name, detail))
        tag = {
            PASS: "[green]PASS[/green]",
            WARN: "[yellow]WARN[/yellow]",
            FAIL: "[red]FAIL[/red]",
        }[status]
        console.print(f"  {tag}  [bold]{name}[/bold] — {detail}")

    @property
    def failed(self) -> list[tuple[str, str, str]]:
        return [r for r in self.results if r[0] == FAIL]

    @property
    def warned(self) -> list[tuple[str, str, str]]:
        return [r for r in self.results if r[0] == WARN]


# ------------------------------------------------------------------------------------
# Checks
# ------------------------------------------------------------------------------------


def check_embeddings(report: Report, conn: object, model: str) -> np.ndarray | None:
    """Complete, from one machine, and numerically sane."""
    total = conn.execute("SELECT COUNT(*) AS n FROM faces").fetchone()["n"]  # type: ignore[attr-defined]
    stored = dict(cluster.available_models(conn))  # type: ignore[arg-type]

    if model not in stored:
        report.add(
            FAIL,
            "embeddings exist",
            f"nothing stored for {model}. Have: {sorted(stored) or 'none'}. "
            f"Run embed_faces.py --model <path> first.",
        )
        return None

    have = stored[model]
    if have < total:
        report.add(
            FAIL,
            "embeddings complete",
            f"{model} covers {have:,} of {total:,} faces — {total - have:,} missing. "
            f"Re-run embed_faces.py; it resumes where it stopped.",
        )
    else:
        report.add(PASS, "embeddings complete", f"{model} covers all {total:,} faces.")

    provenance = conn.execute(  # type: ignore[attr-defined]
        "SELECT DISTINCT platform, onnxruntime_version FROM face_embeddings WHERE model = ?",
        (model,),
    ).fetchall()
    if len(provenance) > 1:
        where = ", ".join(f"{p['platform']}/{p['onnxruntime_version']}" for p in provenance)
        report.add(
            FAIL,
            "one machine only",
            f"{model} was embedded on more than one setup ({where}). Those vectors are not "
            f"comparable to each other, so any score mixing them is meaningless.",
        )
    else:
        entry = provenance[0]
        report.add(
            PASS,
            "one machine only",
            f"all from {entry['platform']}, onnxruntime {entry['onnxruntime_version']}.",
        )

    _, matrix = cluster.load_embeddings(conn, model=model)  # type: ignore[arg-type]

    if matrix.shape[1] != embed.EMBED_DIM:
        report.add(FAIL, "vector shape", f"expected {embed.EMBED_DIM}-d, found {matrix.shape[1]}-d.")
        return matrix

    bad = int((~np.isfinite(matrix)).any(axis=1).sum())
    norms = np.linalg.norm(matrix, axis=1)
    off_unit = int((np.abs(norms - 1.0) > 1e-3).sum())
    dead = int((norms < 1e-6).sum())

    if bad or dead:
        report.add(
            FAIL,
            "vectors usable",
            f"{bad:,} contain NaN/inf and {dead:,} are all-zero. Those poison every distance "
            f"they appear in. Re-embed the affected faces.",
        )
    elif off_unit:
        report.add(
            FAIL,
            "vectors usable",
            f"{off_unit:,} are not unit length, so cosine and Euclidean no longer agree and "
            f"every threshold in the project means something different for them.",
        )
    else:
        report.add(
            PASS,
            "vectors usable",
            f"all {len(matrix):,} are {embed.EMBED_DIM}-d, finite and unit length.",
        )
    return matrix


def check_gold_and_split(report: Report) -> tuple[object, object] | None:
    gold_path = paths.gold_dir() / "labels.csv"
    split_path = paths.gold_dir() / "split.csv"

    if not gold_path.exists():
        report.add(FAIL, "gold set present", f"no {gold_path}. Run export_gold_set.py.")
        return None
    if not split_path.exists():
        report.add(FAIL, "split present", f"no {split_path}. Run make_holdout.py.")
        return None

    gold = load_gold_set(gold_path)
    split = load_split(split_path)
    report.add(
        PASS,
        "gold set present",
        f"{len(gold.identities):,} identity faces across {gold.n_people} people.",
    )

    labelled = {person for person in gold.identities.values()}
    unassigned = labelled - set(split.assignment)
    if unassigned:
        report.add(
            FAIL,
            "everyone is assigned",
            f"{len(unassigned)} labelled people are missing from split.csv "
            f"(e.g. {sorted(unassigned)[:3]}). They silently default to tune, which quietly "
            f"changes what was measured. Re-run make_holdout.py.",
        )
    else:
        report.add(PASS, "everyone is assigned", f"all {len(labelled)} people appear in split.csv.")

    tune_people = split.people(TUNE) & labelled
    hold_people = split.people(HOLDOUT) & labelled
    overlap = tune_people & hold_people
    if overlap:
        report.add(
            FAIL,
            "sides are disjoint",
            f"{len(overlap)} people are on both sides. The test set is already contaminated; "
            f"the split must be rebuilt.",
        )
    else:
        report.add(
            PASS,
            "sides are disjoint",
            f"{len(tune_people)} people for tuning, {len(hold_people)} reserved, no overlap.",
        )

    counts = Counter(gold.identities.values())
    hold_faces = sum(counts[p] for p in hold_people)
    tune_faces = sum(counts[p] for p in tune_people)

    if len(hold_people) < MIN_HOLDOUT_PEOPLE or hold_faces < MIN_HOLDOUT_FACES:
        report.add(
            WARN,
            "test set big enough",
            f"{len(hold_people)} people / {hold_faces:,} faces is small (want at least "
            f"{MIN_HOLDOUT_PEOPLE} / {MIN_HOLDOUT_FACES}). The result will be noisy — read it "
            f"against the tolerance from estimate_error_bar.py, not as an exact figure.",
        )
    else:
        report.add(
            PASS,
            "test set big enough",
            f"{len(hold_people)} people / {hold_faces:,} faces reserved.",
        )

    def median_faces(people: set[str]) -> float:
        values = sorted(counts[p] for p in people)
        return float(np.median(values)) if values else 0.0

    tune_median, hold_median = median_faces(tune_people), median_faces(hold_people)
    if tune_median and hold_median:
        ratio = max(tune_median, hold_median) / min(tune_median, hold_median)
        if ratio > 2.0:
            report.add(
                WARN,
                "sides look alike",
                f"typical person has {tune_median:.0f} faces in tuning but {hold_median:.0f} "
                f"in the test set. The two sides are not comparable, so a difference in score "
                f"may just be that difference.",
            )
        else:
            report.add(
                PASS,
                "sides look alike",
                f"typical person has {tune_median:.0f} faces in tuning, {hold_median:.0f} in "
                f"the test set — close enough to compare.",
            )
    console.print(
        f"         [dim]tuning: {len(tune_people)} people / {tune_faces:,} faces · "
        f"reserved: {len(hold_people)} people / {hold_faces:,} faces[/dim]"
    )
    return gold, split


def check_results_file(report: Report, model: str, algorithm: str, setting: float) -> None:
    """Results readable, and -- the important one -- the test set not already spent."""
    path = paths.results_dir() / "results.csv"
    if not path.exists():
        report.add(
            WARN, "results file", f"no {path} yet, so nothing to cross-check the choice against."
        )
        return

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    broken = [r for r in rows if not r.get("bcubed_f1") or not r.get("split")]
    if broken:
        report.add(
            FAIL,
            "results readable",
            f"{len(broken)} of {len(rows)} rows have shifted or empty columns. Run "
            f"run_experiment.py --prune before trusting anything in this file.",
        )
    else:
        report.add(PASS, "results readable", f"{len(rows)} rows, all columns intact.")

    spent = [r for r in rows if r.get("split") == HOLDOUT]
    if spent:
        labels = ", ".join(sorted({r.get("label", "?") for r in spent})[:4])
        report.add(
            FAIL,
            "test set unspent",
            f"{len(spent)} run(s) already scored against the test set ({labels}). It is no "
            f"longer a clean test — every later choice was made knowing its answer. Either "
            f"accept it as a second tuning set, or reserve fresh identities.",
        )
    else:
        report.add(PASS, "test set unspent", "no run has ever scored against the test set.")

    dev = [
        r
        for r in rows
        if r.get("split") == TUNE and r.get("bcubed_f1") and r.get("embedder") and r.get("algorithm")
    ]
    if not dev:
        report.add(WARN, "choice is the dev winner", "no usable tuning rows to compare against.")
        return

    def f1(row: dict[str, str]) -> float:
        try:
            return float(row["bcubed_f1"])
        except (TypeError, ValueError):
            return -1.0

    best = max(dev, key=f1)
    field = "epsilon" if algorithm == "hdbscan" else "threshold"
    # run_experiment.py records --model verbatim in "embedder"; the table display shortens
    # it for width, so compare against both spellings.
    short = model.replace("w600k_", "").replace(".onnx", "")
    matched = [
        r
        for r in dev
        if str(r.get("embedder", "")) in (model, short)
        and r.get("algorithm") == algorithm
        and _close(r.get(field), setting)
    ]

    if not matched:
        report.add(
            WARN,
            "choice was measured",
            f"no tuning row for {model} / {algorithm} / {field} {setting}. You are about to "
            f"spend the test set on a setting never scored on the dev set.",
        )
        return

    mine = max(matched, key=f1)
    gap = f1(best) - f1(mine)
    detail = (
        f"chosen scores {f1(mine):.4f} on the dev set; best dev row is "
        f"{f1(best):.4f} ({best.get('embedder')}/{best.get('algorithm')})."
    )
    if gap > 0.02:
        report.add(WARN, "choice is near the dev winner", detail + " That is a real gap — confirm it is deliberate.")
    else:
        report.add(PASS, "choice is near the dev winner", detail)


def _close(value: object, target: float, tol: float = 1e-6) -> bool:
    try:
        return abs(float(str(value)) - target) < tol
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--model", required=True, help="e.g. w600k_r50.onnx")
    parser.add_argument(
        "--algorithm", default="components", choices=["hdbscan", "components", "chinese_whispers"]
    )
    parser.add_argument("--similarity", type=float, default=0.53, help="graph algorithms")
    parser.add_argument("--epsilon", type=float, default=0.95, help="hdbscan")
    args = parser.parse_args()

    setting = args.epsilon if args.algorithm == "hdbscan" else args.similarity

    db_path = args.db or paths.index_db_path()
    if not db_path.exists():
        console.print(f"[red]No index at {db_path}.[/red]")
        return 1

    console.print(
        f"[bold]Preflight[/bold] for {args.model} / {args.algorithm} / "
        f"{'epsilon' if args.algorithm == 'hdbscan' else 'similarity'} {setting}\n"
    )

    report = Report()
    console.print("[bold]Embeddings[/bold]")
    with store.open_index(db_path, read_only=True) as conn:
        check_embeddings(report, conn, args.model)

    console.print("\n[bold]Gold set and split[/bold]")
    check_gold_and_split(report)

    console.print("\n[bold]Results history[/bold]")
    check_results_file(report, args.model, args.algorithm, setting)

    console.print()
    if report.failed:
        console.print(
            f"[bold red]NO-GO — {len(report.failed)} check(s) failed.[/bold red] "
            f"Do not run the test set yet."
        )
        for _, name, _ in report.failed:
            console.print(f"  [red]•[/red] {name}")
        console.print("\nFix those, re-run this, and only then spend the test set.")
        return 1

    if report.warned:
        console.print(
            f"[bold yellow]GO, with {len(report.warned)} warning(s).[/bold yellow] "
            f"Nothing is broken, but read these before spending the one test run:"
        )
        for _, name, detail in report.warned:
            console.print(f"  [yellow]•[/yellow] {name}: {detail}")
        return 0

    console.print(
        "[bold green]GO — every check passed.[/bold green] The pipeline is sound, the test "
        "set is untouched, and the chosen setting was measured on the dev set."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
