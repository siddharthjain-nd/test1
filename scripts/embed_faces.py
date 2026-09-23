#!/usr/bin/env python3
"""Embed every aligned crop in the face pool with MobileFaceNet.

Cheap and repeatable: it reads the 112x112 crops, never the original photos, so it can be
re-run on either machine for free. That asymmetry is the whole reason detection was given
the large model and embedding the small one.

MobileFaceNet is deliberate here (Register C6). These vectors exist only to pre-group
crops for labelling; the gold set is *labels*, and everything is re-embedded in Phase 5.
Running ResNet100 at this stage turns a ~15 minute step into hours for zero benefit.

Resumable at face granularity. Interrupt it and rerun the identical command.

Usage
    python scripts/embed_faces.py
    python scripts/embed_faces.py --limit 500        # trial run first
    python scripts/embed_faces.py --stats            # report without embedding
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from faceindex import embed, paths, store

console = Console()


def print_stats(conn: object) -> None:
    """Coverage per model.

    Counting every embedding against the face count made "pending" go negative once two
    models were stored: 127,756 embeddings against 63,878 faces is complete twice over, not
    an overflow. Coverage only means anything per model.
    """
    total = conn.execute("SELECT COUNT(*) AS n FROM faces").fetchone()["n"]  # type: ignore[attr-defined]
    per_model = conn.execute(  # type: ignore[attr-defined]
        "SELECT model, COUNT(*) AS n FROM face_embeddings WHERE embed_version = ? "
        "GROUP BY model ORDER BY n DESC",
        (embed.EMBED_VERSION,),
    ).fetchall()

    table = Table(title=f"Embedding coverage — {total:,} faces in the pool", header_style="bold")
    table.add_column("Model")
    table.add_column("Embedded", justify="right")
    table.add_column("Pending", justify="right")
    table.add_column("", justify="left")

    if not per_model:
        table.add_row("(none yet)", "0", f"{total:,}", "")
    for entry in per_model:
        pending = total - int(entry["n"])
        table.add_row(
            str(entry["model"]),
            f"{int(entry['n']):,}",
            f"{pending:,}",
            "[green]complete[/green]" if pending <= 0 else "[yellow]incomplete[/yellow]",
        )
    console.print(table)

    provenance = conn.execute(  # type: ignore[attr-defined]
        "SELECT model, platform, onnxruntime_version, COUNT(*) AS n FROM face_embeddings "
        "GROUP BY model, platform, onnxruntime_version"
    ).fetchall()
    if provenance:
        prov = Table(title="Provenance", header_style="bold")
        for column in ("Model", "Platform", "ORT", "Rows"):
            prov.add_column(column, justify="right" if column == "Rows" else "left")
        for entry in provenance:
            prov.add_row(
                entry["model"],
                entry["platform"],
                entry["onnxruntime_version"],
                f"{entry['n']:,}",
            )
        console.print(prov)
        platforms = {(e["platform"], e["onnxruntime_version"]) for e in provenance}
        if len(platforms) > 1:
            console.print(
                "[yellow]Warning:[/yellow] embeddings were produced on more than one "
                "platform or runtime version. Those are not comparable to each other "
                "(PLAN.md section 3). Re-embed from a single machine."
            )
        if len({e["model"] for e in provenance}) > 1:
            console.print(
                "Several models are stored side by side. That is intended — choose which "
                "to score with [bold]run_experiment.py --model[/bold]."
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument(
        "--model",
        default="buffalo_sc/w600k_mbf.onnx",
        help="Relative to models/. Default is MobileFaceNet (Register C6).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Embed at most N faces")
    parser.add_argument(
        "--flip-tta",
        action="store_true",
        help="Average the crop and its mirror (Register C8). Off by default; 2x the cost.",
    )
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

    model_path = paths.models_dir() / args.model
    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/red]")
        console.print("Run: python scripts/download_models.py")
        return 1

    config = embed.EmbedConfig(
        model_path=model_path, flip_tta=args.flip_tta, num_threads=args.threads
    )

    with store.open_index(db_path) as conn:
        tasks = embed.pending_faces(conn, model_path.name, limit=args.limit)
        if not tasks:
            console.print("[green]Nothing to do: every face is already embedded.[/green]\n")
            print_stats(conn)
            return 0

        host, ort_version = embed.runtime_fingerprint()
        console.print(f"[bold]Model    :[/bold] {args.model}")
        console.print(f"[bold]Runtime  :[/bold] {host}, onnxruntime {ort_version}, CPU only")
        console.print(f"[bold]Flip TTA :[/bold] {'on' if args.flip_tta else 'off'}")
        console.print(f"[bold]Pending  :[/bold] {len(tasks):,} faces\n")

        embedder = embed.ArcFaceEmbedder(config)
        started = time.perf_counter()
        done = skipped = 0

        progress = tqdm(total=len(tasks), unit="face", smoothing=0.05)
        for batch_done, batch_skipped in embed.run(
            conn, tasks, embedder, batch_size=args.batch_size
        ):
            done += batch_done
            skipped += batch_skipped
            progress.update(batch_done + batch_skipped)
        progress.close()

        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0.0
        console.print(f"\nEmbedded {done:,} faces in {elapsed / 60:.1f} min ({rate:.0f} face/s).")
        if skipped:
            console.print(f"[yellow]{skipped:,} crop(s) unreadable and skipped.[/yellow]")

        print_stats(conn)

    console.print(
        f"\n[green]Embeddings done for {model_path.name}.[/green]\n"
        f"Score them with:\n"
        f"  python scripts/run_experiment.py --model {model_path.name} "
        f"--algorithm chinese_whispers --similarity-sweep 0.30,0.40,0.50,0.60\n"
        f"[dim](bootstrap_cluster.py was the Phase 1 pre-grouping step for labelling; "
        f"with a gold set in place, run_experiment.py is what produces numbers.)[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
