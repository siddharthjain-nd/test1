"""Command line entry point."""

from __future__ import annotations

import platform
import sys

import typer
from rich.console import Console
from rich.table import Table

from faceindex import __version__, paths

app = typer.Typer(add_completion=False, help="Offline face grouping for personal photo libraries.")
console = Console()


@app.command()
def env() -> None:
    """Report the runtime environment.

    Run this on every machine and compare. Differing versions change decoded pixels and
    therefore change embeddings, which makes experiment results incomparable (PLAN.md section 3).
    """
    table = Table(title=f"faceindex {__version__}", show_header=True, header_style="bold")
    table.add_column("Component")
    table.add_column("Value")

    table.add_row("platform", f"{platform.system()} {platform.release()}")
    table.add_row("machine", platform.machine())
    table.add_row("python", sys.version.split()[0])

    for module_name in (
        "numpy",
        "onnxruntime",
        "cv2",
        "PIL",
        "pillow_heif",
        "sklearn",
        "scipy",
        "pandas",
    ):
        try:
            module = __import__(module_name)
            table.add_row(module_name, getattr(module, "__version__", "unknown"))
        except ImportError:
            table.add_row(module_name, "[red]not installed[/red]")

    try:
        import onnxruntime as ort

        table.add_row("ort providers", ", ".join(ort.get_available_providers()))
    except ImportError:
        pass

    console.print(table)
    console.print(
        "\n[yellow]Reminder:[/yellow] only the CPUExecutionProvider may be used to produce "
        "stored embeddings. CoreML and CUDA do not agree bit-for-bit with it."
    )


@app.command(name="paths")
def show_paths() -> None:
    """Show every resolved filesystem location."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name")
    table.add_column("Path")
    table.add_column("Exists")

    entries = {
        "project_root": paths.project_root(),
        "models": paths.models_dir(),
        "data": paths.data_dir(),
        "crops": paths.crops_dir(),
        "context": paths.context_crops_dir(),
        "gold": paths.gold_dir(),
        "results": paths.results_dir(),
        "index_db": paths.index_db_path(),
        "configs": paths.configs_dir(),
    }
    for name, path in entries.items():
        table.add_row(name, str(path), "yes" if path.exists() else "no")

    console.print(table)


if __name__ == "__main__":
    app()
