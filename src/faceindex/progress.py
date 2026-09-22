"""Live progress for work that cannot be subdivided.

Countable work gets a `tqdm` bar — see the face pool and embedding passes. But some steps are
a single opaque call into a library: clustering 64,000 faces is one `fit_predict` that
returns after an hour with nothing in between. A script that prints nothing for that long is
indistinguishable from one that has hung.

So the elapsed time is shown live, along with an estimate where a previous run gives one to
work from. The estimate is explicitly labelled as such: a bar that quietly runs past 100% is
worse than no bar, because it reads as a stall.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

T = TypeVar("T")


def run_with_progress(
    work: Callable[[], T],
    description: str,
    *,
    estimate_seconds: float | None = None,
    console: Console | None = None,
    poll: float = 0.5,
) -> T:
    """Run ``work`` on a worker thread while showing elapsed time on the main one.

    ``estimate_seconds`` drives the bar when a previous run supplies one. Without it only the
    spinner and elapsed clock are shown, which is honest about not knowing rather than
    inventing a figure.
    """
    columns: list[object] = [SpinnerColumn(), TextColumn("[bold]{task.description}")]
    if estimate_seconds:
        columns += [
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}% of estimate"),
        ]
    columns += [TextColumn("·"), TimeElapsedColumn()]

    with Progress(*columns, console=console, transient=False) as progress:  # type: ignore[arg-type]
        task = progress.add_task(description, total=estimate_seconds or None)
        started = time.perf_counter()

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(work)
            while not future.done():
                if estimate_seconds:
                    elapsed = time.perf_counter() - started
                    # Hold at 99% rather than overrunning: a full bar that keeps going looks
                    # like a stall, and the estimate is only ever approximate.
                    progress.update(task, completed=min(elapsed, estimate_seconds * 0.99))
                time.sleep(poll)

            if estimate_seconds:
                progress.update(task, completed=estimate_seconds)
            return future.result()


def format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"
