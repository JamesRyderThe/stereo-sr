from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType

import torch
from rich.console import Console as RichConsole
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskID, TextColumn
from rich.table import Table
from rich.text import Text


def _grid(*styles: str) -> Table:
    t = Table.grid(padding=(0, 2))
    for s in styles:
        t.add_column(style=s)
    return t


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins:02d}m"


class TrainingProgress:
    def __init__(self, console: Console, total_iters: int, initial_step: int = 0) -> None:
        self._console = console
        self._total = total_iters
        self._start_time = 0.0
        self._last_step = initial_step
        self._last_time = 0.0
        self._iter_per_sec = 0.0
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._step = initial_step
        self._loss = 0.0
        self._lr = 0.0
        self._val_psnr: float | None = None
        self._best_psnr: float | None = None

    def __enter__(self) -> TrainingProgress:
        if not self._console.enabled:
            return self
        self._start_time = time.perf_counter()
        self._last_time = self._start_time
        self._progress = Progress(
            TextColumn("[cyan]SISSR"),
            BarColumn(bar_width=40, style="dim", complete_style="cyan"),
            TextColumn("[dim]{task.percentage:>5.1f}%"),
            TextColumn("[dim]{task.completed:>7,}/{task.total:,}"),
            console=self._console._rich,
            expand=False,
        )
        self._task_id = self._progress.add_task("train", total=self._total, completed=self._step)
        self._live = Live(
            self._build_display(),
            console=self._console._rich,
            refresh_per_second=4,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is None and self._step > 0:
            self._console._rich.print(self._build_completion())

    def update(self, step: int, loss: float, lr: float) -> None:
        now = time.perf_counter()
        if step > self._last_step:
            dt = now - self._last_time
            if dt > 0:
                rate = (step - self._last_step) / dt
                self._iter_per_sec = 0.9 * self._iter_per_sec + 0.1 * rate
            self._last_step = step
            self._last_time = now

        self._step, self._loss, self._lr = step, loss, lr
        if not self._console.enabled:
            return
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=step)
        if self._live is not None and step % 10 == 0:
            self._live.update(self._build_display())

    def set_val_metrics(self, psnr: float, best_psnr: float) -> None:
        self._val_psnr = psnr
        self._best_psnr = best_psnr

    def _build_display(self) -> Panel:
        parts: list[Text | Progress | Table] = []
        if self._progress:
            parts.append(self._progress)

        elapsed = time.perf_counter() - self._start_time
        remaining = (self._total - self._step) / max(self._iter_per_sec, 0.01)

        timing = _grid("dim", "bold", "dim", "bold", "dim", "bold")
        timing.add_row(
            "Elapsed",
            _fmt_time(elapsed),
            "ETA",
            _fmt_time(remaining) if self._iter_per_sec > 0 else "--",
            "Speed",
            f"{self._iter_per_sec:.1f} it/s" if self._iter_per_sec > 0 else "--",
        )
        parts.append(timing)

        if self._step > 0:
            parts.append(Text())
            metrics = _grid("dim", "bold", "dim", "bold")
            metrics.add_row("Loss", f"{self._loss:>10.4f}", "LR", f"{self._lr:>10.2e}")
            if self._val_psnr is not None:
                metrics.add_row(
                    "Val PSNR",
                    f"{self._val_psnr:>10.2f} dB",
                    "Best PSNR",
                    f"{self._best_psnr:>10.2f} dB" if self._best_psnr else "",
                )
            parts.append(metrics)

            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024**3
                peak = torch.cuda.max_memory_allocated() / 1024**3
                parts.extend([Text(), Text(f"VRAM: {mem:.1f} / {peak:.1f} GB peak", style="dim")])

        return Panel(
            Group(*parts), title="[cyan]SISSR Training", border_style="dim", padding=(0, 1)
        )

    def _build_completion(self) -> Panel:
        elapsed = time.perf_counter() - self._start_time
        speed = (self._step) / elapsed if elapsed > 0 else 0
        grid = _grid("dim", "bold", "dim", "bold")
        grid.add_row("Iterations", f"{self._step:,}", "Time", _fmt_time(elapsed))
        grid.add_row(
            "Final Loss",
            f"{self._loss:.4f}",
            "Best PSNR",
            f"{self._best_psnr:.2f} dB" if self._best_psnr else "--",
        )
        grid.add_row("Speed", f"{speed:.1f} it/s", "", "")
        return Panel(grid, title="[green]Training Complete", border_style="green")


class Console:
    def __init__(self, enabled: bool = True) -> None:
        self._rich = RichConsole(quiet=not enabled, width=120, force_terminal=True)
        self.enabled = enabled

    @contextmanager
    def spinner(self, message: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        with self._rich.status(f"[cyan]{message}", spinner="dots"):
            yield

    @contextmanager
    def training_progress(
        self, total_iters: int, initial_step: int = 0
    ) -> Iterator[TrainingProgress]:
        progress = TrainingProgress(self, total_iters, initial_step)
        with progress:
            yield progress

    def config_summary(
        self,
        model_name: str,
        params: int,
        total_iters: int,
        batch_size: int,
        accum_steps: int,
        num_gpus: int,
        compile_mode: str,
        precision: str,
    ) -> None:
        if not self.enabled:
            return
        grid = _grid("dim", "bold")
        grid.add_row("Model", f"{model_name} ({params / 1e6:.1f}M params)")
        grid.add_row("Iterations", f"{total_iters:,}")
        effective = batch_size * accum_steps * num_gpus
        grid.add_row(
            "Effective batch",
            f"{batch_size} × {accum_steps} accum × {num_gpus} GPU = {effective}",
        )
        grid.add_row("Compile", compile_mode)
        grid.add_row("Precision", precision)
        self._rich.print(
            Panel(grid, title="[cyan]Configuration", border_style="dim", padding=(0, 1))
        )

    def print(self, message: str, style: str | None = None) -> None:
        if self.enabled:
            self._rich.print(message, style=style)
