"""Live terminal monitor for an ongoing training run.

Polls the HDF5 stats file written by `EpisodeStatsCallback` and prints a
rolling summary table every N seconds. Run this in a second terminal while
`train.py` is running:

    python live_monitor.py runs/stats/run_s1_<timestamp>.h5

Options
-------
--interval   Refresh interval in seconds (default: 10)
--window     Rolling window for mean/std (default: 100)
--follow     Keep polling until Ctrl-C even after the file stops growing
             (useful to leave open after training ends for a final look)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

# Disable HDF5's advisory lock so this poller never holds a read lock that blocks the
# trainer's concurrent append. Must precede the h5py import below.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

try:
    import h5py
    import pandas as pd
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}. Run: pip install h5py pandas") from e

console = Console()
app = typer.Typer(add_completion=False)


def _read_hdf5(path: Path) -> pd.DataFrame | None:
    """Read episodes group from HDF5. Returns None if the file isn't ready yet."""
    try:
        with h5py.File(path, "r") as f:
            if "episodes" not in f:
                return None
            data = {k: f["episodes"][k][:] for k in f["episodes"].keys()}
        df = pd.DataFrame(data)
        if df.empty:
            return None
        if "timestep" in df.columns:
            df = df.sort_values("timestep").reset_index(drop=True)
        return df
    except (OSError, RuntimeError, ValueError):
        return None


def _make_table(path: Path, df: pd.DataFrame, window: int, interval: int) -> Table:
    n = len(df)
    tail = df.tail(window)

    reward_mean = float(tail["episode_reward"].mean())
    reward_std = float(tail["episode_reward"].std(ddof=0))
    reward_max = float(df["episode_reward"].max())
    length_mean = float(tail["episode_length"].mean())
    score_mean = float(tail["score"].mean()) if "score" in tail.columns else float("nan")
    last_step = int(df["timestep"].max()) if "timestep" in df.columns else 0

    table = Table(
        title=f"[bold cyan]{path.stem}[/bold cyan]  "
              f"[dim](refresh every {interval}s · window={window})[/dim]",
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )
    table.add_column("metric", style="dim", width=24)
    table.add_column("value", justify="right", width=16)

    def _fmt(v: float, fmt: str = ".3f") -> str:
        return "—" if np.isnan(v) else f"{v:{fmt}}"

    table.add_row("episodes total", f"{n:,}")
    table.add_row("env steps", f"{last_step:,}")
    table.add_row("─" * 22, "─" * 14)
    table.add_row(f"reward mean (last {window})", _fmt(reward_mean))
    table.add_row(f"reward std  (last {window})", _fmt(reward_std))
    table.add_row("reward max (all time)", _fmt(reward_max))
    table.add_row(f"ep length mean (last {window})", _fmt(length_mean, ".1f"))
    table.add_row(f"score mean (last {window})", _fmt(score_mean))

    # Trend arrow: compare last window/2 to previous window/2
    half = max(window // 2, 1)
    if n >= window:
        prev_mean = float(df["episode_reward"].iloc[-window:-half].mean())
        curr_mean = float(df["episode_reward"].iloc[-half:].mean())
        delta = curr_mean - prev_mean
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "→")
        colour = "green" if delta > 0 else ("red" if delta < 0 else "yellow")
        table.add_row("trend (Δ reward)", Text(f"{arrow} {delta:+.3f}", style=colour))

    return table


@app.command()
def monitor(
    stats_path: Path = typer.Argument(..., help="Path to the .h5 stats file"),
    interval: int = typer.Option(10, "--interval", "-i", help="Refresh interval (seconds)"),
    window: int = typer.Option(100, "--window", "-w", help="Rolling window size"),
    follow: bool = typer.Option(True, "--follow/--no-follow", help="Keep polling after training ends"),
) -> None:
    """Live-tail a training run's HDF5 stats file in the terminal."""

    console.print(f"[bold]Monitoring[/bold] {stats_path}  "
                  f"[dim](Ctrl-C to quit)[/dim]")

    if not stats_path.exists():
        console.print(f"[yellow]Waiting for {stats_path} to appear…[/yellow]")
        while not stats_path.exists():
            time.sleep(2)

    with Live(console=console, refresh_per_second=1, screen=False) as live:
        prev_n = 0
        stale_ticks = 0

        while True:
            df = _read_hdf5(stats_path)

            if df is None:
                live.update(Text("[yellow]Waiting for first episodes…[/yellow]"))
            else:
                curr_n = len(df)
                if curr_n == prev_n:
                    stale_ticks += 1
                else:
                    stale_ticks = 0
                prev_n = curr_n

                table = _make_table(stats_path, df, window, interval)

                if stale_ticks > 0:
                    staleness = f"  [dim](no new episodes for {stale_ticks * interval}s)[/dim]"
                    table.title += staleness  # type: ignore[operator]

                live.update(table)

                if stale_ticks >= 6 and not follow:
                    console.print("[green]No new data for 60 s — looks like training ended.[/green]")
                    break

            time.sleep(interval)


if __name__ == "__main__":
    app()
