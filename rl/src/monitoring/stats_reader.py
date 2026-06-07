"""Offline analysis of training runs.

The training callbacks write per-episode rows to HDF5 (`runs/stats/*.h5`).
This module reads them back for inspection, rolling-statistic analysis,
multi-run comparison, and quick plots.

Columns produced by `EpisodeStatsCallback`:

    timestep        : env step when the episode ended
    episode_reward  : total reward over the episode
    episode_length  : number of env steps in the episode
    score           : env-defined score signal at episode end
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def read_stats(path: str | Path) -> pd.DataFrame:
    """Load a single stats file into a tidy DataFrame, sorted by timestep."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")
    with h5py.File(path, "r") as f:
        if "episodes" not in f:
            raise ValueError(f"No 'episodes' group in {path}")
        data = {k: f["episodes"][k][:] for k in f["episodes"].keys()}
    df = pd.DataFrame(data)
    if "timestep" in df.columns:
        df = df.sort_values("timestep").reset_index(drop=True)
    df["episode_idx"] = np.arange(len(df))
    return df


def read_runs(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load multiple runs and concatenate with a `run` column for grouping."""
    frames = []
    for p in paths:
        p = Path(p)
        df = read_stats(p)
        df["run"] = p.stem
        frames.append(df)
    if not frames:
        raise ValueError("read_runs received no paths")
    return pd.concat(frames, ignore_index=True)


def discover_runs(stats_dir: str | Path, pattern: str = "*.h5") -> list[Path]:
    """List `*.h5` files in a directory, excluding eval-callback npz files."""
    stats_dir = Path(stats_dir)
    return sorted(p for p in stats_dir.glob(pattern) if p.is_file())


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
@dataclass
class RunSummary:
    """Aggregated metrics for a single run."""

    name: str
    episodes: int
    total_steps: int
    mean_reward: float
    median_reward: float
    std_reward: float
    max_reward: float
    min_reward: float
    mean_length: float
    score_mean: float
    last_100_mean_reward: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "episodes": self.episodes,
            "total_steps": self.total_steps,
            "mean_reward": self.mean_reward,
            "median_reward": self.median_reward,
            "std_reward": self.std_reward,
            "max_reward": self.max_reward,
            "min_reward": self.min_reward,
            "mean_length": self.mean_length,
            "score_mean": self.score_mean,
            "last_100_mean_reward": self.last_100_mean_reward,
        }


def summarise(df: pd.DataFrame, name: str = "run") -> RunSummary:
    """Compute a single-run summary. `df` may come from `read_stats`."""
    if df.empty:
        raise ValueError(f"Cannot summarise empty DataFrame for '{name}'")

    tail = df.tail(100)
    return RunSummary(
        name=name,
        episodes=len(df),
        total_steps=int(df["timestep"].max()),
        mean_reward=float(df["episode_reward"].mean()),
        median_reward=float(df["episode_reward"].median()),
        std_reward=float(df["episode_reward"].std(ddof=0)),
        max_reward=float(df["episode_reward"].max()),
        min_reward=float(df["episode_reward"].min()),
        mean_length=float(df["episode_length"].mean()),
        score_mean=float(df["score"].mean()) if "score" in df else float("nan"),
        last_100_mean_reward=float(tail["episode_reward"].mean()),
    )


def summarise_all(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Summary table for many runs. Returns one row per run."""
    rows = []
    for p in paths:
        p = Path(p)
        df = read_stats(p)
        rows.append(summarise(df, name=p.stem).as_dict())
    return pd.DataFrame(rows).set_index("name")


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------
def rolling(
    df: pd.DataFrame,
    window: int = 100,
    columns: Iterable[str] = ("episode_reward", "episode_length", "score"),
) -> pd.DataFrame:
    """Add rolling-mean columns suffixed with `_rollN` for plotting."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[f"{col}_roll{window}"] = (
            out[col].rolling(window, min_periods=1).mean()
        )
    return out


def bin_by_step(
    df: pd.DataFrame,
    bin_size: int = 100_000,
    columns: Iterable[str] = ("episode_reward", "episode_length", "score"),
) -> pd.DataFrame:
    """Bin episodes by env timestep. Useful for comparing runs of different
    episode counts on an equal-step axis."""
    if "timestep" not in df.columns:
        raise ValueError("bin_by_step requires a 'timestep' column")

    bins = (df["timestep"] // bin_size) * bin_size
    cols = [c for c in columns if c in df.columns]
    grouped = df.assign(_bin=bins).groupby("_bin")[cols].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(c) for c in grouped.columns]
    grouped.index.name = "timestep_bin"
    return grouped.reset_index()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
try:
    import typer
    from rich.console import Console
    from rich.table import Table

    _console = Console()
    app = typer.Typer(help="Offline analysis of training run HDF5 stats files.")

    @app.command()
    def summary(
        path: Path = typer.Argument(..., exists=True, help="Path to a stats .h5 file"),
    ) -> None:
        """Print a one-run summary."""
        df = read_stats(path)
        s = summarise(df, name=path.stem)
        table = Table(title=f"Summary — {path.stem}")
        table.add_column("metric")
        table.add_column("value", justify="right")
        for k, v in s.as_dict().items():
            if isinstance(v, float):
                table.add_row(k, f"{v:.4f}")
            else:
                table.add_row(k, str(v))
        _console.print(table)

    @app.command()
    def compare(
        stats_dir: Path = typer.Argument(..., exists=True, help="Directory of *.h5 files"),
        pattern: str = typer.Option("*.h5", "--pattern"),
    ) -> None:
        """Print a comparison table over all runs in a directory."""
        paths = discover_runs(stats_dir, pattern)
        if not paths:
            _console.print(f"[yellow]No runs matching {pattern} in {stats_dir}[/yellow]")
            raise typer.Exit(code=1)
        df = summarise_all(paths)
        table = Table(title=f"Runs in {stats_dir} ({len(paths)})")
        table.add_column("run")
        for col in df.columns:
            table.add_column(col, justify="right")
        for name, row in df.iterrows():
            cells = [name]
            for v in row:
                cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
            table.add_row(*cells)
        _console.print(table)

    if __name__ == "__main__":
        app()

except ImportError:  # pragma: no cover - typer/rich are training-time deps
    pass
