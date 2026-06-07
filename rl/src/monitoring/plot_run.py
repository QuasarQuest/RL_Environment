"""Post-run matplotlib analysis.

Single run:
    python plot_run.py runs/stats/run_s1_<timestamp>.h5

Compare all runs in a directory:
    python plot_run.py --compare runs/stats/

Save to PNG:
    python plot_run.py runs/stats/run.h5 --out report.png
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False)

try:
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from monitoring.stats_reader import read_stats
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}. Run: pip install matplotlib pandas h5py") from e


def _rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def _plot_single(df: pd.DataFrame, name: str, window: int, out: Optional[Path]) -> None:
    cols = ["episode_reward", "episode_length", "score"]
    titles = ["Episode Reward", "Episode Length", "Score"]
    present = [c for c in cols if c in df.columns]
    n_panels = len(present)

    n_rows = max(n_panels // 2 + n_panels % 2, 1)
    fig = plt.figure(figsize=(14, 3.5 * n_rows))
    fig.suptitle(f"Training run: {name}", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.45, wspace=0.3)

    x = df["timestep"] if "timestep" in df.columns else df.index

    for panel, col in enumerate(present):
        title = titles[cols.index(col)]
        row, col_pos = divmod(panel, 2)
        ax = fig.add_subplot(gs[row, col_pos])
        y = df[col]

        ax.plot(x, y, alpha=0.2, color="steelblue", linewidth=0.8, label="raw")
        rolled = _rolling(y, window)
        ax.plot(x, rolled, color="steelblue", linewidth=1.8, label=f"rolling-{window}")

        roll_std = y.rolling(window, min_periods=1).std(ddof=0).fillna(0)
        ax.fill_between(x, rolled - roll_std, rolled + roll_std, alpha=0.15, color="steelblue")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("env step", fontsize=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="upper left")

    # Summary box in empty slot when n_panels is odd
    if n_panels % 2 == 1 and n_panels > 1:
        row, col_pos = divmod(n_panels, 2)
        ax = fig.add_subplot(gs[row, col_pos])
        ax.axis("off")
        tail = df["episode_reward"].tail(window)
        lines = [
            f"Episodes : {len(df):,}",
            f"Steps    : {int(df['timestep'].max()):,}" if "timestep" in df.columns else "",
            "",
            "Reward",
            f"  mean     : {df['episode_reward'].mean():.3f}",
            f"  max      : {df['episode_reward'].max():.3f}",
            f"  last {window} : {tail.mean():.3f}",
        ]
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

    _finish(fig, out)


def _plot_compare(paths: list[Path], window: int, metric: str, out: Optional[Path]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, p in enumerate(paths):
        try:
            df = read_stats(p)
        except Exception as exc:
            print(f"  skipping {p.name}: {exc}")
            continue
        if metric not in df.columns:
            print(f"  skipping {p.name}: column '{metric}' not found")
            continue
        x = df["timestep"] if "timestep" in df.columns else df.index
        ax.plot(x, _rolling(df[metric], window), label=p.stem,
                color=cmap[i % len(cmap)], linewidth=1.8)

    ax.set_title(f"Run comparison — {metric} (rolling-{window} mean)", fontsize=12)
    ax.set_xlabel("env step")
    ax.set_ylabel(metric)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    _finish(fig, out)


def _finish(fig, out: Optional[Path]) -> None:
    fig.tight_layout()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


@app.command()
def plot(
    path: Path = typer.Argument(..., help="Path to a .h5 file, or directory for --compare"),
    compare: bool = typer.Option(False, "--compare", "-c", help="Compare all runs in directory"),
    window: int = typer.Option(100, "--window", "-w", help="Rolling window size"),
    metric: str = typer.Option("episode_reward", "--metric", "-m", help="Metric for compare mode"),
    pattern: str = typer.Option("*.h5", "--pattern", help="Glob pattern for compare mode"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Save PNG instead of showing"),
) -> None:
    """Plot training stats — single run or multi-run comparison."""
    if compare or path.is_dir():
        paths = sorted(p for p in path.glob(pattern) if p.is_file())
        if not paths:
            raise typer.BadParameter(f"No files matching {pattern} in {path}")
        print(f"Comparing {len(paths)} runs…")
        _plot_compare(paths, window, metric, out)
    else:
        if not path.exists():
            raise typer.BadParameter(f"File not found: {path}")
        print(f"Loading {path.name}…")
        df = read_stats(path)
        _plot_single(df, path.stem, window, out)


if __name__ == "__main__":
    app()
