#!/usr/bin/env python3
"""
ATB run analysis — loads stats.h5 + evaluations.npz and saves matplotlib plots.

Usage
-----
# Auto-discover all runs in runs/:
    python rl/scripts/plot_runs.py

# All runs whose name starts with "seq":
    python rl/scripts/plot_runs.py --prefix seq

# Specific run directories:
    python rl/scripts/plot_runs.py --dirs runs/run_s1_XYZ runs/run_s2_ABC

# Save without opening a GUI window:
    python rl/scripts/plot_runs.py --prefix seq --no-show

Plots saved to  runs/<run_dir>/plots/*.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import NamedTuple

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"

# ── Style ─────────────────────────────────────────────────────────────────────

STAGE_COLORS = {
    1: "#4CAF50",
    2: "#2196F3",
    3: "#FF9800",
    4: "#9C27B0",
    5: "#F44336",
    6: "#00BCD4",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#ccc",
    "axes.titlecolor": "#eee",
    "axes.grid": True,
    "grid.color": "#333",
    "grid.linewidth": 0.5,
    "xtick.color": "#999",
    "ytick.color": "#999",
    "text.color": "#ccc",
    "legend.facecolor": "#1a1a2e",
    "legend.edgecolor": "#444",
    "legend.framealpha": 0.9,
    "lines.linewidth": 1.6,
    "font.size": 9,
})

# ── Data classes ──────────────────────────────────────────────────────────────


class EpisodeData(NamedTuple):
    timestep: np.ndarray   # (N,) — timestep each episode ended at
    reward: np.ndarray     # (N,)
    score: np.ndarray      # (N,)
    win: np.ndarray        # (N,) int 0/1
    ep_length: np.ndarray  # (N,)


class EvalData(NamedTuple):
    timesteps: np.ndarray   # (K,)
    mean: np.ndarray        # (K,)
    std: np.ndarray         # (K,)
    ep_lengths: np.ndarray  # (K,)


class RunInfo(NamedTuple):
    path: Path
    name: str
    stage: int
    episodes: EpisodeData | None
    evals: EvalData | None


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_episodes(run_dir: Path) -> EpisodeData | None:
    h5 = run_dir / "stats.h5"
    if not h5.exists():
        return None
    with h5py.File(h5, "r") as f:
        g = f["episodes"]
        return EpisodeData(
            timestep=g["timestep"][:],
            reward=g["episode_reward"][:],
            score=g["score"][:],
            win=g["win"][:],
            ep_length=g["episode_length"][:],
        )


def load_evals(run_dir: Path) -> EvalData | None:
    npz = run_dir / "eval_log" / "evaluations.npz"
    if not npz.exists():
        return None
    d = np.load(npz)
    results = d["results"]  # shape (K, n_eval_eps)
    return EvalData(
        timesteps=d["timesteps"],
        mean=results.mean(axis=1),
        std=results.std(axis=1),
        ep_lengths=d["ep_lengths"].mean(axis=1),
    )


_STAGE_RE = re.compile(r"_s(\d+)_")


def stage_from_name(name: str) -> int:
    m = _STAGE_RE.search(name)
    return int(m.group(1)) if m else 0


def discover_runs(
    run_dirs: list[Path] | None = None,
    prefix: str | None = None,
) -> list[RunInfo]:
    if run_dirs:
        dirs = [Path(d) for d in run_dirs]
    else:
        pattern = f"{prefix}_s*_*" if prefix else "*_s*_*"
        dirs = sorted(
            [d for d in RUNS_DIR.glob(pattern) if d.is_dir()],
            key=lambda p: (stage_from_name(p.name), int(p.name.rsplit("_", 1)[-1])),
        )

    runs: list[RunInfo] = []
    for d in dirs:
        if not d.is_dir():
            print(f"  [skip] {d} — not a directory")
            continue
        stage = stage_from_name(d.name)
        episodes = load_episodes(d)
        evals = load_evals(d)
        if episodes is None and evals is None:
            print(f"  [skip] {d.name} — no stats.h5 or evaluations.npz")
            continue
        runs.append(RunInfo(path=d, name=d.name, stage=stage, episodes=episodes, evals=evals))
        print(f"  loaded  {d.name}  (stage {stage})")

    return runs


# ── Smoothing ─────────────────────────────────────────────────────────────────

def _rolling(
    x: np.ndarray, y: np.ndarray, window: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling mean ± std over *window* samples. Returns (x_out, mean, std)."""
    n = len(y)
    if n < window:
        window = max(1, n)
    idx = np.arange(window - 1, n)
    means = np.array([y[i - window + 1:i + 1].mean() for i in idx])
    stds = np.array([y[i - window + 1:i + 1].std() for i in idx])
    return x[idx], means, stds


def _format_millions(ax: plt.Axes) -> None:
    """Format x-axis tick labels as '1.0M', '500K', etc."""
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K")
    )


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  saved  {path}")
    if show:
        plt.show()
    plt.close(fig)


# ── Figure 1: Per-run learning curves ────────────────────────────────────────

def plot_learning_curves(runs: list[RunInfo], show: bool) -> None:
    """One row per run: rolling reward | rolling score | eval reward."""
    n = len(runs)
    fig, axes = plt.subplots(n, 3, figsize=(14, 3.5 * n), squeeze=False)
    fig.suptitle("Learning Curves — per run", fontsize=12, color="#eee", y=1.01)

    for row, info in enumerate(runs):
        color = STAGE_COLORS.get(info.stage, "#aaa")
        ax_r, ax_s, ax_e = axes[row]

        ax_r.set_title(f"{info.name}\nEpisode Reward (rolling 100)", fontsize=8)
        if info.episodes is not None:
            ts, mu, sd = _rolling(info.episodes.timestep, info.episodes.reward)
            ax_r.plot(ts, mu, color=color)
            ax_r.fill_between(ts, mu - sd, mu + sd, color=color, alpha=0.15)
        ax_r.set_ylabel("Reward")
        _format_millions(ax_r)

        ax_s.set_title("Score (rolling 100)", fontsize=8)
        if info.episodes is not None:
            ts, mu, sd = _rolling(info.episodes.timestep, info.episodes.score)
            ax_s.plot(ts, mu, color=color)
            ax_s.fill_between(ts, mu - sd, mu + sd, color=color, alpha=0.15)
        ax_s.set_ylabel("Score")
        _format_millions(ax_s)

        ax_e.set_title("Eval Reward (mean ± std)", fontsize=8)
        if info.evals is not None:
            ev = info.evals
            ax_e.plot(ev.timesteps, ev.mean, color=color, marker="o", markersize=3)
            ax_e.fill_between(ev.timesteps, ev.mean - ev.std, ev.mean + ev.std,
                              color=color, alpha=0.20)
        ax_e.set_ylabel("Eval Reward")
        _format_millions(ax_e)

        for ax in (ax_r, ax_s, ax_e):
            ax.set_xlabel("Timestep")

    fig.tight_layout()

    for info in runs:
        _save(fig, info.path / "plots" / "learning_curves.png", show=False)
    if show:
        plt.show()
    plt.close(fig)


# ── Figure 2: Cross-stage comparison ─────────────────────────────────────────

def plot_stage_comparison(runs: list[RunInfo], show: bool, prefix: str) -> None:
    """One figure summarising final performance across stages."""
    if len(runs) < 2:
        return

    stages = [r.stage for r in runs]
    colors = [STAGE_COLORS.get(s, "#aaa") for s in stages]
    labels = [f"S{r.stage}" for r in runs]

    final_eval_mean = [r.evals.mean[-1] if r.evals is not None else np.nan for r in runs]
    final_eval_std = [r.evals.std[-1] if r.evals is not None else 0.0 for r in runs]

    def tail_scores(r: RunInfo) -> np.ndarray:
        if r.episodes is None:
            return np.array([np.nan])
        n = max(1, len(r.episodes.score) // 10)
        return r.episodes.score[-n:]

    tail_score_data = [tail_scores(r) for r in runs]
    final_score_mean = [d.mean() for d in tail_score_data]
    final_score_std = [d.std() for d in tail_score_data]

    def tail_win_rate(r: RunInfo) -> float:
        if r.episodes is None:
            return np.nan
        n = max(1, len(r.episodes.win) // 10)
        return float(r.episodes.win[-n:].mean())

    win_rates = [tail_win_rate(r) for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Cross-Stage Comparison (final performance)", fontsize=11, color="#eee")

    x = np.arange(len(runs))

    ax = axes[0]
    bars = ax.bar(x, final_eval_mean, yerr=final_eval_std, color=colors,
                  error_kw={"ecolor": "#888", "capsize": 4}, width=0.6)
    ax.set_title("Final Eval Reward (mean ± std)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Eval Reward")
    for bar, v in zip(bars, final_eval_mean):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7, color="#eee")

    ax = axes[1]
    bars = ax.bar(x, final_score_mean, yerr=final_score_std, color=colors,
                  error_kw={"ecolor": "#888", "capsize": 4}, width=0.6)
    ax.set_title("Final Score (last 10 % of episodes, mean ± std)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    for bar, v in zip(bars, final_score_mean):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#eee")

    ax = axes[2]
    bars = ax.bar(x, [w * 100 for w in win_rates], color=colors, width=0.6)
    ax.set_title("Win Rate (last 10 % of episodes, %)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Win %")
    ax.set_ylim(0, 110)
    for bar, v in zip(bars, win_rates):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v*100:.1f}%", ha="center", va="bottom", fontsize=7, color="#eee")

    fig.tight_layout()

    out_root = RUNS_DIR / f"{prefix}_comparison.png" if prefix else RUNS_DIR / "comparison.png"
    _save(fig, out_root, show=show)
    for info in runs:
        _save(fig, info.path / "plots" / "stage_comparison.png", show=False)


# ── Figure 3: Score distributions (violin) ───────────────────────────────────

def plot_score_distributions(runs: list[RunInfo], show: bool) -> None:
    """Violin plots of the score distribution per run."""
    data: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []
    for info in runs:
        if info.episodes is not None and len(info.episodes.score) > 0:
            data.append(info.episodes.score)
            labels.append(f"S{info.stage}\n{info.name[-12:]}")
            colors.append(STAGE_COLORS.get(info.stage, "#aaa"))

    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(data)), 5))
    fig.suptitle("Score Distribution (all episodes)", fontsize=11, color="#eee")

    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    parts["cmedians"].set_color("#fff")
    parts["cbars"].set_color("#888")
    parts["cmins"].set_color("#888")
    parts["cmaxes"].set_color("#888")

    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Score")

    for info in runs:
        _save(fig, info.path / "plots" / "score_distribution.png", show=False)
    if show:
        plt.show()
    plt.close(fig)


# ── Figure 4: Eval curve overlay ─────────────────────────────────────────────

def plot_eval_overlay(runs: list[RunInfo], show: bool, prefix: str) -> None:
    """All stage eval curves on one axis for easy comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Eval Reward — all stages overlaid", fontsize=11, color="#eee")

    for info in runs:
        if info.evals is None:
            continue
        color = STAGE_COLORS.get(info.stage, "#aaa")
        ev = info.evals
        label = f"Stage {info.stage}"
        x_norm = ev.timesteps / ev.timesteps[-1]

        ax1.plot(ev.timesteps, ev.mean, color=color, label=label, marker="o", markersize=3)
        ax1.fill_between(ev.timesteps, ev.mean - ev.std, ev.mean + ev.std,
                         color=color, alpha=0.12)

        ax2.plot(x_norm * 100, ev.mean, color=color, label=label, marker="o", markersize=3)
        ax2.fill_between(x_norm * 100, ev.mean - ev.std, ev.mean + ev.std,
                         color=color, alpha=0.12)

    ax1.set_title("Absolute timestep")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Eval Reward")
    _format_millions(ax1)
    ax1.legend(fontsize=8)

    ax2.set_title("Normalised progress (%)")
    ax2.set_xlabel("Training progress (%)")
    ax2.set_ylabel("Eval Reward")
    ax2.legend(fontsize=8)

    fig.tight_layout()

    out = RUNS_DIR / f"{prefix}_eval_overlay.png" if prefix else RUNS_DIR / "eval_overlay.png"
    _save(fig, out, show=show)
    for info in runs:
        _save(fig, info.path / "plots" / "eval_overlay.png", show=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prefix", help="Run-name prefix to match (e.g. 'seq')")
    group.add_argument("--dirs", nargs="+", type=Path, metavar="DIR",
                       help="Explicit run directories")
    parser.add_argument("--no-show", action="store_true",
                        help="Save plots without opening a GUI window")
    args = parser.parse_args()

    show = not args.no_show

    print("\nDiscovering runs …")
    runs = discover_runs(run_dirs=args.dirs, prefix=args.prefix)

    if not runs:
        print("No runs found. Use --prefix or --dirs to specify.")
        return

    print(f"\nFound {len(runs)} run(s). Generating plots …\n")

    prefix = args.prefix or ""

    plot_learning_curves(runs, show=False)
    plot_score_distributions(runs, show=False)

    if len(runs) >= 2:
        plot_stage_comparison(runs, show=False, prefix=prefix)
        plot_eval_overlay(runs, show=show, prefix=prefix)
    elif show:
        plot_eval_overlay(runs, show=True, prefix=prefix)

    print("\nDone.")


if __name__ == "__main__":
    main()
