#!/usr/bin/env python3
"""
ATB run analysis — loads stats.h5 + evaluations.npz + TensorBoard scalars and
saves matplotlib plots, organised into folders.

Per-stage output  → runs/<run_dir>/plots/
    learning_curves/   reward.png   score.png   eval.png
    score/             distribution.png  ecdf.png  over_time.png  box.png  summary.png
    training_metrics/  one <metric>.png per TensorBoard scalar
  Built from THAT run only — a stage folder never contains another stage's data.

Combined output   → runs/<prefix>_combined_<YYYYmmdd-HHMMSS>/
    learning_curves/   reward.png  score.png  eval.png  eval_normalised.png   (all stages overlaid)
    comparison/        final_eval.png  final_score.png  win_rate.png
    training_metrics/  one <metric>.png per scalar, all stages overlaid
    manifest.txt       date + stages included
  Written once, into a single timestamped folder — never back into stage folders.

Usage
-----
    python rl/scripts/plot_runs.py                       # all runs in runs/
    python rl/scripts/plot_runs.py --prefix seq          # runs named seq_*
    python rl/scripts/plot_runs.py --dirs runs/run_s1_X  # explicit dirs
    python rl/scripts/plot_runs.py --prefix seq --no-show
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
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

# One-line explanation per scalar — shown under each metric plot so the output
# doubles as a "what does this mean" reference.
METRIC_HELP = {
    "game/episode_reward":          "shaped reward per episode (tick + pickup + deposit − wall)",
    "game/episode_length":          "decision-steps (options) per episode",
    "game/score":                   "gold banked at base — the true objective",
    "game/win_rate":                "fraction of episodes flagged win (always 0 in gold-rush)",
    "perf/steps_per_sec":           "training throughput (env steps / second)",
    "rollout/ep_len_mean":          "SB3 rolling mean episode length",
    "rollout/ep_rew_mean":          "SB3 rolling mean episode reward",
    "time/fps":                     "SB3 frames per second",
    "train/ent_coef":               "entropy-bonus weight — higher = more exploration",
    "train/approx_kl":              "policy change per update — spikes = steps too large",
    "train/clip_fraction":          "fraction of PPO ratios clipped — high = steps too large",
    "train/clip_range":             "PPO clip epsilon (the ± ratio bound)",
    "train/entropy_loss":           "−entropy; magnitude shrinks as the policy sharpens",
    "train/explained_variance":     "value-fit quality: 1 = perfect, 0 = mean, <0 = worse",
    "train/learning_rate":          "optimizer learning rate (scheduled)",
    "train/loss":                   "total PPO loss (policy + value·vf_coef − entropy·ent_coef)",
    "train/policy_gradient_loss":   "actor (policy-gradient) loss",
    "train/value_loss":             "critic loss — MSE of value vs returns",
    "eval/mean_reward":             "deterministic eval reward (masked greedy policy)",
    "eval/mean_ep_length":          "deterministic eval episode length",
    "policy/chosen_gold_dist":      "distance to the gold it chose — high = ranging far past nearer gold",
    "policy/own_region_skip_rate":  "when local gold exists & it collects, how often it picks a DIFFERENT region",
    "policy/own_region_has_gold_rate": "fraction of decisions where the agent's own region had collectible gold",
    "policy/option_len":            "mean ticks per option — lower = shorter trips; near MAX_OPTION_TICKS = wandering",
}

# ── Data classes ──────────────────────────────────────────────────────────────


class EpisodeData(NamedTuple):
    timestep: np.ndarray
    reward: np.ndarray
    score: np.ndarray
    win: np.ndarray
    ep_length: np.ndarray


class EvalData(NamedTuple):
    timesteps: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    ep_lengths: np.ndarray


class RunInfo(NamedTuple):
    path: Path
    name: str
    stage: int
    episodes: EpisodeData | None
    evals: EvalData | None
    metrics: dict[str, tuple[np.ndarray, np.ndarray]]  # tag -> (steps, values)


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
    results = d["results"]
    return EvalData(
        timesteps=d["timesteps"],
        mean=results.mean(axis=1),
        std=results.std(axis=1),
        ep_lengths=d["ep_lengths"].mean(axis=1),
    )


def load_tb_scalars(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every TensorBoard scalar for a run: tag -> (steps, values)."""
    tb_root = run_dir / "tensorboard"
    if not tb_root.exists():
        return {}
    events = sorted(tb_root.glob("**/events.out.tfevents*"))
    if not events:
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  [note] tensorboard not installed — skipping training-metric plots")
        return {}

    acc = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag in acc.Tags().get("scalars", []):
        s = acc.Scalars(tag)
        out[tag] = (
            np.array([e.step for e in s], dtype=np.int64),
            np.array([e.value for e in s], dtype=np.float64),
        )
    return out


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
        metrics = load_tb_scalars(d)
        if episodes is None and evals is None and not metrics:
            print(f"  [skip] {d.name} — no stats.h5 / evaluations.npz / tensorboard")
            continue
        runs.append(RunInfo(path=d, name=d.name, stage=stage,
                            episodes=episodes, evals=evals, metrics=metrics))
        print(f"  loaded  {d.name}  (stage {stage}, {len(metrics)} tb metrics)")
    return runs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rolling(
    x: np.ndarray, y: np.ndarray, window: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling mean ± std; window shrinks for short runs so a curve still appears."""
    n = len(y)
    if n == 0:
        return x, y, np.zeros_like(y)
    w = window if n >= window else max(1, n // 3)
    idx = np.arange(w - 1, n)
    means = np.array([y[i - w + 1:i + 1].mean() for i in idx])
    stds = np.array([y[i - w + 1:i + 1].std() for i in idx])
    return x[idx], means, stds


def _format_millions(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K")
    )


def _tag_filename(tag: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", tag).strip("_")


def _save(fig: plt.Figure, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  saved  {path}")
    if show:
        plt.show()
    plt.close(fig)


def _curve(
    out_path: Path,
    title: str,
    series: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, str, str]],
    ylabel: str,
    xlabel: str = "Timestep",
    millions: bool = True,
    markers: bool = False,
    show: bool = False,
) -> None:
    """One line chart. series = [(x, y, std|None, color, label), …]."""
    if not series:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle(title, fontsize=11, color="#eee")
    for x, y, sd, color, label in series:
        ax.plot(x, y, color=color, label=label, marker="o" if markers else None, markersize=3)
        if sd is not None:
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if millions:
        _format_millions(ax)
    if len(series) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_path, show=show)


# ══════════════════════════════════════════════════════════════════════════════
# PER-STAGE plots — one run each, into runs/<run>/plots/<group>/.
# ══════════════════════════════════════════════════════════════════════════════

def plot_run_learning_curves(info: RunInfo, out_dir: Path, show: bool) -> None:
    d = out_dir / "learning_curves"
    color = STAGE_COLORS.get(info.stage, "#aaa")
    if info.episodes is not None:
        ts, mu, sd = _rolling(info.episodes.timestep, info.episodes.reward)
        _curve(d / "reward.png", f"{info.name} — Episode Reward (rolling 100)",
               [(ts, mu, sd, color, "reward")], "Reward", show=show)
        ts, mu, sd = _rolling(info.episodes.timestep, info.episodes.score)
        _curve(d / "score.png", f"{info.name} — Score (rolling 100)",
               [(ts, mu, sd, color, "score")], "Score")
    if info.evals is not None and len(info.evals.timesteps):
        ev = info.evals
        _curve(d / "eval.png", f"{info.name} — Eval Reward (mean ± std)",
               [(ev.timesteps, ev.mean, ev.std, color, "eval")], "Eval Reward", markers=True)


def plot_run_score_stats(info: RunInfo, out_dir: Path) -> None:
    if info.episodes is None or len(info.episodes.score) == 0:
        return
    d = out_dir / "score"
    color = STAGE_COLORS.get(info.stage, "#aaa")
    score = info.episodes.score.astype(float)
    ts = info.episodes.timestep

    # Histogram.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.suptitle(f"{info.name} — score distribution", fontsize=11, color="#eee")
    ax.hist(score, bins=min(40, max(5, len(score) // 3)), color=color, alpha=0.75)
    ax.axvline(float(np.median(score)), color="#fff", ls="--", lw=1.2, label=f"median {np.median(score):.2f}")
    ax.axvline(float(score.mean()), color="#ffd54f", ls=":", lw=1.2, label=f"mean {score.mean():.2f}")
    ax.set_xlabel("Score")
    ax.set_ylabel("Episodes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, d / "distribution.png", show=False)

    # ECDF.
    xs = np.sort(score)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.suptitle(f"{info.name} — score ECDF", fontsize=11, color="#eee")
    ax.step(xs, ys, where="post", color=color)
    for q in (0.5, 0.9):
        ax.axhline(q, color="#555", ls=":", lw=0.8)
    ax.set_xlabel("Score")
    ax.set_ylabel("P(score ≤ x)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, d / "ecdf.png", show=False)

    # Score over time (raw + rolling).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle(f"{info.name} — score over training", fontsize=11, color="#eee")
    ax.scatter(ts, score, s=4, alpha=0.12, color=color)
    tsr, mu, sd = _rolling(ts, score)
    ax.plot(tsr, mu, color=color, label="rolling 100")
    ax.fill_between(tsr, mu - sd, mu + sd, color=color, alpha=0.15)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Score")
    _format_millions(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, d / "over_time.png", show=False)

    # Boxplot.
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.suptitle(f"{info.name} — score spread", fontsize=11, color="#eee")
    bp = ax.boxplot(score, vert=True, patch_artist=True, showfliers=True,
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.3})
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("#fff")
    ax.set_xticks([])
    ax.set_ylabel("Score")
    fig.tight_layout()
    _save(fig, d / "box.png", show=False)

    # Text summary.
    tail = score[-max(1, len(score) // 10):]
    lines = [
        f"episodes      : {len(score):,}",
        f"mean          : {score.mean():.3f}",
        f"median        : {np.median(score):.3f}",
        f"std           : {score.std():.3f}",
        f"min / max     : {score.min():.2f} / {score.max():.2f}",
        f"p10 / p90     : {np.percentile(score, 10):.2f} / {np.percentile(score, 90):.2f}",
        f"last 10% mean : {tail.mean():.3f}",
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle(f"{info.name} — score summary", fontsize=11, color="#eee")
    ax.axis("off")
    ax.text(0.02, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=11,
            va="top", fontfamily="monospace", color="#eee")
    fig.tight_layout()
    _save(fig, d / "summary.png", show=False)


def plot_run_training_metrics(info: RunInfo, out_dir: Path) -> None:
    """One PNG per TensorBoard scalar, this run only."""
    if not info.metrics:
        return
    d = out_dir / "training_metrics"
    color = STAGE_COLORS.get(info.stage, "#aaa")
    for tag in sorted(info.metrics):
        steps, vals = info.metrics[tag]
        fig, ax = plt.subplots(figsize=(7, 4.3))
        fig.suptitle(tag, fontsize=11, color="#eee")
        ax.plot(steps, vals, color=color, marker="o", markersize=3)
        ax.set_title(METRIC_HELP.get(tag, ""), fontsize=8, color="#bbb")
        ax.set_xlabel("step")
        _format_millions(ax)
        fig.tight_layout()
        _save(fig, d / f"{_tag_filename(tag)}.png", show=False)


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED plots — cross-stage, into runs/<prefix>_combined_<ts>/<group>/.
# ══════════════════════════════════════════════════════════════════════════════

def plot_combined_learning_curves(runs: list[RunInfo], out_dir: Path) -> None:
    d = out_dir / "learning_curves"

    def overlay(field: str, ylabel: str, fname: str) -> None:
        series = []
        for info in runs:
            if info.episodes is None:
                continue
            ts, mu, sd = _rolling(info.episodes.timestep, getattr(info.episodes, field))
            series.append((ts, mu, sd, STAGE_COLORS.get(info.stage, "#aaa"), f"S{info.stage}"))
        _curve(d / fname, f"{ylabel} — all stages (rolling 100)", series, ylabel)

    overlay("reward", "Reward", "reward.png")
    overlay("score", "Score", "score.png")

    # Eval overlay — absolute and normalised progress.
    abs_series, norm_series = [], []
    for info in runs:
        if info.evals is None or len(info.evals.timesteps) == 0:
            continue
        ev = info.evals
        color = STAGE_COLORS.get(info.stage, "#aaa")
        abs_series.append((ev.timesteps, ev.mean, ev.std, color, f"S{info.stage}"))
        norm_series.append((ev.timesteps / ev.timesteps[-1] * 100, ev.mean, ev.std, color, f"S{info.stage}"))
    _curve(d / "eval.png", "Eval Reward — all stages", abs_series, "Eval Reward", markers=True)
    _curve(d / "eval_normalised.png", "Eval Reward — normalised progress", norm_series,
           "Eval Reward", xlabel="Training progress (%)", millions=False, markers=True)


def plot_combined_comparison(runs: list[RunInfo], out_dir: Path) -> None:
    d = out_dir / "comparison"
    stages = [r.stage for r in runs]
    colors = [STAGE_COLORS.get(s, "#aaa") for s in stages]
    labels = [f"S{r.stage}" for r in runs]
    x = np.arange(len(runs))

    def tail(arr_name: str, frac: int = 10):
        out = []
        for r in runs:
            if r.episodes is None:
                out.append(np.array([np.nan]))
            else:
                a = getattr(r.episodes, arr_name).astype(float)
                out.append(a[-max(1, len(a) // frac):])
        return out

    def bar(values, errs, title, ylabel, fmt, fname, ylim=None):
        fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(runs)), 4.5))
        fig.suptitle(title, fontsize=11, color="#eee")
        bars = ax.bar(x, values, yerr=errs, color=colors,
                      error_kw={"ecolor": "#888", "capsize": 4}, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        for b, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt(v),
                        ha="center", va="bottom", fontsize=8, color="#eee")
        fig.tight_layout()
        _save(fig, d / fname, show=False)

    final_eval_mean = [r.evals.mean[-1] if r.evals is not None and len(r.evals.mean) else np.nan for r in runs]
    final_eval_std = [r.evals.std[-1] if r.evals is not None and len(r.evals.std) else 0.0 for r in runs]
    bar(final_eval_mean, final_eval_std, "Final Eval Reward (mean ± std)", "Eval Reward",
        lambda v: f"{v:.1f}", "final_eval.png")

    sc = tail("score")
    bar([d_.mean() for d_ in sc], [d_.std() for d_ in sc],
        "Final Score (last 10 % of episodes)", "Score", lambda v: f"{v:.2f}", "final_score.png")

    wn = tail("win")
    bar([d_.mean() * 100 for d_ in wn], None, "Win Rate (last 10 % of episodes)", "Win %",
        lambda v: f"{v:.1f}%", "win_rate.png", ylim=(0, 110))


def plot_combined_training_metrics(runs: list[RunInfo], out_dir: Path) -> None:
    """One PNG per scalar, all stages overlaid."""
    all_tags = sorted({t for r in runs for t in r.metrics})
    if not all_tags:
        return
    d = out_dir / "training_metrics"
    for tag in all_tags:
        fig, ax = plt.subplots(figsize=(7, 4.3))
        fig.suptitle(tag, fontsize=11, color="#eee")
        plotted = False
        for info in runs:
            if tag not in info.metrics:
                continue
            steps, vals = info.metrics[tag]
            ax.plot(steps, vals, color=STAGE_COLORS.get(info.stage, "#aaa"),
                    label=f"S{info.stage}", marker="o", markersize=2)
            plotted = True
        ax.set_title(METRIC_HELP.get(tag, ""), fontsize=8, color="#bbb")
        ax.set_xlabel("step")
        _format_millions(ax)
        if plotted:
            ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, d / f"{_tag_filename(tag)}.png", show=False)


def write_manifest(runs: list[RunInfo], out_dir: Path, stamp: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "ATB combined curriculum plots",
        f"generated : {datetime.now().isoformat(timespec='seconds')}",
        f"stamp     : {stamp}",
        "",
        "stages included:",
    ]
    for r in runs:
        n_eps = len(r.episodes.reward) if r.episodes is not None else 0
        last_eval = (f"{r.evals.mean[-1]:.2f}" if r.evals is not None and len(r.evals.mean) else "—")
        lines.append(f"  stage {r.stage:<2}  {r.name}  "
                     f"(episodes={n_eps}, last_eval={last_eval}, tb_metrics={len(r.metrics)})")
    (out_dir / "manifest.txt").write_text("\n".join(lines) + "\n")
    print(f"  saved  {out_dir / 'manifest.txt'}")


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

    print(f"\nFound {len(runs)} run(s). Generating per-stage plots …\n")
    for info in runs:
        out = info.path / "plots"
        plot_run_learning_curves(info, out, show=show and len(runs) == 1)
        plot_run_score_stats(info, out)
        plot_run_training_metrics(info, out)

    if len(runs) >= 2:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        label = args.prefix or "all"
        combined = RUNS_DIR / f"{label}_combined_{stamp}"
        print(f"\nGenerating combined plots → {combined}/\n")
        plot_combined_learning_curves(runs, combined)
        plot_combined_comparison(runs, combined)
        plot_combined_training_metrics(runs, combined)
        write_manifest(runs, combined, stamp)

    print("\nDone.")


if __name__ == "__main__":
    main()
