#!/usr/bin/env python3
"""
Sequential curriculum training — stages 1 through 6.

Each stage hot-starts from the eval_best checkpoint of the previous stage,
transferring policy weights and VecNormalize running stats.

Usage
-----
# Full run from scratch (1.5 M steps per stage):
    python rl/scripts/train_sequential.py

# Quick exploratory run (override timesteps):
    python rl/scripts/train_sequential.py --timesteps 500000

# Only specific stages:
    python rl/scripts/train_sequential.py --stages 3 4 5

# Provide an explicit stage-1 checkpoint to start from:
    python rl/scripts/train_sequential.py --from-run runs/run_seq_s1_XYZ

# Name the run series (default: "seq"):
    python rl/scripts/train_sequential.py --name myexp --timesteps 1000000

All atb-train output is streamed to the terminal in real time.
After all stages complete, plot_runs.py is called automatically.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parent  # …/rl
PROJECT_ROOT = RL_DIR.parent  # …/algorithm_test_bed
RUNS_DIR = PROJECT_ROOT / "runs"

# ── Per-stage timesteps (override all with --timesteps) ───────────────────────

DEFAULT_TIMESTEPS: dict[int, int] = {
    1: 2_500_000,
    2: 2_500_000,
    3: 2_500_000,
    4: 3_000_000,
    5: 4_500_000,
    6: 4_000_000,
}

PLOT_SCRIPT = SCRIPT_DIR / "plot_runs.py"

console = Console()


def plot_single_stage(run_dir: Path) -> None:
    """Generate the individual plots for one stage's run directory."""
    if not PLOT_SCRIPT.exists():
        return
    console.print(f"\n  [dim]plotting  {run_dir.name}/plots/[/dim]")
    subprocess.run([sys.executable, str(PLOT_SCRIPT),
                    "--dirs", str(run_dir), "--no-show"])


def plot_curriculum(run_name: str) -> None:
    """Generate the combined cross-stage plots for the whole run series."""
    if not PLOT_SCRIPT.exists():
        return
    console.print()
    console.rule(f"[bold]Curriculum plots[/bold]  [dim]prefix: {run_name}[/dim]")
    console.print()
    subprocess.run([sys.executable, str(PLOT_SCRIPT),
                    "--prefix", run_name, "--no-show"])


def _fmt_dur(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def find_latest_run(stage: int, run_name: str) -> Path | None:
    """Return the most recent run directory for *stage* matching *run_name*."""
    pattern = f"{run_name}_s{stage}_*"
    candidates = sorted(
        RUNS_DIR.glob(pattern),
        key=lambda p: int(p.name.rsplit("_", 1)[-1]),
    )
    return candidates[-1] if candidates else None


def eval_best_path(run_dir: Path) -> Path:
    return run_dir / "eval_best"


def run_stage(
        stage: int,
        run_name: str,
        timesteps: int,
        resume: Path | None,
) -> tuple[Path, float]:
    """Invoke atb-train for *stage* and return (run_dir, elapsed_seconds)."""
    cmd = [
        "atb-train",
        f"env=stage{stage}",
        f"train.run_name={run_name}",
        f"train.total_timesteps={timesteps}",
    ]

    if stage in (4, 5, 6):
        # Deeper heads for combat stages — only applied by train.py when
        # algo != recurrent_ppo (RecurrentPPO ignores net_arch; LSTM depth
        # is controlled by n_lstm_layers instead).
        cmd += ["ppo.net_arch_pi=[128,64]", "ppo.net_arch_vf=[128,64]"]

    if resume is not None:
        # Hydra requires + prefix for keys not in the YAML defaults.
        cmd.append(f"+train.resume={resume}")

    console.print()
    console.rule(f"[bold cyan]Stage {stage}[/bold cyan]  [dim]{timesteps:,} steps[/dim]")
    if resume:
        console.print(f"  [dim]resume[/dim]  {resume}")
    console.print(f"  [dim]cmd[/dim]     {cmd[0]}")
    for arg in cmd[1:]:
        console.print(f"          {arg}")
    console.print()

    t0 = time.time()
    result = subprocess.run(cmd, cwd=RL_DIR)
    elapsed = time.time() - t0

    if result.returncode != 0:
        console.print()
        console.print(
            f"  [bold red]✗ Stage {stage} failed[/bold red]"
            f"  exit {result.returncode}  ·  {_fmt_dur(elapsed)}"
        )
        sys.exit(result.returncode)

    run_dir = find_latest_run(stage, run_name)
    if run_dir is None:
        console.print(f"  [bold red]✗ Run directory not found for stage {stage}[/bold red]")
        sys.exit(1)

    console.print()
    console.print(
        f"  [bold green]✓ Stage {stage} complete[/bold green]"
        f"  [dim]{_fmt_dur(elapsed)}[/dim]  [dim]→[/dim]  {run_dir.name}"
    )
    return run_dir, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", type=int, default=list(range(1, 7)),
                        metavar="N", help="Stages to run (default: 1 2 3 4 5 6)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override timesteps for every stage")
    parser.add_argument("--name", default="seq",
                        help="Run-name prefix (default: seq)")
    parser.add_argument("--from-run", type=Path, default=None, metavar="DIR",
                        help="Use this run dir as stage-(stages[0]-1) checkpoint")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip auto-plotting after all stages complete")
    args = parser.parse_args()

    stages: list[int] = sorted(int(s) for s in args.stages)
    run_name = args.name
    prev_dir: Path | None = args.from_run

    # If first stage > 1 and no explicit --from-run, try to auto-discover.
    if prev_dir is None and stages[0] > 1:
        prev_dir = find_latest_run(stages[0] - 1, run_name)
        if prev_dir:
            console.print(f"  [dim]auto-resume  stage {stages[0] - 1} → {prev_dir.name}[/dim]")

    console.rule(f"[bold green]Sequential Training — {run_name}[/bold green]  [dim]stages {stages}[/dim]")
    console.print()

    results: list[tuple[int, str, float]] = []
    t_total = time.time()

    for stage in stages:
        ts = args.timesteps if args.timesteps else DEFAULT_TIMESTEPS[stage]
        resume = eval_best_path(prev_dir) if prev_dir is not None else None
        # run_stage always returns a concrete Path; keep it in its own variable
        # so the type stays Path (not Path | None) for .name and plotting below.
        run_dir, elapsed = run_stage(stage, run_name, ts, resume)
        results.append((stage, run_dir.name, elapsed))

        # Individual plots after every stage (including the first) so progress
        # is visible mid-run without waiting for the whole curriculum.
        if not args.no_plot:
            plot_single_stage(run_dir)

        prev_dir = run_dir  # hot-start the next stage from this run's eval_best

    total_elapsed = time.time() - t_total

    # Summary table
    console.print()
    table = Table(title="Training Summary", box=box.SIMPLE, title_style="bold green", padding=(0, 1))
    table.add_column("Stage", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("Run directory", style="dim")
    table.add_column("Duration", justify="right", no_wrap=True)
    for s, name, dur in results:
        table.add_row(str(s), name, _fmt_dur(dur))
    table.add_section()
    table.add_row("", "[bold]total[/bold]", f"[bold]{_fmt_dur(total_elapsed)}[/bold]")
    console.print(table)

    # Combined cross-stage plots once all stages are done.
    if not args.no_plot:
        plot_curriculum(run_name)


if __name__ == "__main__":
    main()
