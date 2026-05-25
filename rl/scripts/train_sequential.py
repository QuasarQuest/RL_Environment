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
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parent  # …/rl
PROJECT_ROOT = RL_DIR.parent  # …/algorithm_test_bed
RUNS_DIR = PROJECT_ROOT / "runs"

# ── Per-stage timesteps (override all with --timesteps) ───────────────────────

DEFAULT_TIMESTEPS: dict[int, int] = {
    1: 1_500_000,
    2: 1_500_000,
    3: 2_000_000,
    4: 2_000_000,
    5: 2_500_000,
    6: 2_500_000,
}


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
) -> Path:
    """Invoke atb-train for *stage* and return the new run directory."""
    cmd = [
        "atb-train",
        f"env=stage{stage}",
        f"train.run_name={run_name}",
        f"train.total_timesteps={timesteps}",
    ]

    if stage in (4, 5, 6):
        # Deeper heads for combat stages — override net_arch.
        cmd += ["ppo.net_arch_pi=[128,64]", "ppo.net_arch_vf=[128,64]"]

    if resume is not None:
        # Hydra requires + prefix for keys not in the YAML defaults.
        cmd.append(f"+train.resume={resume}")

    print(f"\n{'=' * 60}")
    print(f"  Stage {stage}  —  {timesteps:,} steps")
    if resume:
        print(f"  resume : {resume}")
    print(f"{'=' * 60}\n")
    print("CMD:", " ".join(cmd), "\n")

    result = subprocess.run(cmd, cwd=RL_DIR)
    if result.returncode != 0:
        print(f"\n[ERROR] Stage {stage} failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)

    run_dir = find_latest_run(stage, run_name)
    if run_dir is None:
        print(f"[ERROR] Could not find run directory for stage {stage} after training.")
        sys.exit(1)

    return run_dir


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
            print(f"[auto] Using {prev_dir} as resume for stage {stages[0]}")

    for stage in stages:
        ts = args.timesteps if args.timesteps else DEFAULT_TIMESTEPS[stage]
        resume = eval_best_path(prev_dir) if prev_dir else None
        prev_dir = run_stage(stage, run_name, ts, resume)

    print(f"\n{'=' * 60}")
    print("  All stages complete.")
    print(f"{'=' * 60}")

    if not args.no_plot:
        plot_script = SCRIPT_DIR / "plot_runs.py"
        if plot_script.exists():
            print(f"\nRunning analysis: {plot_script} --prefix {run_name}\n")
            subprocess.run([sys.executable, str(plot_script), "--prefix", run_name, "--no-show"])
        else:
            print("plot_runs.py not found — skipping analysis.")


if __name__ == "__main__":
    main()
