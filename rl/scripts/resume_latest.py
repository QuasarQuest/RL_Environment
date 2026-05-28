#!/usr/bin/env python3
"""
Resume training from the latest available checkpoint.

Searches runs/ for the most recent run matching --stage and --name, then
picks the best checkpoint in this priority order:
  1. eval_best/   (best validated model)
  2. final.zip    (end-of-run model)
  3. checkpoints/step_N.zip  (latest periodic snapshot)

Usage
-----
# Resume the latest stage-1 run:
    python rl/scripts/resume_latest.py

# Resume a specific stage:
    python rl/scripts/resume_latest.py --stage 3

# Resume a specific run-name prefix:
    python rl/scripts/resume_latest.py --name myexp --stage 2

# Override timesteps for the resumed run:
    python rl/scripts/resume_latest.py --timesteps 1000000

# Dry-run: print what would be resumed without launching:
    python rl/scripts/resume_latest.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
RL_DIR       = SCRIPT_DIR.parent
PROJECT_ROOT = RL_DIR.parent
RUNS_DIR     = PROJECT_ROOT / "runs"

DEFAULT_TIMESTEPS: dict[int, int] = {
    1: 1_000_000,
    2: 1_000_000,
    3: 1_000_000,
    4: 1_000_000,
    5: 1_000_000,
    6: 1_000_000,
}


def find_latest_run(stage: int, name: str) -> Path | None:
    pattern = f"{name}_s{stage}_*"
    candidates = sorted(
        RUNS_DIR.glob(pattern),
        key=lambda p: int(p.name.rsplit("_", 1)[-1]),
    )
    return candidates[-1] if candidates else None


def best_checkpoint(run_dir: Path) -> Path | None:
    """Return the best available checkpoint path inside run_dir."""
    eval_best = run_dir / "eval_best"
    if eval_best.is_dir() and (eval_best / "best_model.zip").exists():
        return eval_best

    final = run_dir / "final.zip"
    if final.exists():
        return final

    ckpts = sorted(
        (run_dir / "checkpoints").glob("step_*.zip"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return ckpts[-1] if ckpts else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=int, default=1, help="Stage to resume (default: 1)")
    parser.add_argument("--name",  default="seq",       help="Run-name prefix (default: seq)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total_timesteps (default: per-stage value)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved checkpoint path without launching training")
    args = parser.parse_args()

    run_dir = find_latest_run(args.stage, args.name)
    if run_dir is None:
        print(f"[ERROR] No run found matching '{args.name}_s{args.stage}_*' in {RUNS_DIR}")
        sys.exit(1)

    ckpt = best_checkpoint(run_dir)
    if ckpt is None:
        print(f"[ERROR] No checkpoint found in {run_dir}")
        sys.exit(1)

    timesteps = args.timesteps or DEFAULT_TIMESTEPS.get(args.stage, 2_500_000)

    resume_path = ckpt  # absolute path — matches train_sequential.py behaviour

    print(f"Run dir   : {run_dir.name}")
    print(f"Checkpoint: {resume_path}")
    print(f"Stage     : {args.stage}   Timesteps: {timesteps:,}")

    if args.dry_run:
        print("\n[dry-run] Would execute:")
        print(f"  atb-train env=stage{args.stage} train.run_name={args.name} "
              f"train.total_timesteps={timesteps} +train.resume={resume_path}")
        return

    cmd = [
        "atb-train",
        f"env=stage{args.stage}",
        f"train.run_name={args.name}",
        f"train.total_timesteps={timesteps}",
        f"+train.resume={resume_path}",
    ]
    if args.stage in (4, 5, 6):
        # Only applies when algo != recurrent_ppo (RecurrentPPO ignores net_arch).
        cmd += ["ppo.net_arch_pi=[128,64]", "ppo.net_arch_vf=[128,64]"]

    print()
    result = subprocess.run(cmd, cwd=RL_DIR)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
