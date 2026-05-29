#!/usr/bin/env python3
"""Export the best model from the latest run to models/eval_best/policy.onnx.

If the eval callback already wrote models/eval_best/policy.onnx during training,
this script just reports the path. Otherwise it loads the best available checkpoint
and generates the file.

Usage
-----
    python rl/scripts/export_latest.py
    python rl/scripts/export_latest.py --stage 2
    python rl/scripts/export_latest.py --name seq --stage 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = RL_DIR.parent
RUNS_DIR = PROJECT_ROOT / "runs"

sys.path.insert(0, str(RL_DIR / "src"))


def find_latest_run(stage: int, name: str) -> Path:
    candidates = sorted(
        RUNS_DIR.glob(f"{name}_s{stage}_*"),
        key=lambda p: int(p.name.rsplit("_", 1)[-1]),
    )
    if not candidates:
        sys.exit(f"[ERROR] No run matching '{name}_s{stage}_*' in {RUNS_DIR}")
    return candidates[-1]


def best_checkpoint(run_dir: Path) -> Path:
    eval_best = run_dir / "eval_best"
    if eval_best.is_dir() and (eval_best / "best_model.zip").exists():
        return eval_best / "best_model"

    for name in ("final", "interrupted"):
        cand = run_dir / f"{name}.zip"
        if cand.exists():
            return run_dir / name

    ckpts = sorted(
        (run_dir / "checkpoints").glob("step_*.zip"),
        key=lambda c: int(c.stem.split("_")[1]),
    )
    if ckpts:
        return ckpts[-1].with_suffix("")

    sys.exit(f"[ERROR] No checkpoint found in {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--name", default="seq")
    args = parser.parse_args()

    run_dir = find_latest_run(args.stage, args.name)
    onnx_out = run_dir / "models" / "eval_best" / "policy.onnx"

    print(f"Run : {run_dir.name}")

    # If the eval callback already wrote it during training, nothing to do.
    if onnx_out.exists():
        print(f"ONNX: {onnx_out}  (already present)")
        return

    # Otherwise generate from the best available checkpoint.
    ckpt = best_checkpoint(run_dir)
    if ckpt.name == "best_model":
        vn_path = Path(str(ckpt.parent) + "_vecnorm.pkl")
    else:
        vn_path = Path(str(ckpt) + "_vecnorm.pkl")

    print(f"Checkpoint: {ckpt}")

    from network.export import export_to_onnx  # noqa: PLC0415

    try:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(str(ckpt))
    except (ImportError, ValueError, RuntimeError, KeyError, TypeError):
        from stable_baselines3 import PPO
        model = PPO.load(str(ckpt))

    vecnorm = vn_path if vn_path and vn_path.exists() else None
    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    export_to_onnx(model.policy, onnx_out, vecnorm_path=vecnorm)
    print(f"Exported: {onnx_out}")


if __name__ == "__main__":
    main()
