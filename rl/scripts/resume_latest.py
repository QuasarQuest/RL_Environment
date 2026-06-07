#!/usr/bin/env python3
"""Resume training from the latest run's eval_best checkpoint.

Finds the most recent run directory — the one with the highest timestamp suffix in
its `<name>_s<stage>_<unix_secs>` name — that has an `eval_best/best_model.zip`, then
resumes `atb-train` on that stage from its eval_best. train.py loads the model +
VecNormalize and sets `reset_num_timesteps=False` on resume, so the step counter
*continues* rather than restarting.

Step accounting
---------------
SB3 trains until `num_timesteps` reaches `train.total_timesteps`, and resume keeps
the saved count — so passing a total ≤ the model's current count would train nothing.
This script therefore reads the run's furthest checkpoint (`checkpoints/step_<N>.zip`)
and, by default, trains `--add` steps ON TOP of it. Use `--timesteps` to set an
absolute target total instead.

Usage
-----
    python rl/scripts/resume_latest.py                     # latest run, +1,000,000 steps
    python rl/scripts/resume_latest.py --add 2000000       # +2M steps on top of current
    python rl/scripts/resume_latest.py --timesteps 6000000 # train to this ABSOLUTE total
    python rl/scripts/resume_latest.py --name seq           # only consider seq_* runs
    python rl/scripts/resume_latest.py --run runs/seq_s3_123 --add 500000   # explicit run
    python rl/scripts/resume_latest.py --dry-run            # print the command, don't run
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from rich.console import Console

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parent              # …/rl
PROJECT_ROOT = RL_DIR.parent            # …/algorithm_test_bed
RUNS_DIR = PROJECT_ROOT / "runs"

# Matches a stage run dir: "<name>_s<stage>_<unix_secs>" (e.g. seq_s3_1780781848).
# The "_combined_<date>" / plot dirs deliberately don't match.
RUN_RE = re.compile(r"^(?P<name>.+)_s(?P<stage>\d+)_(?P<ts>\d+)$")

console = Console()


def find_latest_run(name: str | None) -> tuple[Path, str, int, int]:
    """Return (run_dir, name, stage, timestamp) for the newest resumable run."""
    best: tuple[int, Path, str, int] | None = None  # (ts, dir, name, stage)
    for d in RUNS_DIR.glob("*"):
        if not d.is_dir():
            continue
        m = RUN_RE.match(d.name)
        if not m:
            continue
        if name is not None and m["name"] != name:
            continue
        if not (d / "eval_best" / "best_model.zip").exists():
            continue  # no usable checkpoint yet
        ts = int(m["ts"])
        if best is None or ts > best[0]:
            best = (ts, d, m["name"], int(m["stage"]))
    if best is None:
        console.print(f"[bold red]No resumable run found in {RUNS_DIR}[/bold red] "
                      f"(need a '<name>_s<stage>_<ts>' dir with eval_best/best_model.zip)")
        sys.exit(1)
    ts, d, nm, stage = best
    return d, nm, stage, ts


def current_steps(run_dir: Path) -> int:
    """Best estimate of the run's trained step count: the highest checkpoint step."""
    steps = []
    for p in (run_dir / "checkpoints").glob("step_*.zip"):
        m = re.search(r"step_(\d+)\.zip$", p.name)
        if m:
            steps.append(int(m.group(1)))
    return max(steps, default=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=None,
                    help="Explicit run dir to resume (skips latest-run auto-discovery).")
    ap.add_argument("--name", default=None,
                    help="Only consider runs with this name prefix (default: any).")
    ap.add_argument("--add", type=int, default=1_000_000,
                    help="Additional steps to train on top of the run's current count "
                         "(default: 1,000,000). Ignored if --timesteps is given.")
    ap.add_argument("--timesteps", type=int, default=None,
                    help="Absolute target total step count (overrides --add).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the atb-train command and exit without running.")
    args = ap.parse_args()

    if args.run is not None:
        run_dir = args.run if args.run.is_absolute() else (PROJECT_ROOT / args.run)
        run_dir = run_dir.resolve()
        m = RUN_RE.match(run_dir.name)
        if not m:
            console.print(f"[bold red]--run {run_dir.name} is not a '<name>_s<stage>_<ts>' dir[/bold red]")
            sys.exit(1)
        if not (run_dir / "eval_best" / "best_model.zip").exists():
            console.print(f"[bold red]{run_dir}/eval_best/best_model.zip not found[/bold red]")
            sys.exit(1)
        name, stage = m["name"], int(m["stage"])
    else:
        run_dir, name, stage, _ts = find_latest_run(args.name)

    cur = current_steps(run_dir)
    total = args.timesteps if args.timesteps is not None else cur + args.add
    eval_best = run_dir / "eval_best"

    console.rule(f"[bold green]Resume — {run_dir.name}[/bold green]  [dim]stage {stage}[/dim]")
    console.print(f"  resume     {eval_best}")
    console.print(f"  current    ~{cur:,} steps (from latest checkpoint)")
    console.print(f"  target     {total:,} total steps  "
                  f"[dim]({'absolute --timesteps' if args.timesteps else f'+{args.add:,} via --add'})[/dim]")
    if total <= cur:
        console.print(f"  [bold yellow]⚠ target ({total:,}) ≤ current (~{cur:,}); "
                      f"resume would train ~0 steps. Raise --add/--timesteps.[/bold yellow]")

    cmd = [
        "atb-train",
        f"env=stage{stage}",
        f"train.run_name={name}",
        f"train.total_timesteps={total}",
        f"+train.resume={eval_best}",
    ]
    console.print(f"  [dim]cmd[/dim]        {' '.join(cmd)}")

    if args.dry_run:
        console.print("[dim]--dry-run: not executing.[/dim]")
        return

    console.print()
    sys.exit(subprocess.run(cmd, cwd=RL_DIR).returncode)


if __name__ == "__main__":
    main()
