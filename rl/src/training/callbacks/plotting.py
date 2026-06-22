"""Periodic plotting during training.

By default plots are only produced *after* a stage finishes. This callback
regenerates the run's per-stage plots every `plot_freq` env steps so you can watch
learning curves / training metrics evolve mid-run.

It spawns ``plot_runs.py --dirs <run_dir> --no-show`` as a NON-BLOCKING subprocess
(headless Agg backend) so training never stalls on rendering, and skips a tick if
the previous plot job is still running (no pile-up). Output is appended to
``<run_dir>/plot.log``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


class PeriodicPlotCallback(BaseCallback):
    """Regenerate this run's plots every `plot_freq` env steps (0 = disabled)."""

    def __init__(
        self,
        plot_freq: int,
        run_dir: Path,
        plot_script: Path,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.plot_freq = int(plot_freq)
        self.run_dir = Path(run_dir)
        self.plot_script = Path(plot_script)
        self._last = 0
        self._proc: Optional[subprocess.Popen] = None

    # ── internal ────────────────────────────────────────────────────────────────

    def _spawn(self) -> None:
        if not self.plot_script.exists():
            return
        # Don't pile up: skip if the previous plot job is still rendering.
        if self._proc is not None and self._proc.poll() is None:
            if self.verbose:
                print("  [plot] previous plot job still running — skipping")
            return

        env = {**os.environ, "MPLBACKEND": "Agg"}  # headless render, no GUI needed
        self.run_dir.mkdir(parents=True, exist_ok=True)
        log = self.run_dir / "plot.log"
        self._logfh = open(log, "a")
        self._proc = subprocess.Popen(
            [sys.executable, str(self.plot_script),
             "--dirs", str(self.run_dir), "--no-show"],
            stdout=self._logfh, stderr=subprocess.STDOUT, env=env,
        )
        if self.verbose:
            print(f"  [plot] refreshing plots @ {self.num_timesteps:,} steps → {self.run_dir}/plots/")

    # ── hooks ───────────────────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        if self.plot_freq <= 0:
            return True
        if self.num_timesteps - self._last >= self.plot_freq:
            self._last = self.num_timesteps
            self._spawn()
        return True

    def _on_training_end(self) -> None:
        if self.plot_freq <= 0:
            return
        # Wait for any in-flight job, then render a final, up-to-date set (blocking).
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                pass
        self._spawn()
        if self._proc is not None:
            try:
                self._proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                pass
