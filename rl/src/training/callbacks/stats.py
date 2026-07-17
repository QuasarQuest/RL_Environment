"""Episode statistics fan-out to TensorBoard and HDF5."""
from __future__ import annotations

import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from monitoring.stats_writer import HDF5StatsWriter


class EpisodeStatsCallback(BaseCallback):
    """Record per-episode stats to TensorBoard and an HDF5 file."""

    def __init__(self, stats_path: Path, flush_every: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._writer = HDF5StatsWriter(Path(stats_path), flush_every=flush_every)
        self._start_time = time.time()
        self._start_steps = 0

    def _on_training_start(self) -> None:
        # Anchor throughput to THIS learn() call — on resume num_timesteps
        # continues from the checkpoint while wall time restarts.
        self._start_time = time.time()
        self._start_steps = self.num_timesteps

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            ep = info["episode"]
            row = {
                "timestep": self.num_timesteps,
                "episode_reward": float(ep["r"]),
                "episode_length": int(ep["l"]),
                "score": float(info.get("score", 0.0)),
            }
            self._writer.append(row)

            self.logger.record("game/episode_reward", row["episode_reward"])
            self.logger.record("game/episode_length", row["episode_length"])
            self.logger.record("game/score", row["score"])

        elapsed = max(time.time() - self._start_time, 1.0)
        self.logger.record("perf/steps_per_sec", (self.num_timesteps - self._start_steps) / elapsed)
        return True

    def _on_training_end(self) -> None:
        self._writer.close()

    def close(self) -> None:
        """Flush buffered rows. SB3 skips on_training_end when KeyboardInterrupt
        propagates out of learn(), so train.py calls this from its finally block —
        otherwise up to flush_every−1 episode rows are lost on Ctrl-C."""
        self._writer.close()
