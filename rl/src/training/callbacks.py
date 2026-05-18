# rl/src/training/callbacks.py

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class CheckpointCallback(BaseCallback):
    """
    Saves a SB3 model checkpoint and VecNormalize statistics every
    `save_freq` steps. Also saves the best model by mean episode reward.
    """

    def __init__(
        self,
        save_freq:     int,
        models_dir:    str,
        run_name:      str,
        vec_normalize: Optional[VecNormalize] = None,
        verbose:       int = 1,
    ):
        super().__init__(verbose)
        self.save_freq     = save_freq
        self.models_dir    = Path(models_dir)
        self.run_name      = run_name
        self.vec_normalize = vec_normalize
        self.best_reward   = -float("inf")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._save(tag=f"step_{self.num_timesteps}")

        ep_rewards = [
            info["episode"]["r"]
            for info in self.locals.get("infos", [])
            if "episode" in info
        ]
        if ep_rewards:
            mean_r = float(np.mean(ep_rewards))
            if mean_r > self.best_reward:
                self.best_reward = mean_r
                self._save(tag="best")
                if self.verbose:
                    print(f"  ✓ New best model — mean_r={mean_r:.3f} "
                          f"@ step {self.num_timesteps}")
        return True

    def _save(self, tag: str) -> None:
        path = self.models_dir / f"{self.run_name}_{tag}"
        self.model.save(str(path))
        if self.vec_normalize is not None:
            self.vec_normalize.save(str(path) + "_vecnorm.pkl")


class EpisodeStatsCallback(BaseCallback):
    """
    Logs per-episode game stats to HDF5 and TensorBoard.

    HDF5 schema /episodes/:
      timestep, episode_reward, episode_length,
      kills, deaths, score, win
    """

    def __init__(self, stats_dir: str, run_name: str, verbose: int = 0):
        super().__init__(verbose)
        self.stats_path = Path(stats_dir) / f"{run_name}.h5"
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer:     list[dict] = []
        self._flush_every = 100
        self._start_time  = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            ep = info["episode"]
            self._buffer.append({
                "timestep":       self.num_timesteps,
                "episode_reward": float(ep["r"]),
                "episode_length": int(ep["l"]),
                "kills":          int(info.get("kills",  0)),
                "deaths":         int(info.get("deaths", 0)),
                "score":          float(info.get("score", 0.0)),
                "win":            int(info.get("win",    0)),
            })
            self.logger.record("game/episode_reward", ep["r"])
            self.logger.record("game/episode_length", ep["l"])
            self.logger.record("game/kills",          info.get("kills",  0))
            self.logger.record("game/deaths",         info.get("deaths", 0))
            self.logger.record("game/score",          info.get("score",  0.0))
            self.logger.record("game/win_rate",       info.get("win",    0))
            self.logger.record("perf/steps_per_sec",
                self.num_timesteps / max(time.time() - self._start_time, 1))

        if len(self._buffer) >= self._flush_every:
            self._flush()
        return True

    def _on_training_end(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        keys   = list(self._buffer[0].keys())
        arrays = {k: np.array([row[k] for row in self._buffer]) for k in keys}
        with h5py.File(self.stats_path, "a") as f:
            grp = f.require_group("episodes")
            for k, arr in arrays.items():
                if k in grp:
                    ds = grp[k]
                    old_len = ds.shape[0]
                    ds.resize(old_len + len(arr), axis=0)
                    ds[old_len:] = arr
                else:
                    grp.create_dataset(k, data=arr, maxshape=(None,), compression="gzip")
        self._buffer.clear()