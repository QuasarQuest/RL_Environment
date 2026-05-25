"""Periodic and rolling-best model checkpointing."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class CheckpointCallback(BaseCallback):
    """Save model snapshots periodically and on rolling-best reward.

    The rolling window protects against false "new best" events from a
    single lucky episode. For rigorous best-model tracking use
    `EvalWithVecNorm` instead.

    save_freq is compared against `num_timesteps` (total env steps) rather
    than `n_calls` (number of _on_step invocations). With n_envs=48 one
    _on_step call = 48 steps, so an n_calls check would fire ~48x too often
    relative to the configured timestep budget.
    """

    def __init__(
            self,
            save_freq: int,
            ckpt_dir: Path,
            vec_normalize: Optional[VecNormalize] = None,
            rolling_window: int = 100,
            verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.ckpt_dir = Path(ckpt_dir)
        self.vec_normalize = vec_normalize
        self._recent_rewards: deque[float] = deque(maxlen=rolling_window)
        self._best_mean: float = -float("inf")
        self._last_save_step: int = 0
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save_step >= self.save_freq:
            self._save(f"step_{self.num_timesteps}")
            self._last_save_step = self.num_timesteps

        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._recent_rewards.append(float(info["episode"]["r"]))

        if len(self._recent_rewards) == self._recent_rewards.maxlen:
            mean_r = float(np.mean(self._recent_rewards))
            if mean_r > self._best_mean:
                self._best_mean = mean_r
                self._save("best_rolling")
                if self.verbose:
                    print(
                        f"  ✓ New rolling-best mean reward {mean_r:.3f} "
                        f"(window={self._recent_rewards.maxlen}) @ step "
                        f"{self.num_timesteps}"
                    )
        return True

    def _save(self, tag: str) -> None:
        path = self.ckpt_dir / tag
        self.model.save(str(path))
        if self.vec_normalize is not None:
            self.vec_normalize.save(str(path) + "_vecnorm.pkl")
