"""Entropy coefficient schedule callback."""
from __future__ import annotations

from typing import Callable

from stable_baselines3.common.callbacks import BaseCallback


class EntropyCoefScheduleCallback(BaseCallback):
    """Drive `model.ent_coef` along a schedule each rollout.

    PPO's `ent_coef` is a plain float read inside the loss; SB3 ignores any
    callable passed there. This callback updates the attribute at the start of
    each rollout, mirroring how `learning_rate` and `clip_range` are natively
    handled.
    """

    def __init__(
            self,
            schedule: Callable[[float], float],
            total_timesteps: int,
            verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._schedule = schedule
        self._total = max(int(total_timesteps), 1)

    def _on_rollout_start(self) -> None:
        progress_remaining = max(0.0, min(1.0, 1.0 - (self.num_timesteps / self._total)))
        new_val = float(self._schedule(progress_remaining))
        self.model.ent_coef = new_val
        self.logger.record("train/ent_coef", new_val)

    def _on_step(self) -> bool:
        return True
