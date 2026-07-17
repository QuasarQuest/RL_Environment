"""Entropy coefficient schedule callback."""
from __future__ import annotations

from collections.abc import Callable

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
            verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._schedule = schedule

    def _on_rollout_start(self) -> None:
        # SB3's own progress clock: 1 → 0 over THIS learn() call's budget. A
        # hand-rolled num_timesteps/total collapses on resume (cumulative counter
        # over a per-stage budget) and desyncs from the lr/clip schedules.
        progress_remaining = self.model._current_progress_remaining
        new_val = float(self._schedule(progress_remaining))
        # setattr: ent_coef lives on the PPO subclasses, not BaseAlgorithm.
        setattr(self.model, "ent_coef", new_val)  # noqa: B010
        self.logger.record("train/ent_coef", new_val)

    def _on_step(self) -> bool:
        return True
