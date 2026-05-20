"""Hyperparameter schedules.

PPO natively schedules `learning_rate` and `clip_range` (any callable taking
`progress_remaining` and returning a float works). It does *not* schedule
`ent_coef`: that attribute is read as a constant inside the loss. To get a
scheduled entropy coefficient we update `model.ent_coef` from a callback
(see `EntropyCoefScheduleCallback` in `training.callbacks`).
"""
from __future__ import annotations

from typing import Callable


def linear_schedule(start: float, end: float) -> Callable[[float], float]:
    """Linear schedule from `start` to `end`.

    SB3 passes `progress_remaining` which goes from 1.0 at the first step to
    0.0 at the last. So the returned value is `end` at training end and
    `start` at training start — exactly what users intuitively expect when
    writing `linear_schedule(3e-4, 1e-5)`.
    """
    if start == end:
        return lambda _: float(start)

    def _schedule(progress_remaining: float) -> float:
        return float(end + progress_remaining * (start - end))

    return _schedule
