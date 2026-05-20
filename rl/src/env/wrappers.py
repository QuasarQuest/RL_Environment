"""Reward-side Gymnasium wrappers for `AtbEnv`.

Each wrapper is intentionally single-purpose so they can be composed and
toggled independently from config. Observation normalisation is handled by
SB3's `VecNormalize` in the env factory, not here.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class ClipRewardWrapper(gym.RewardWrapper):
    """Symmetric reward clipping. Stabilises training when rare large rewards
    occur (e.g. terminal kill/death events) before `VecNormalize` has built up
    a representative running variance.
    """

    def __init__(self, env: gym.Env, max_abs: float = 10.0) -> None:
        super().__init__(env)
        if max_abs <= 0.0:
            raise ValueError(f"max_abs must be > 0, got {max_abs}")
        self._max_abs = float(max_abs)

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, -self._max_abs, self._max_abs))


class RewardScaleWrapper(gym.RewardWrapper):
    """Multiplicative reward scaling. Useful when reward magnitudes are far
    from the unit scale PPO's value head prefers.
    """

    def __init__(self, env: gym.Env, scale: float = 1.0) -> None:
        super().__init__(env)
        self._scale = float(scale)

    def reward(self, reward: float) -> float:
        return reward * self._scale
