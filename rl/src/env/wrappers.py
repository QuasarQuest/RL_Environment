from __future__ import annotations

import numpy as np
import gymnasium as gym


class ClipRewardWrapper(gym.RewardWrapper):
    def __init__(self, env: gym.Env, max_abs: float = 10.0) -> None:
        super().__init__(env)
        self._max_abs = max_abs

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, -self._max_abs, self._max_abs))


class RewardScaleWrapper(gym.RewardWrapper):
    def __init__(self, env: gym.Env, scale: float = 1.0) -> None:
        super().__init__(env)
        self._scale = scale

    def reward(self, reward: float) -> float:
        return reward * self._scale