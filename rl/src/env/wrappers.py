# src/env/wrappers.py
#
# Optional wrappers applied on top of AtbEnv.
# VecNormalize (obs normalization) is applied at the VecEnv level in train.py.
# Put any additional reward shaping or obs transforms here.

import numpy as np
import gymnasium as gym


class RewardScaleWrapper(gym.RewardWrapper):
    """
    Scales all rewards by a constant factor.
    Useful when switching between reward functions with different magnitudes.
    Default scale=1.0 is a no-op.
    """

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self._scale = scale

    def reward(self, reward: float) -> float:
        return reward * self._scale


class ClipRewardWrapper(gym.RewardWrapper):
    """
    Clips rewards to [-max_abs, max_abs].
    Prevents large kill/delivery spikes from dominating early training.
    """

    def __init__(self, env: gym.Env, max_abs: float = 10.0):
        super().__init__(env)
        self._max_abs = max_abs

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, -self._max_abs, self._max_abs))