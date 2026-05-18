# rl/src/env/wrappers.py
#
# Optional wrappers applied on top of AtbEnv.
#
# Applied in train.py in this order:
#   AtbEnv → ClipRewardWrapper → RewardScaleWrapper → Monitor → VecNormalize
#
# VecNormalize (obs + reward normalisation) is applied at VecEnv level.
# Schedules (LR, ent_coef) are passed as callables to PPO in train.py.

import numpy as np
import gymnasium as gym


class RewardScaleWrapper(gym.RewardWrapper):
    """
    Scales all rewards by a constant factor.
    Default scale=1.0 is a no-op. Use when comparing runs with different
    reward magnitudes or when reward normalisation is disabled.
    """

    def __init__(self, env: gym.Env, scale: float = 1.0) -> None:
        super().__init__(env)
        self._scale = scale

    def reward(self, reward: float) -> float:
        return reward * self._scale


class ClipRewardWrapper(gym.RewardWrapper):
    """
    Clips rewards to [-max_abs, max_abs] before VecNormalize sees them.

    Prevents large kill/delivery spikes (DELIVERY_SCALE=20 × score_delta)
    from dominating early gradient updates. Works in tandem with
    VecNormalize(norm_reward=True) — clip first, then normalise.
    """

    def __init__(self, env: gym.Env, max_abs: float = 10.0) -> None:
        super().__init__(env)
        self._max_abs = max_abs

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, -self._max_abs, self._max_abs))