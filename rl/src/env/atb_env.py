from __future__ import annotations

from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "assets" / "world" / "config.ron").exists():
            return parent
    raise RuntimeError(f"Cannot find project root starting from {here}")


PROJECT_ROOT = _find_project_root()
DEFAULT_CONFIG = str(PROJECT_ROOT / "assets" / "world" / "config.ron")


class AtbEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config_path: str = DEFAULT_CONFIG) -> None:
        super().__init__()

        import os
        os.chdir(PROJECT_ROOT)
        import atb

        self._atb = atb
        self._env = atb.PyRlEnv(config_path)

        obs_dim = atb.PyRlEnv.obs_dim()
        action_size = atb.PyRlEnv.action_size()

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(action_size)

        self._episode_reward: float = 0.0
        self._episode_length: int = 0
        self._score: float = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._score = 0.0
        obs = np.array(self._env.reset(), dtype=np.float32)
        return obs, {}

    def step(self, action: int):
        obs_raw, reward, done = self._env.step(int(action))
        obs = np.array(obs_raw, dtype=np.float32)

        self._episode_reward += float(reward)
        self._episode_length += 1

        score_delta = max(0.0, float(reward))
        self._score += score_delta

        info: dict = {
            "score": self._score,
            "win": 0,
        }

        if done:
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._episode_length,
            }

        return obs, reward, done, False, info

    def render(self):
        pass

    def close(self):
        pass