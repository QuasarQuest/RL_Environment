"""Gymnasium adapter for single-env eval/debug runs.

For training use BatchVecEnv (via factory.build_vec_env) which is faster.
This class exists so the factory's eval path and DummyVecEnv work correctly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "assets" / "world" / "config.ron").exists():
            return parent
    raise RuntimeError(f"Cannot find project root starting from {here}")


PROJECT_ROOT = _find_project_root()
DEFAULT_CONFIG = str(PROJECT_ROOT / "assets" / "world" / "config.ron")

_OBS_DTYPE = np.float32


class AtbEnv(gym.Env):
    """Single-agent Gymnasium env backed by a 1-env PyBatchEnv.

    Score tracking note
    -------------------
    `self._score` accumulates every positive reward, which includes both
    PICKUP (0.5) and DEPOSIT (5.0) events. It is therefore an approximation
    of the true Rust-side `agent.score` (deposit-only). A precise score
    would require an additional `get_agents()` call per step, which is too
    expensive for the training hot path. The approximation is acceptable for
    logging; do not use it as a training signal.
    """

    metadata = {"render_modes": []}

    def __init__(
            self,
            config_path: str = DEFAULT_CONFIG,
            *,
            agent_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        config_path = str(Path(config_path).expanduser().resolve())

        import atb
        self._env = atb.PyBatchEnv(1, config_path)

        obs_shape = atb.PyBatchEnv.obs_shape()
        action_size = atb.PyBatchEnv.action_size()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=_OBS_DTYPE
        )
        self.action_space = spaces.Discrete(action_size)
        self._obs_shape = obs_shape
        self._episode_reward: float = 0.0
        self._episode_length: int = 0
        self._score: float = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._score = 0.0
        obs_ba = self._env.reset_all()
        obs = np.frombuffer(obs_ba, dtype=_OBS_DTYPE).reshape(self._obs_shape).copy()
        return obs, {}

    def step(self, action: int):
        obs_ba, rews, dones = self._env.step_batch([int(action)])
        obs = np.frombuffer(obs_ba, dtype=_OBS_DTYPE).reshape(self._obs_shape).copy()
        reward = float(rews[0])
        terminated = bool(dones[0])

        self._episode_reward += reward
        self._episode_length += 1
        if reward > 0.0:
            self._score += reward

        info: dict[str, Any] = {"score": self._score, "win": 0}
        if terminated:
            info["episode"] = {"r": self._episode_reward, "l": self._episode_length}

        return obs, reward, terminated, False, info

    def render(self):
        return None

    def close(self):
        return None