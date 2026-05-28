"""Gymnasium adapter for single-env eval/debug runs.

For training use BatchVecEnv (via factory.build_vec_env) which is faster.
This class exists so the factory's eval path and DummyVecEnv work correctly.

Observation layout
------------------
Rust returns a flat float32 buffer of shape (OBS_TOTAL,) = (9629,):
  [0       : 8750)   main egocentric crop  — (14, 25, 25)
  [8750    : 9617)   minimap               — ( 3, 17, 17)
  [9617    : 9629)   cluster features      — (12,)

AtbCnnExtractor in policy.py splits and reshapes internally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from network.extractor import ACTION_SIZE, OBS_TOTAL as _OBS_TOTAL


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "assets" / "world" / "config_stage1.ron").exists():
            return parent
    raise RuntimeError(f"Cannot find project root starting from {here}")


PROJECT_ROOT = _find_project_root()
DEFAULT_CONFIG = str(PROJECT_ROOT / "assets" / "world" / "config_stage1.ron")

_OBS_DTYPE = np.float32
_OBS_FLAT_SHAPE = (_OBS_TOTAL,)


class AtbEnv(gym.Env):
    """Single-agent Gymnasium env backed by a 1-env PyBatchEnv."""

    metadata = {"render_modes": []}

    def __init__(
            self,
            config_path: str = DEFAULT_CONFIG,
            stage: int = 1,
    ) -> None:
        super().__init__()
        config_path = str(Path(config_path).expanduser().resolve())

        import atb
        self._env = atb.PyBatchEnv(1, config_path)
        self._stage = stage

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=_OBS_FLAT_SHAPE, dtype=_OBS_DTYPE
        )
        self.action_space = spaces.Discrete(ACTION_SIZE)

        self._episode_reward: float = 0.0
        self._episode_length: int = 0
        self._score: float = 0.0

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._score = 0.0
        obs_ba = self._env.reset_all()
        obs = np.frombuffer(obs_ba, dtype=_OBS_DTYPE).reshape(_OBS_FLAT_SHAPE).copy()
        return obs, {}

    def step(self, action: int):
        obs_ba, rews, dones = self._env.step_batch([int(action)])
        obs = np.frombuffer(obs_ba, dtype=_OBS_DTYPE).reshape(_OBS_FLAT_SHAPE).copy()
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