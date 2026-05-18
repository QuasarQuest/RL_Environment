# rl/src/env/atb_env.py

import os
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Set cwd to project root before importing atb ─────────────────────────────
# The Rust sim loads assets/world/config.ron relative to cwd.
# Walk up from this file until we find the assets/ directory.
def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "assets" / "world" / "config.ron").exists():
            return parent
    raise RuntimeError(
        f"Cannot find project root (assets/world/config.ron) "
        f"starting from {here}"
    )

_PROJECT_ROOT = _find_project_root()
os.chdir(_PROJECT_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

import atb


class AtbEnv(gym.Env):
    """
    Single-agent Gymnasium environment backed by the Bevy/Rust sim.

    Observation space : Box(53,) float32
    Action space      : Discrete(26)
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self._env = atb.PyRlEnv()

        obs_dim     = atb.PyRlEnv.obs_dim()
        action_size = atb.PyRlEnv.action_size()

        self.observation_space = spaces.Box(
            low   = -1.0,
            high  =  1.0,
            shape = (obs_dim,),
            dtype = np.float32,
        )
        self.action_space = spaces.Discrete(action_size)

        self.episode_reward: float = 0.0
        self.episode_length: int   = 0
        self._last_info:     dict[str, object] = {}

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_reward = 0.0
        self.episode_length = 0
        self._last_info     = {}
        obs = np.array(self._env.reset(), dtype=np.float32)
        return obs, {}

    def step(self, action: int):
        obs_raw, reward, done = self._env.step(int(action))
        obs = np.array(obs_raw, dtype=np.float32)

        self.episode_reward += reward
        self.episode_length += 1

        info: dict[str, object] = {
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
        }
        if done:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_length,
            }

        self._last_info = info
        return obs, reward, done, False, info

    def render(self):
        pass

    def close(self):
        pass