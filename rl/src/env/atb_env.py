# rl/src/env/atb_env.py

import os
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Set cwd to project root before importing atb ─────────────────────────────
# The Rust sim loads assets/world/config.ron relative to cwd.
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

    Observation space : Box(55,) float32
                        Bounds are [-1, 1] for relative/normalised features.
                        Some features (e.g. exists flags) are in [0, 1].
                        Using [-1, 1] as a conservative outer bound — VecNormalize
                        will rescale further during training.

    Action space      : Discrete(26)
                        0..7   Move       (8 directions)
                        8..15  Attack     (8 directions)
                        16..23 RangedAttack (8 directions)
                        24     Drop
                        25     Wait
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()

        self._env = atb.PyRlEnv()

        obs_dim     = atb.PyRlEnv.obs_dim()      # 55
        action_size = atb.PyRlEnv.action_size()  # 26

        # Observation bounds.
        # Relative position features are in (-1, 1); distance / normalised
        # scalars in [0, 1]; flags in {0, 1}. Using (-inf, inf) here and
        # relying on VecNormalize (clip_obs=10) is also valid, but explicit
        # bounds help SB3 policy initialisation.
        self.observation_space = spaces.Box(
            low   = -1.0,
            high  =  1.0,
            shape = (obs_dim,),
            dtype = np.float32,
        )
        self.action_space = spaces.Discrete(action_size)

        # Episode accumulators — reset in reset()
        self.episode_reward: float = 0.0
        self.episode_length: int   = 0

        # Game stat accumulators exposed via info dict
        self._kills:  int = 0
        self._deaths: int = 0
        self._score:  int = 0

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_reward = 0.0
        self.episode_length = 0
        self._kills  = 0
        self._deaths = 0
        self._score  = 0
        obs = np.array(self._env.reset(), dtype=np.float32)
        return obs, {}

    def step(self, action: int):
        obs_raw, reward, done = self._env.step(int(action))
        obs = np.array(obs_raw, dtype=np.float32)

        self.episode_reward += float(reward)
        self.episode_length += 1

        # Build info dict — game stats are populated by Rust via pyo3 when
        # available; fall back to zero if the Rust side doesn't expose them yet.
        info: dict = {
            "kills":  self._kills,
            "deaths": self._deaths,
            "score":  self._score,
            "win":    0,
        }

        if done:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_length,
            }

        return obs, reward, done, False, info

    def render(self):
        pass

    def close(self):
        pass