"""Gymnasium adapter around the Rust `atb.PyRlEnv` simulation.

This module deliberately stays minimal: it adapts the PyO3 environment to the
Gymnasium API and nothing more. Reward shaping, normalisation, clipping, and
score/win accounting live in `env.wrappers`. Vectorisation lives in
`env.factory`.

Observation contract
--------------------
The Rust side returns a flat `list[float]` of length `obs_dim()`. We reshape
it to `(C, H, W)` using `obs_shape()` so PyTorch / SB3's `CnnPolicy` sees a
proper image tensor. The reshape is a zero-copy view.

Multi-agent forward-compat
--------------------------
`agent_id` is accepted but currently unused. When the Rust side gains
per-agent observations, this class will be wrapped (or subclassed) by a
PettingZoo `ParallelEnv` that constructs one `AtbEnv` per agent or queries
a shared simulation by id. Today it is single-agent and ignores the value.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _find_project_root() -> Path:
    """Walk upward to find the Rust project root (contains assets/world/config.ron)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "assets" / "world" / "config.ron").exists():
            return parent
    raise RuntimeError(f"Cannot find project root starting from {here}")


PROJECT_ROOT = _find_project_root()
DEFAULT_CONFIG = str(PROJECT_ROOT / "assets" / "world" / "config.ron")


class AtbEnv(gym.Env):
    """Single-agent Gymnasium adapter over `atb.PyRlEnv`."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG,
        *,
        agent_id: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Resolve the config to an absolute path so we never rely on the
        # process working directory. The previous implementation called
        # `os.chdir(PROJECT_ROOT)` here which mutated global state and broke
        # under SubprocVecEnv. We avoid that entirely.
        config_path = str(Path(config_path).expanduser().resolve())

        # Import lazily so unit tests that mock atb can patch it before import.
        import atb

        self._atb = atb
        self._agent_id = agent_id  # reserved for future multi-agent use
        self._env = atb.PyRlEnv(config_path)

        # Spatial observation: (C, H, W). Values are already in [0, 1] from
        # the Rust side, so we set a tight Box. `normalize_images=False` is
        # set on the CNN policy side so SB3 won't divide by 255.
        self._obs_shape: tuple[int, int, int] = atb.PyRlEnv.obs_shape()
        action_size = atb.PyRlEnv.action_size()

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._obs_shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(action_size)

        self._episode_reward: float = 0.0
        self._episode_length: int = 0
        self._score: float = 0.0

    # ------------------------------------------------------------------ API

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._score = 0.0
        obs_flat = self._env.reset()
        obs = np.asarray(obs_flat, dtype=np.float32).reshape(self._obs_shape)
        return obs, {}

    def step(self, action: int):
        obs_flat, reward, done = self._env.step(int(action))
        obs = np.asarray(obs_flat, dtype=np.float32).reshape(self._obs_shape)
        reward = float(reward)

        self._episode_reward += reward
        self._episode_length += 1
        # `score` is the cumulative positive-reward signal we surface to the
        # logger. Negative shaping terms still affect `episode_reward`.
        if reward > 0.0:
            self._score += reward

        # The Rust env exposes a single `done` flag. Without a separate
        # truncation signal we cannot distinguish a true terminal state from
        # a step-limit cutoff, which matters for GAE bootstrapping. Until
        # the Rust side exposes both flags, treat `done` as termination and
        # rely on a `gymnasium.wrappers.TimeLimit` (added in the env factory)
        # to set `truncated` for step-limit cases.
        terminated = bool(done)
        truncated = False

        info: dict[str, Any] = {"score": self._score, "win": 0}
        if terminated:
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._episode_length,
            }

        return obs, reward, terminated, truncated, info

    def render(self):  # pragma: no cover - no-op for headless training
        return None

    def close(self):  # pragma: no cover - PyO3 env owns its resources
        return None