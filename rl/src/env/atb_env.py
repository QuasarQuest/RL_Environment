"""Gymnasium adapter for single-env eval/debug runs.

For training use BatchVecEnv (via factory.build_vec_env) which is faster.
This class exists so the factory's eval path and DummyVecEnv work correctly.

Observation layout
------------------
Rust returns a flat float32 buffer of shape (OBS_TOTAL,) = (10231,):
  [0    : 8750)   main egocentric crop  — (14, 25, 25)
  [8750 : 10195)  minimap               — ( 5, 17, 17)
  [10195: 10231)  cluster features      — (36,)

AtbCnnExtractor in policy.py splits and reshapes internally.

Termination semantics
---------------------
Rust ends an episode only at match_duration_ticks; the agent respawns on death
rather than terminating. So the Rust `done` is a time-limit TRUNCATION, not a
true MDP termination — we report it as `truncated`, which lets SB3 bootstrap the
final-state value (a terminal state would be treated as value 0). See
BatchVecEnv for the same reasoning on the training path.
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

# Index of the RL agent (team 0) and the score field in the get_agents() tuple
# (x, y, team, gold_carried, score) — mirrors batch_vec_env.
_RL_AGENT_IDX = 0
_SCORE_FIELD = 4


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

        # Fail loudly on a stale Rust build whose observation/action layout
        # disagrees with network.extractor. Shared with the training path.
        from env.batch_vec_env import _assert_rust_python_contract
        _assert_rust_python_contract(atb)

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

        # Match-timer end is a truncation, not a termination (the agent respawns
        # on death). Reporting truncated=True makes SB3 bootstrap V(final_obs).
        done = bool(dones[0])
        terminated = False
        truncated = done

        self._episode_reward += reward
        self._episode_length += 1
        # True score = gold banked at base (read from Rust), NOT a sum of positive
        # rewards — matches BatchVecEnv's score so eval and training agree.
        self._score = float(self._env.get_agents(0)[_RL_AGENT_IDX][_SCORE_FIELD])

        info: dict[str, Any] = {"score": self._score}
        if done:
            info["episode"] = {"r": self._episode_reward, "l": self._episode_length}

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Per-step action mask (ACTION_SIZE,) bool for MaskablePPO / MaskableEvalCallback.

        Reflects the state after the most recent step/reset. DummyVecEnv.env_method
        ("action_masks") routes here for the eval path.
        """
        return np.asarray(self._env.action_masks(), dtype=bool)

    def render(self):
        return None

    def close(self):
        return None
