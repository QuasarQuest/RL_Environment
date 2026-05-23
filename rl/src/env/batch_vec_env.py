"""SB3-compatible VecEnv backed by the Rayon-parallel Rust BatchEnv.

Observation protocol:
    Rust returns a flat Python bytearray of shape [n_envs * OBS_TOTAL * 4 bytes].
    np.frombuffer(ba, dtype=np.float32) creates a zero-copy view; reshape gives
    (n_envs, C, H, W) without any Python float object allocation.
    This replaces the old list[list[float]] path which required ~200K Python
    float allocations per step.

Score tracking:
    The true episode score (gold deposited × deposit value) is read from Rust
    via get_agents() on episode end. This is a single FFI call per done env
    per episode — negligible overhead — and gives the exact Rust-side score
    rather than an approximation from accumulated normalised rewards.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv

_OBS_DTYPE = np.float32

# Index of the RL agent (team 0) in the get_agents() tuple list.
_RL_AGENT_IDX = 0
# Index of the score field in the (x, y, team, gold_carried, score) tuple.
_SCORE_FIELD = 4


class BatchVecEnv(VecEnv):
    """Rayon-parallel vectorised env. Drop-in for SubprocVecEnv."""

    def __init__(
            self,
            n_envs: int,
            config_path: str,
            *,
            clip_reward: Optional[float] = None,
            reward_scale: float = 1.0,
    ) -> None:
        import atb

        self._batch = atb.PyBatchEnv(n_envs, config_path)

        obs_shape = atb.PyBatchEnv.obs_shape()
        action_size = atb.PyBatchEnv.action_size()

        obs_space = spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=_OBS_DTYPE)
        act_space = spaces.Discrete(action_size)

        super().__init__(n_envs, obs_space, act_space)

        self._clip_reward = float(clip_reward) if clip_reward is not None else None
        self._reward_scale = float(reward_scale)

        self._ep_rewards = np.zeros(n_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(n_envs, dtype=np.int32)

        self._pending_actions: Optional[np.ndarray] = None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _ba_to_obs(self, ba, shape) -> np.ndarray:
        """Zero-copy bytearray → numpy view. Call .copy() if ownership needed."""
        return np.frombuffer(ba, dtype=_OBS_DTYPE).reshape(shape)

    def _rust_score(self, env_idx: int) -> float:
        """Read the true episode score from Rust for env_idx.

        get_agents() returns [(x, y, team, gold_carried, score), ...].
        We take agents[0] (RL agent, team 0) and its score field.
        Called only on episode end — one FFI call per done env per episode.
        """
        agents = self._batch.get_agents(env_idx)
        return float(agents[_RL_AGENT_IDX][_SCORE_FIELD])

    # ── VecEnv ─────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        obs_ba = self._batch.reset_all()
        self._ep_rewards[:] = 0.0
        self._ep_lengths[:] = 0
        return self._ba_to_obs(obs_ba, (self.num_envs, *self.observation_space.shape)).copy()

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = actions

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        assert self._pending_actions is not None, "step_async must be called first"

        obs_ba, rews, dones = self._batch.step_batch(self._pending_actions.tolist())
        self._pending_actions = None

        obs = self._ba_to_obs(obs_ba, (self.num_envs, *self.observation_space.shape))
        rews = np.array(rews, dtype=np.float32)
        dones = np.array(dones, dtype=bool)

        if self._clip_reward is not None:
            rews = np.clip(rews, -self._clip_reward, self._clip_reward)
        rews *= self._reward_scale

        self._ep_rewards += rews
        self._ep_lengths += 1

        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for i in np.where(dones)[0]:
            infos[i]["terminal_observation"] = obs[i].copy()
            infos[i]["episode"] = {
                "r": float(self._ep_rewards[i]),
                "l": int(self._ep_lengths[i]),
            }
            # FIX: read true score from Rust instead of accumulating
            # normalised rewards. get_agents() is called before reset_env()
            # so the score reflects the completed episode, not a fresh state.
            infos[i]["score"] = self._rust_score(int(i))
            infos[i]["win"] = 0

            new_ba = self._batch.reset_env(int(i))
            obs[i] = np.frombuffer(new_ba, dtype=_OBS_DTYPE).reshape(
                self.observation_space.shape
            )
            self._ep_rewards[i] = 0.0
            self._ep_lengths[i] = 0

        return obs.copy(), rews, dones, infos

    def close(self) -> None:
        pass

    # ── SB3 extras ─────────────────────────────────────────────────────────────

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> list[bool]:
        indices = self._get_target_indices(indices)
        if wrapper_class is Monitor:
            return [True] * len(indices)
        return [False] * len(indices)

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return [None] * len(self._get_target_indices(indices))

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        pass

    def env_method(
            self,
            method_name: str,
            *method_args: Any,
            indices=None,
            **method_kwargs: Any,
    ) -> list[Any]:
        return [None] * len(self._get_target_indices(indices))

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        return [None] * self.num_envs

    def render(self, mode: str = "human") -> None:
        return None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_target_indices(self, indices) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    def __repr__(self) -> str:
        return (
            f"BatchVecEnv(n_envs={self.num_envs}, "
            f"obs={self.observation_space.shape}, "
            f"act={self.action_space.n})"
        )