"""SB3-compatible VecEnv backed by the Rayon-parallel Rust BatchEnv.

Observation protocol
--------------------
Rust returns a flat bytearray of shape (n_envs * OBS_TOTAL * 4 bytes).
np.frombuffer gives (n_envs, OBS_TOTAL) = (n_envs, 9629) — flat, not (C,H,W).
AtbCnnExtractor in policy.py splits crop + minimap + cluster features internally.

Score tracking
--------------
True episode score read from Rust via get_agents() on episode end.
One FFI call per done env per episode — negligible overhead.

Profiling
---------
Set ATB_PROFILE_STEPS=1 in env to print sim/total wall-time ratio every
100 step_wait calls. Lets you see whether throughput is sim-bound or
PPO-update bound without a full profiler run.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional, cast

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv

from network.extractor import ACTION_SIZE, OBS_TOTAL as _OBS_TOTAL

_OBS_DTYPE = np.float32
_OBS_FLAT_SHAPE = (_OBS_TOTAL,)

# Index of the RL agent (team 0) in the get_agents() tuple list.
_RL_AGENT_IDX = 0
# Index of the score field in the (x, y, team, gold_carried, score) tuple.
_SCORE_FIELD = 4

# Profiling: print sim/total ratio every N step_wait calls when enabled.
_PROFILE_ENABLED = os.environ.get("ATB_PROFILE_STEPS", "0") == "1"
_PROFILE_INTERVAL = 100


class BatchVecEnv(VecEnv):
    """Rayon-parallel vectorised env. Drop-in for SubprocVecEnv."""

    def __init__(
            self,
            n_envs: int,
            config_path: str,
            stage: int = 1,
            *,
            clip_reward: Optional[float] = None,
            reward_scale: float = 1.0,
    ) -> None:
        import atb

        self._batch: Any = atb.PyBatchEnv(n_envs, config_path)
        self._stage = stage

        # Flat 1D observation space — AtbCnnExtractor splits crop + minimap + cluster features.
        obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=_OBS_FLAT_SHAPE, dtype=_OBS_DTYPE
        )
        act_space = spaces.Discrete(ACTION_SIZE)

        super().__init__(n_envs, obs_space, act_space)

        self._clip_reward = float(clip_reward) if clip_reward is not None else None
        self._reward_scale = float(reward_scale)

        self._ep_rewards = np.zeros(n_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(n_envs, dtype=np.int32)

        self._pending_actions: Optional[np.ndarray] = None

        # ── Profiling state ──────────────────────────────────────────────────
        # _t_sim   : cumulative wall-time spent in Rust step_batch FFI call.
        # _t_total : cumulative wall-time spent in step_wait (sim + Python post).
        # _n_steps : step_wait calls since last report.
        self._t_sim: float = 0.0
        self._t_total: float = 0.0
        self._n_steps: int = 0

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _ba_to_obs(ba, n: int) -> np.ndarray:
        """Zero-copy bytearray → (n, OBS_TOTAL) numpy array."""
        return np.frombuffer(ba, dtype=_OBS_DTYPE).reshape(n, _OBS_TOTAL)

    def _rust_score(self, env_idx: int) -> float:
        """Read the true episode score from Rust for env_idx."""
        agents = self._batch.get_agents(env_idx)
        return float(agents[_RL_AGENT_IDX][_SCORE_FIELD])

    # ── VecEnv ─────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        obs_ba = self._batch.reset_all()
        self._ep_rewards[:] = 0.0
        self._ep_lengths[:] = 0
        return self._ba_to_obs(obs_ba, self.num_envs).copy()

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = actions

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        assert self._pending_actions is not None, "step_async must be called first"

        t_wait_start = time.perf_counter() if _PROFILE_ENABLED else 0.0

        t_sim_start = time.perf_counter() if _PROFILE_ENABLED else 0.0
        obs_ba, rews, dones = self._batch.step_batch(self._pending_actions.tolist())
        if _PROFILE_ENABLED:
            self._t_sim += time.perf_counter() - t_sim_start

        self._pending_actions = None

        obs = self._ba_to_obs(obs_ba, self.num_envs)
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
            # Read true score from Rust before reset.
            infos[i]["score"] = self._rust_score(int(i))
            infos[i]["win"] = 0

            new_ba = self._batch.reset_env(int(i))
            obs[i] = np.frombuffer(new_ba, dtype=_OBS_DTYPE).reshape(_OBS_TOTAL)
            self._ep_rewards[i] = 0.0
            self._ep_lengths[i] = 0

        result = obs.copy(), rews, dones, infos

        if _PROFILE_ENABLED:
            self._t_total += time.perf_counter() - t_wait_start
            self._n_steps += 1
            if self._n_steps >= _PROFILE_INTERVAL:
                sim_ms = (self._t_sim / self._n_steps) * 1000.0
                tot_ms = (self._t_total / self._n_steps) * 1000.0
                ratio = self._t_sim / self._t_total if self._t_total > 0 else 0.0
                py_ms = tot_ms - sim_ms
                # \r overwrites in-place when running standalone; keep \n for log capture.
                print(
                    f"[BatchVecEnv profile] "
                    f"step_wait {tot_ms:6.2f}ms  "
                    f"(sim {sim_ms:6.2f}ms = {ratio*100:5.1f}%, "
                    f"py-post {py_ms:5.2f}ms)  "
                    f"n_envs={self.num_envs}",
                    flush=True,
                )
                self._t_sim = 0.0
                self._t_total = 0.0
                self._n_steps = 0

        return result

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
        target = self._get_target_indices(indices)
        return [None] * len(target)

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        return [None] * self.num_envs

    def render(self, mode: str = "human") -> None:
        return None

    def _get_target_indices(self, indices) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    def __repr__(self) -> str:
        return (
            f"BatchVecEnv(n_envs={self.num_envs}, stage={self._stage}, "
            f"obs_flat={_OBS_TOTAL}, act={cast(spaces.Discrete, self.action_space).n})"
        )