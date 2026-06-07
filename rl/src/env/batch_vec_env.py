"""SB3-compatible VecEnv backed by the Rayon-parallel Rust BatchEnv.

Observation protocol
--------------------
Rust returns a flat bytearray of shape (n_envs * OBS_TOTAL * 4 bytes).
np.frombuffer gives (n_envs, OBS_TOTAL) = (n_envs, 10222) — flat, not (C,H,W).
AtbCnnExtractor in policy.py splits crop + minimap + cluster features internally.

Score tracking
--------------
True episode score read from Rust via get_agents() on episode end.
One FFI call per done env per episode — negligible overhead.

Truncation vs termination
-------------------------
Rust ends an episode ONLY when the match timer expires (match_duration_ticks);
the agent respawns on death rather than terminating. Every `done` is therefore a
time-limit truncation, not a true MDP terminal. We surface this to SB3 via
`infos[i]["TimeLimit.truncated"] = True`, which is the flag SB3's PPO checks
before bootstrapping V(terminal_observation). Without it, the value target at
every episode boundary is wrongly treated as 0, biasing value estimates low.
(If a true terminal condition — win/loss — is ever added to the sim, expose a
per-env `truncated` flag from Rust and set this from that instead of `True`.)

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


def _assert_rust_python_contract(atb_module: Any) -> None:
    """Fail loudly if the compiled Rust extension and the Python observation
    layout disagree. Cheap insurance against a stale `maturin develop` build
    after changing OBS_CHANNELS / minimap / cluster constants on either side —
    a mismatch would otherwise silently misalign every env's observation.
    """
    rust_total = atb_module.PyBatchEnv.obs_total()
    if rust_total != _OBS_TOTAL:
        raise RuntimeError(
            f"Observation size mismatch: Rust exposes OBS_TOTAL={rust_total} but "
            f"network.extractor expects {_OBS_TOTAL}. Rebuild the atb extension "
            f"(`maturin develop`) after changing the Rust observation layout."
        )
    rust_actions = atb_module.PyBatchEnv.action_size()
    if rust_actions != ACTION_SIZE:
        raise RuntimeError(
            f"Action size mismatch: Rust exposes ACTION_SIZE={rust_actions} but "
            f"network.extractor expects {ACTION_SIZE}. Rebuild the atb extension."
        )


class BatchVecEnv(VecEnv):
    """Rayon-parallel vectorised env. Drop-in for SubprocVecEnv."""

    def __init__(
            self,
            n_envs: int,
            config_path: str,
            stage: int = 1,
    ) -> None:
        import atb

        _assert_rust_python_contract(atb)

        self._batch: Any = atb.PyBatchEnv(n_envs, config_path)
        self._stage = stage

        # Flat 1D observation space — AtbCnnExtractor splits crop + minimap + cluster features.
        obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=_OBS_FLAT_SHAPE, dtype=_OBS_DTYPE
        )
        act_space = spaces.Discrete(ACTION_SIZE)

        super().__init__(n_envs, obs_space, act_space)

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

    @property
    def batch(self) -> Any:
        """The underlying Rust ``PyBatchEnv``.

        Exposed for offline tooling (e.g. the trace recorder) that needs direct
        per-env introspection — get_agents / get_items / get_trace / get_tiles /
        reward_weights — which the SB3 VecEnv API doesn't surface. Training code
        should prefer the public VecEnv methods above.
        """
        return self._batch

    @staticmethod
    def _ba_to_obs(ba, n: int) -> np.ndarray:
        """Zero-copy bytearray → (n, OBS_TOTAL) numpy array.

        The returned array is a *view* over ``ba`` (numpy keeps ``ba`` alive via
        ``.base``), and ``ba`` is writable since it is a bytearray — so callers
        may patch reset rows in place (see step_wait's done-env handling).

        Why no defensive copy: the Rust ``step_batch``/``reset_*`` calls hand back
        a freshly-allocated bytearray every call (PyByteArray::new memcpies out of
        the reused obs_flat buffer), so each step's view already owns independent,
        Python-managed memory. SB3's rollout buffer copies obs into its own
        preallocated storage on ``add()``, so nothing downstream aliases this view
        across steps. An extra ``.copy()`` here would be a redundant ~2.5 MB/step
        memcpy (n_envs=64 × 10222 × 4 B).
        """
        return np.frombuffer(ba, dtype=_OBS_DTYPE).reshape(n, _OBS_TOTAL)

    def _rust_score(self, env_idx: int) -> float:
        """Read the true episode score from Rust for env_idx."""
        agents = self._batch.get_agents(env_idx)
        return float(agents[_RL_AGENT_IDX][_SCORE_FIELD])

    def _action_masks_all(self) -> np.ndarray:
        """Per-env action masks as a (n_envs, ACTION_SIZE) bool array.

        Reflects the state after the most recent step_batch / reset (Rust keeps
        the mask buffer in sync). MaskablePPO reaches this via env_method below.
        """
        flat = np.asarray(self._batch.action_masks(), dtype=bool)
        return flat.reshape(self.num_envs, ACTION_SIZE)

    # ── VecEnv ─────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        obs_ba = self._batch.reset_all()
        self._ep_rewards[:] = 0.0
        self._ep_lengths[:] = 0
        return self._ba_to_obs(obs_ba, self.num_envs)

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

        self._ep_rewards += rews
        self._ep_lengths += 1

        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for i in np.where(dones)[0]:
            infos[i]["terminal_observation"] = obs[i].copy()
            # Every episode end is a match-timer truncation (see module docstring),
            # so flag it as such — SB3 bootstraps V(terminal_observation) only when
            # this is True; otherwise it treats the boundary as a true terminal.
            infos[i]["TimeLimit.truncated"] = True
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

        # No defensive copy — see _ba_to_obs: obs is a view over this step's
        # freshly-allocated bytearray, and SB3 copies it into the rollout buffer.
        result = obs, rews, dones, infos

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
                    f"(sim {sim_ms:6.2f}ms = {ratio * 100:5.1f}%, "
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
        # MaskablePPO's get_action_masks(VecEnv) calls env_method("action_masks").
        if method_name == "action_masks":
            masks = self._action_masks_all()
            return [masks[i] for i in target]
        return [None] * len(target)

    def action_masks(self) -> np.ndarray:
        """Direct accessor (n_envs, ACTION_SIZE) — convenience alongside env_method."""
        return self._action_masks_all()

    def decision_telemetry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-env decision telemetry from the most recent step_batch.

        Returns (chosen_gold_dist, is_cluster, own_region_had_gold, skipped_own),
        each shape (n_envs,). chosen_gold_dist is Chebyshev distance to the gold the
        policy committed to (−1 for non-gold actions). Consumed by
        PolicyTelemetryCallback to log chosen-region distance + own-region skip rate.
        """
        dist, is_cluster, own_gold, skipped = self._batch.decision_telemetry()
        return (
            np.asarray(dist, dtype=np.int64),
            np.asarray(is_cluster, dtype=bool),
            np.asarray(own_gold, dtype=bool),
            np.asarray(skipped, dtype=bool),
        )

    def option_ticks(self) -> np.ndarray:
        """Per-env option length (sim ticks) from the most recent step_batch, shape
        (n_envs,). Drives the SMDP γ^k cross-option discount (see SmdpRolloutBuffer)."""
        return np.asarray(self._batch.option_ticks(), dtype=np.float32)

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
