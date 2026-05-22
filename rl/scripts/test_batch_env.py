# rl/scripts/test_batch_env.py
#
# Sanity check for PyBatchEnv (Rust) and BatchVecEnv (Python/SB3 wrapper).
#
# Run from rl/:
#   python scripts/test_batch_env.py

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def find_config() -> str:
    here = Path(__file__).resolve()
    for parent in here.parents:
        c = parent / "assets" / "world" / "config.ron"
        if c.exists():
            return str(c)
    raise FileNotFoundError("Cannot find assets/world/config.ron")


def section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def fail(msg: str) -> None:
    print(f"  ✗  {msg}")
    sys.exit(1)


# ── 1. Raw PyBatchEnv (Rust) ──────────────────────────────────────────────────

section("1. PyBatchEnv — static metadata")
import atb

obs_shape   = atb.PyBatchEnv.obs_shape()
action_size = atb.PyBatchEnv.action_size()
obs_dim     = atb.PyBatchEnv.obs_dim()
print(f"  obs_shape={obs_shape}  action_size={action_size}  obs_dim={obs_dim}")
assert obs_dim == obs_shape[0] * obs_shape[1] * obs_shape[2]
ok("metadata consistent")

section("2. PyBatchEnv — construction + reset_all")
N = 8
config = find_config()
batch = atb.PyBatchEnv(N, config)
assert batch.n_envs() == N
ok(f"constructed {N} envs")

obs_list = batch.reset_all()
assert len(obs_list) == N, f"expected {N} obs, got {len(obs_list)}"
for i, flat in enumerate(obs_list):
    assert len(flat) == obs_dim, f"env {i}: flat len {len(flat)} != {obs_dim}"
ok("reset_all returns correct shapes")

section("3. PyBatchEnv — step_batch")
actions = [25] * N  # all Wait
obs_list2, rews, dones = batch.step_batch(actions)
assert len(obs_list2) == N and len(rews) == N and len(dones) == N
assert all(r < 0.0 for r in rews), f"Wait should give only tick penalty, got {rews}"
assert not any(dones), "should not be done after one step"
ok(f"step_batch: rewards={[round(r,4) for r in rews[:3]]}...")

section("4. PyBatchEnv — reset_env (individual)")
flat = batch.reset_env(0)
assert len(flat) == obs_dim
ok("reset_env(0) returns correct obs length")

section("5. PyBatchEnv — step to episode end, check auto-reset via step_batch_auto_reset")
# Use a small helper env to find match_duration_ticks from config
import re
with open(config) as f:
    text = f.read()
m = re.search(r'match_duration_ticks:\s*(\d+)', text)
match_ticks = int(m.group(1)) if m else 10_000
print(f"  match_duration_ticks = {match_ticks}")

# Run a single env to done with Wait actions
single = atb.PyBatchEnv(1, config)
single.reset_all()
found_done = False
for _ in range(match_ticks + 5):
    _, _, dones = single.step_batch([25])
    if dones[0]:
        found_done = True
        break
assert found_done, "episode never terminated"
ok(f"episode terminated at tick ≤ {match_ticks}")

# ── 2. BatchVecEnv (Python SB3 wrapper) ──────────────────────────────────────

section("6. BatchVecEnv — construction")
from env.batch_vec_env import BatchVecEnv

N = 4
vec = BatchVecEnv(N, config)
assert vec.num_envs == N
print(f"  {vec}")
ok("constructed")

section("7. BatchVecEnv — reset returns stacked ndarray")
obs = vec.reset()
assert obs.shape == (N, *obs_shape), f"expected {(N,)+obs_shape}, got {obs.shape}"
assert obs.dtype == np.float32
assert obs.min() >= 0.0 and obs.max() <= 1.0
ok(f"reset obs shape={obs.shape} dtype={obs.dtype}")

section("8. BatchVecEnv — step (step_async + step_wait)")
actions = np.zeros(N, dtype=np.int64)  # Wait action = 0... actually 25
actions[:] = 25
vec.step_async(actions)
obs2, rews, dones, infos = vec.step_wait()
assert obs2.shape == (N, *obs_shape)
assert rews.shape == (N,)
assert dones.shape == (N,)
assert len(infos) == N
assert all(r < 0.0 for r in rews), f"Wait should give tick penalty: {rews}"
ok(f"step: rews={rews.round(4).tolist()}  dones={dones.tolist()}")

section("9. BatchVecEnv — episode info in infos when done")
# Run to episode end on one of 4 envs by stepping a separate single-env vec
single_vec = BatchVecEnv(1, config)
single_vec.reset()
found_episode_info = False
for _ in range(match_ticks + 5):
    single_vec.step_async(np.array([25], dtype=np.int64))
    _, _, dones, infos = single_vec.step_wait()
    if dones[0]:
        assert "episode" in infos[0], f"no episode key in infos: {infos[0]}"
        assert "terminal_observation" in infos[0], "no terminal_observation in infos"
        ep = infos[0]["episode"]
        assert "r" in ep and "l" in ep
        print(f"  episode r={ep['r']:.3f}  l={ep['l']}")
        found_episode_info = True
        break
assert found_episode_info, "never reached done"
ok("episode info and terminal_observation present on done")

section("10. BatchVecEnv — env_is_wrapped(Monitor) returns True")
from stable_baselines3.common.monitor import Monitor
wrapped = vec.env_is_wrapped(Monitor)
assert wrapped == [True] * N, f"expected all True, got {wrapped}"
ok("env_is_wrapped(Monitor) = True")

section("11. BatchVecEnv — throughput smoke")
vec64 = BatchVecEnv(64, config)
vec64.reset()
actions64 = np.full(64, 25, dtype=np.int64)
N_STEPS = 200
t0 = time.perf_counter()
for _ in range(N_STEPS):
    vec64.step_async(actions64)
    vec64.step_wait()
elapsed = time.perf_counter() - t0
steps_per_sec = (N_STEPS * 64) / elapsed
print(f"  {N_STEPS} steps × 64 envs in {elapsed:.2f}s → {steps_per_sec:,.0f} steps/sec")
ok("throughput check complete")

print(f"\n{'═' * 50}")
print("  All checks passed.")
print('═' * 50)
