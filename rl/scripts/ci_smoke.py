#!/usr/bin/env python3
"""CI smoke test: Rust↔Python contract, seeded determinism, short CPU training.

Run locally with:  rl/.venv/bin/python rl/scripts/ci_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import atb
import numpy as np

CFG = str(Path(__file__).resolve().parents[2] / "assets" / "world" / "config_stage1.ron")


def rollout(seed: int) -> float:
    """Checksum of a masked-random rollout — equal iff the sim is deterministic."""
    env = atb.PyBatchEnv(3, CFG, seed)
    total = float(np.frombuffer(env.reset_all(), dtype=np.float32).sum())
    rng = np.random.default_rng(0)
    for _ in range(100):
        masks = np.asarray(env.action_masks(), dtype=bool).reshape(3, -1)
        acts = [int(rng.choice(np.flatnonzero(m))) for m in masks]
        ob, rews, dones = env.step_batch(acts)
        total += float(np.frombuffer(ob, dtype=np.float32).sum()) + sum(rews)
        for j, done in enumerate(dones):
            if done:
                env.reset_env(j)
    return total


def main() -> None:
    from env.batch_vec_env import _assert_rust_python_contract
    _assert_rust_python_contract(atb)
    assert rollout(7) == rollout(7), "seeded rollout not reproducible"
    assert rollout(7) != rollout(8), "different seeds produced identical rollouts"
    print("contract + determinism OK")

    from sb3_contrib import MaskablePPO

    from env.factory import build_vec_env
    from network.policy import ATB_POLICY_KWARGS
    from training.config import EnvConfig
    from training.smdp import SmdpDiscountCallback, smdp_buffer_class

    env = build_vec_env(
        EnvConfig(stage=1, n_envs=4, seed=3, normalize_obs=False, normalize_reward=True),
        gamma=0.99,
    )
    model = MaskablePPO(
        "MlpPolicy", env, n_steps=64, batch_size=128, n_epochs=1,
        rollout_buffer_class=smdp_buffer_class("maskable_ppo"),
        policy_kwargs=ATB_POLICY_KWARGS, verbose=0, device="cpu",
    )
    model.learn(512, callback=[SmdpDiscountCallback()])
    env.close()
    print("training smoke OK")


if __name__ == "__main__":
    main()
