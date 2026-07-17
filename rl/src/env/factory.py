"""Environment construction factory.

This is the *only* place envs should be built. Both training and tuning
import from here, which guarantees that hyperparameters discovered during
tuning transfer to training without surprises.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
)

from env.atb_env import AtbEnv
from env.batch_vec_env import BatchVecEnv
from training.config import EnvConfig


def make_single_env(cfg: EnvConfig, rank: int = 0) -> Callable[[], gym.Env]:
    def _thunk() -> gym.Env:
        seed = cfg.seed + rank if cfg.seed is not None else None
        env: gym.Env = AtbEnv(cfg.config_path, stage=cfg.stage, seed=seed)
        if cfg.max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
        env = Monitor(env)
        return env
    return _thunk


def build_vec_env(cfg: EnvConfig, *, eval_mode: bool = False,
                  gamma: float | None = None) -> VecEnv:
    """`gamma` must be ppo.gamma when reward normalization is on — VecNormalize's
    running-return discount would otherwise silently default to 0.99."""
    if eval_mode:
        # Offset the eval seed past every training env so eval never replays
        # training env 0's episode stream.
        eval_cfg = replace(cfg, seed=cfg.seed + cfg.n_envs if cfg.seed is not None else None)
        vec_env: VecEnv = DummyVecEnv([make_single_env(eval_cfg, rank=0)])
    elif cfg.n_envs == 1:
        vec_env = DummyVecEnv([make_single_env(cfg, rank=0)])
    else:
        if cfg.max_episode_steps is not None:
            raise ValueError(
                "env.max_episode_steps is not supported with n_envs > 1 "
                "(BatchVecEnv episodes end at the Rust match timer; set "
                "match_duration_ticks in the world .ron instead)."
            )
        vec_env = BatchVecEnv(cfg.n_envs, cfg.config_path, stage=cfg.stage, seed=cfg.seed)

    if cfg.normalize_obs or cfg.normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=cfg.normalize_obs,
            norm_reward=cfg.normalize_reward and not eval_mode,
            clip_obs=cfg.clip_obs,
            training=not eval_mode,
            gamma=gamma if gamma is not None else 0.99,
        )

    return vec_env
