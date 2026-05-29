"""Environment construction factory.

This is the *only* place envs should be built. Both training and tuning
import from here, which guarantees that hyperparameters discovered during
tuning transfer to training without surprises.
"""
from __future__ import annotations

from typing import Callable

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
        env: gym.Env = AtbEnv(cfg.config_path, stage=cfg.stage)
        if cfg.max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
        env = Monitor(env)
        if cfg.seed is not None:
            env.reset(seed=cfg.seed + rank)
        return env
    return _thunk


def build_vec_env(cfg: EnvConfig, *, eval_mode: bool = False) -> VecEnv:
    if eval_mode:
        vec_env: VecEnv = DummyVecEnv([make_single_env(cfg, rank=0)])
    elif cfg.n_envs == 1:
        vec_env = DummyVecEnv([make_single_env(cfg, rank=0)])
    else:
        vec_env = BatchVecEnv(cfg.n_envs, cfg.config_path, stage=cfg.stage)

    if cfg.normalize_obs or cfg.normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=cfg.normalize_obs,
            norm_reward=cfg.normalize_reward and not eval_mode,
            clip_obs=cfg.clip_obs,
            training=not eval_mode,
        )

    return vec_env
