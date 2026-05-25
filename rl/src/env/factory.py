"""Environment construction factory.

This is the *only* place envs should be built. Both training and tuning
import from here, which guarantees that hyperparameters discovered during
tuning transfer to training without surprises.

Action masking
--------------
`stage` is forwarded to AtbEnv and BatchVecEnv so they can pre-compute the
correct action mask. When using MaskablePPO (algo=maskable_ppo), the env's
action_masks() method is called every step — no extra wiring needed here.

Multi-agent forward-compat
--------------------------
When PettingZoo support is added, this module will gain a parallel
`build_marl_env(cfg)` entry point. Today only `build_vec_env(cfg)` exists.
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
from env.wrappers import ClipRewardWrapper, RewardScaleWrapper
from training.config import EnvConfig


def make_single_env(cfg: EnvConfig, rank: int = 0) -> Callable[[], gym.Env]:
    """Return a thunk that builds one wrapped AtbEnv.

    A thunk (rather than a constructed env) is required by SB3's vec env
    constructors, which need to lazily instantiate envs inside worker
    processes when n_envs > 1.

    stage is forwarded so AtbEnv can pre-compute the correct action mask
    for MaskablePPO.
    """

    def _thunk() -> gym.Env:
        env: gym.Env = AtbEnv(cfg.config_path, stage=cfg.stage)

        if cfg.max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)

        if cfg.clip_reward:
            env = ClipRewardWrapper(env, max_abs=cfg.clip_reward_max)
        if cfg.reward_scale != 1.0:
            env = RewardScaleWrapper(env, scale=cfg.reward_scale)

        env = Monitor(env)

        if cfg.seed is not None:
            env.reset(seed=cfg.seed + rank)
        return env

    return _thunk


def build_vec_env(cfg: EnvConfig, *, eval_mode: bool = False) -> VecEnv:
    """Construct the vectorised env used for training or evaluation.

    Parameters
    ----------
    cfg
        Environment configuration (EnvConfig from Hydra).
    eval_mode
        When True, uses a single DummyVecEnv and sets VecNormalize to
        inference mode — no running-stat updates, no reward normalisation.
    """
    if eval_mode:
        # Single env for eval — DummyVecEnv wraps a thunk.
        vec_env: VecEnv = DummyVecEnv([make_single_env(cfg, rank=0)])
    elif cfg.n_envs == 1:
        vec_env = DummyVecEnv([make_single_env(cfg, rank=0)])
    else:
        # Batch env: Rayon-parallel, replaces SubprocVecEnv for training.
        # Reward clipping/scaling handled inside BatchVecEnv so VecNormalize
        # sees the same scale as the single-env path.
        vec_env = BatchVecEnv(
            cfg.n_envs,
            cfg.config_path,
            stage=cfg.stage,                                   # ← action mask
            clip_reward=cfg.clip_reward_max if cfg.clip_reward else None,
            reward_scale=cfg.reward_scale,
        )

    if cfg.normalize_obs or cfg.normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=cfg.normalize_obs,
            norm_reward=cfg.normalize_reward and not eval_mode,
            clip_obs=cfg.clip_obs,
            clip_reward=cfg.clip_reward_max,
            training=not eval_mode,
        )

    return vec_env
