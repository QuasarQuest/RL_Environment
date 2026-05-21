"""Environment construction factory.

This is the *only* place envs should be built. Both training and tuning
import from here, which guarantees that hyperparameters discovered during
tuning transfer to training without surprises.

Multi-agent forward-compat
--------------------------
When PettingZoo support is added, this module will gain a parallel
``build_marl_env(cfg)`` entry point. Today only ``build_vec_env(cfg)`` exists.
"""
from __future__ import annotations

from typing import Callable

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)

from env.atb_env import AtbEnv
from env.wrappers import ClipRewardWrapper, RewardScaleWrapper
from training.config import EnvConfig


def make_single_env(cfg: EnvConfig, rank: int = 0) -> Callable[[], gym.Env]:
    """Return a thunk that builds one wrapped environment.

    A thunk (rather than a constructed env) is required by SB3's vec env
    constructors, which need to lazily instantiate envs inside worker
    processes when ``n_envs > 1``.
    """

    def _thunk() -> gym.Env:
        env: gym.Env = AtbEnv(cfg.config_path)

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
        Environment configuration (``EnvConfig`` from Hydra).
    eval_mode
        When True, uses a single ``DummyVecEnv`` and sets ``VecNormalize``
        to inference mode — no running-stat updates, no reward normalisation.
    """
    # Eval always runs as a single env — no point spinning up n_envs workers
    # just to measure deterministic rollouts.
    n = 1 if eval_mode else cfg.n_envs
    thunks = [make_single_env(cfg, rank=i) for i in range(n)]

    if n == 1:
        vec_env: VecEnv = DummyVecEnv(thunks)
    else:
        vec_env = SubprocVecEnv(thunks, start_method="spawn")

    if cfg.normalize_obs or cfg.normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=cfg.normalize_obs,
            norm_reward=cfg.normalize_reward and not eval_mode,
            clip_obs=10.0,
            clip_reward=10.0,
            training=not eval_mode,
        )

    return vec_env
