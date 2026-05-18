# rl/src/training/config.py
#
# Single source of truth for all training hyperparameters.
# Edit here — train.py and tune.py read from this module.
#
# PPO parameter guide:
#   n_steps        : steps collected per env per update. Set to 1 full episode.
#   batch_size     : must divide n_steps evenly. 10000 / 500 = 20 batches.
#   n_epochs       : passes over collected data per update. 10 is SB3 default.
#   gamma          : discount factor. 0.99 = care about future.
#   gae_lambda     : GAE smoothing. 0.95 is standard.
#   clip_range     : PPO clip. 0.2 is standard.
#   ent_coef       : entropy bonus. Higher = more exploration.
#   vf_coef        : value function loss weight.
#   learning_rate  : Adam LR.

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PpoConfig:
    # n_steps must equal or be a multiple of episode length (10000 ticks)
    # batch_size must divide n_steps * n_envs evenly
    n_steps:       int   = 10_000
    batch_size:    int   = 500     # 10000 / 500 = 20 batches, no truncation
    n_epochs:      int   = 10
    gamma:         float = 0.99
    gae_lambda:    float = 0.95
    clip_range:    float = 0.2
    ent_coef:      float = 0.01
    vf_coef:       float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5


@dataclass
class TrainConfig:
    # Total environment steps to train for
    total_timesteps: int = 10_000_000

    # Save a checkpoint every N steps
    checkpoint_freq: int = 100_000

    # Run name
    run_name: str = "run"

    # Output directories (relative to rl/ directory)
    models_dir:      str = "runs/models"
    stats_dir:       str = "runs/stats"
    tensorboard_dir: str = "runs/tensorboard"

    # Reward wrappers
    clip_reward:     bool  = True
    clip_reward_max: float = 10.0
    reward_scale:    float = 1.0

    # VecNormalize
    normalize_obs:    bool = True
    normalize_reward: bool = False   # reward already small scale

    # PPO hyperparameters
    ppo: PpoConfig = field(default_factory=PpoConfig)

    # Device: "auto" selects CUDA if available
    device: str = "auto"

    # Random seed
    seed: Optional[int] = 42


@dataclass
class TuneConfig:
    """Optuna search space bounds for hyperparameter tuning."""
    n_trials:    int = 50
    n_timesteps: int = 1_000_000   # shorter runs per trial

    # PPO search bounds
    lr_min:          float = 1e-5
    lr_max:          float = 1e-3
    clip_range_min:  float = 0.1
    clip_range_max:  float = 0.4
    ent_coef_min:    float = 0.0
    ent_coef_max:    float = 0.05
    gamma_min:       float = 0.9
    gamma_max:       float = 0.9999
    gae_lambda_min:  float = 0.8
    gae_lambda_max:  float = 0.99
    # Must all divide n_steps=10000 evenly
    batch_size_opts: list  = field(default_factory=lambda: [100, 200, 500, 1000, 2000])
    n_epochs_opts:   list  = field(default_factory=lambda: [5, 10, 20])