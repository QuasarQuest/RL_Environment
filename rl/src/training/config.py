# rl/src/training/config.py
#
# Single source of truth for all training hyperparameters.
# Edit here — train.py and tune.py read from this module.
#
# PPO parameter guide:
#   n_steps        : steps collected per env per update. 1 full episode = 10000.
#   batch_size     : must divide n_steps evenly. Larger = more stable gradients.
#   n_epochs       : passes over collected data per update.
#   gamma          : discount factor. Must match GAMMA in reward.rs (0.99).
#   gae_lambda     : GAE smoothing. 0.95 is standard.
#   clip_range     : PPO clip. 0.2 is standard.
#   ent_coef       : entropy bonus. Decays via schedule in train.py.
#   vf_coef        : value function loss weight.
#   learning_rate  : Adam LR — decayed linearly via schedule in train.py.
#   normalize_reward: enabled — reduces variance from mixed reward scales.

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PpoConfig:
    # n_steps = 1 full episode (10 000 ticks × 1 env).
    # Larger rollouts give better GAE estimates for long-horizon tasks.
    n_steps: int = 10_000

    # batch_size: 10000 / 1000 = 10 minibatches per update.
    # Larger than v1 (500) → more stable gradient estimates per minibatch.
    batch_size: int = 1_000

    # More epochs: more reuse of expensive Rust rollout data.
    n_epochs: int = 15

    # Must match GAMMA constant in reward.rs exactly.
    gamma: float = 0.99

    gae_lambda: float = 0.95

    clip_range: float = 0.2

    # Entropy coefficient: start at 0.05 for strong early exploration.
    # Linearly decayed to 0.005 over training in train.py.
    ent_coef: float = 0.05

    vf_coef: float = 0.5

    # Learning rate: start at 3e-4, linearly decayed to 1e-5 in train.py.
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
    # Enabled: reward signal spans SURVIVAL_PENALTY (-0.005/tick × 10k = -50)
    # through DELIVERY_SCALE (20×score_delta). Normalisation stabilises training.
    normalize_reward: bool = True

    # PPO hyperparameters
    ppo: PpoConfig = field(default_factory=PpoConfig)

    # Device: "auto" selects CUDA if available (CPU preferred for MlpPolicy PPO)
    device: str = "cpu"

    # Random seed
    seed: Optional[int] = 42


@dataclass
class TuneConfig:
    """Optuna search space bounds for hyperparameter tuning."""
    n_trials:    int = 50
    n_timesteps: int = 1_000_000   # shorter runs per trial

    # PPO search bounds
    lr_min:          float = 1e-5
    lr_max:          float = 5e-4
    clip_range_min:  float = 0.1
    clip_range_max:  float = 0.3
    ent_coef_min:    float = 0.01
    ent_coef_max:    float = 0.1
    gamma_min:       float = 0.97
    gamma_max:       float = 0.999
    gae_lambda_min:  float = 0.9
    gae_lambda_max:  float = 0.99
    # Must divide n_steps=10000 evenly
    batch_size_opts: list  = field(default_factory=lambda: [500, 1000, 2000])
    n_epochs_opts:   list  = field(default_factory=lambda: [10, 15, 20])