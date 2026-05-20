"""Training and tuning configuration dataclasses.

These are the *only* place to define defaults. `train.py` and `tune.py`
construct them and never duplicate values.

Observation normalisation note
------------------------------
`normalize_obs` defaults to False because our CNN observations are already
in [0, 1] with semantic per-channel meaning (binary masks, broadcast
indicator planes). Running them through `VecNormalize`'s running-mean
subtraction would destroy that semantics. Re-enable only if returning to
the legacy flat-vector observation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from env.atb_env import PROJECT_ROOT


def stage_config_path(stage: int) -> str:
    """Resolve `assets/world/config_stage{N}.ron` with a useful error message.

    Stage support is currently unused (stage=1 falls back to config.ron) —
    kept so curriculum staging can be added without touching this signature.
    """
    if stage <= 1:
        # No staged config yet — use the canonical single config.
        path = PROJECT_ROOT / "assets" / "world" / "config.ron"
        if not path.exists():
            raise FileNotFoundError(f"No world config: {path}")
        return str(path)
    path = PROJECT_ROOT / "assets" / "world" / f"config_stage{stage}.ron"
    if not path.exists():
        raise FileNotFoundError(f"No config for stage {stage}: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# PPO hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class PpoConfig:
    """PPO hyperparameters.

    `*_final` fields are the end-of-training values for linear schedules; if
    a `*_final` equals its base counterpart, the schedule degenerates to a
    constant (handled in `schedules.linear_schedule`).
    """

    n_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Scheduled fields.
    learning_rate: float = 3e-4
    learning_rate_final: float = 1e-5
    clip_range: float = 0.2
    clip_range_final: float = 0.2  # constant by default
    ent_coef: float = 0.05
    ent_coef_final: float = 0.005

    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


# ---------------------------------------------------------------------------
# Training-loop configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # Identity / stage.
    stage: int = 1
    run_name: str = "run"
    algo: str = "ppo"

    # Budget.
    total_timesteps: int = 2_000_000
    checkpoint_freq: int = 50_000
    eval_freq: int = 50_000
    eval_episodes: int = 10

    # Output directories (relative to the `rl/` package root).
    models_dir: Path = Path("runs/models")
    stats_dir: Path = Path("runs/stats")
    tensorboard_dir: Path = Path("runs/tensorboard")

    # Env wrappers.
    n_envs: int = 16
    max_episode_steps: Optional[int] = None
    clip_reward: bool = True
    clip_reward_max: float = 10.0
    reward_scale: float = 1.0
    # See module docstring: CNN obs are already in [0, 1].
    normalize_obs: bool = False
    normalize_reward: bool = True

    # Algo hyperparameters. Only one of these will be active at a time;
    # when more algos are added, swap `ppo` for an `algo_config` union or a
    # registry-driven loader.
    ppo: PpoConfig = field(default_factory=PpoConfig)

    # Runtime.
    device: str = "auto"
    seed: Optional[int] = 42

    @property
    def config_path(self) -> str:
        return stage_config_path(self.stage)


# ---------------------------------------------------------------------------
# Optuna search space
# ---------------------------------------------------------------------------
@dataclass
class TuneConfig:
    n_trials: int = 50
    n_timesteps: int = 500_000
    stage: int = 1
    pruning: bool = True

    lr_min: float = 1e-5
    lr_max: float = 5e-4
    clip_range_min: float = 0.1
    clip_range_max: float = 0.3
    ent_coef_min: float = 0.01
    ent_coef_max: float = 0.1
    gamma_min: float = 0.97
    gamma_max: float = 0.999
    gae_lambda_min: float = 0.9
    gae_lambda_max: float = 0.99
    batch_size_opts: list[int] = field(default_factory=lambda: [100, 200, 500])
    n_epochs_opts: list[int] = field(default_factory=lambda: [5, 10, 15])