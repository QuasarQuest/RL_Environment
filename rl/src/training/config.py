from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from env.atb_env import PROJECT_ROOT


def stage_config_path(stage: int) -> str:
    path = PROJECT_ROOT / "assets" / "world" / f"config_stage{stage}.ron"
    if not path.exists():
        raise FileNotFoundError(f"No config for stage {stage}: {path}")
    return str(path)


@dataclass
class PpoConfig:
    n_steps: int = 2_000
    batch_size: int = 200
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.05
    ent_coef_final: float = 0.005
    vf_coef: float = 0.5
    learning_rate: float = 3e-4
    learning_rate_final: float = 1e-5
    max_grad_norm: float = 0.5


@dataclass
class TrainConfig:
    total_timesteps: int = 2_000_000
    checkpoint_freq: int = 50_000
    run_name: str = "run"
    stage: int = 1

    models_dir: Path = Path("runs/models")
    stats_dir: Path = Path("runs/stats")
    tensorboard_dir: Path = Path("runs/tensorboard")

    clip_reward: bool = True
    clip_reward_max: float = 10.0
    reward_scale: float = 1.0

    normalize_obs: bool = True
    normalize_reward: bool = True

    ppo: PpoConfig = field(default_factory=PpoConfig)

    device: str = "cpu"
    seed: Optional[int] = 42

    @property
    def config_path(self) -> str:
        return stage_config_path(self.stage)


@dataclass
class TuneConfig:
    n_trials: int = 50
    n_timesteps: int = 500_000
    stage: int = 1

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
    batch_size_opts: list = field(default_factory=lambda: [100, 200, 500])
    n_epochs_opts: list = field(default_factory=lambda: [5, 10, 15])