"""Structured config schemas for Hydra.

Dataclasses define *types and structure only* — no hardcoded defaults.
All values live in ``configs/*.yaml``. Hydra populates these at runtime.

Adding a new algorithm
-----------------------
1. Add a ``FooConfig`` dataclass here.
2. Register it in ``ConfigStore`` below.
3. Add ``configs/foo/default.yaml``.
4. Wire it into ``algos.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from env.atb_env import PROJECT_ROOT


def resolve_config_path(stage: int) -> str:
    """Return the absolute path to the Bevy world config for *stage*."""
    path = PROJECT_ROOT / "assets" / "world" / f"config_stage{stage}.ron"
    if not path.exists():
        raise FileNotFoundError(f"World config not found for stage {stage}: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class PpoConfig:
    n_steps: int = MISSING
    batch_size: int = MISSING
    n_epochs: int = MISSING
    gamma: float = MISSING
    gae_lambda: float = MISSING
    learning_rate: float = MISSING
    learning_rate_final: float = MISSING
    clip_range: float = MISSING
    clip_range_final: float = MISSING
    ent_coef: float = MISSING
    ent_coef_final: float = MISSING
    vf_coef: float = MISSING
    max_grad_norm: float = MISSING
    target_kl: float = MISSING
    # pi/vf head depth — override per ppo config to match stage complexity.
    # Stages 1-3: [64]  (lean, fast convergence, low overfitting risk)
    # Stages 4-6: [128, 64]  (deeper for enemy/combat decisions)
    net_arch_pi: list[int] = field(default_factory=lambda: [64])
    net_arch_vf: list[int] = field(default_factory=lambda: [64])


@dataclass
class EnvConfig:
    stage: int = MISSING
    n_envs: int = MISSING
    seed: Optional[int] = None
    max_episode_steps: Optional[int] = None
    clip_reward: bool = MISSING
    clip_reward_max: float = MISSING
    clip_obs: float = 10.0              # VecNormalize obs clip — 10.0 is the SB3 default
    reward_scale: float = MISSING
    normalize_obs: bool = MISSING
    normalize_reward: bool = MISSING

    @property
    def config_path(self) -> str:
        return resolve_config_path(self.stage)


@dataclass
class TrainConfig:
    run_name: str = MISSING
    algo: str = MISSING
    total_timesteps: int = MISSING
    checkpoint_freq: int = MISSING
    eval_freq: int = MISSING
    eval_episodes: int = MISSING
    output_dir: str = MISSING           # root dir; each run gets its own subfolder
    device: str = MISSING
    resume: Optional[str] = None        # optional resume path


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class ATBConfig:
    """Root structured config. Hydra composes this from YAML groups."""
    train: TrainConfig = MISSING
    ppo: PpoConfig = MISSING
    env: EnvConfig = MISSING


# ---------------------------------------------------------------------------
# Optuna search space — one per algorithm, tune.py only, not Hydra-driven
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
    batch_size_opts: list = field(default_factory=lambda: [256, 512, 1024])
    n_epochs_opts: list = field(default_factory=lambda: [5, 10, 15])


# ---------------------------------------------------------------------------
# ConfigStore registration
# ---------------------------------------------------------------------------

def register_configs() -> None:
    """Call once at process startup (done in train.py / tune.py)."""
    cs = ConfigStore.instance()
    cs.store(name="atb_config", node=ATBConfig)
    cs.store(group="ppo", name="schema", node=PpoConfig)
    cs.store(group="env", name="schema", node=EnvConfig)
