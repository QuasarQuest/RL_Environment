"""Algorithm registry.

Each entry binds an SB3 algorithm class to a default policy string and a
config dataclass. `train.py` and `tune.py` resolve algorithms by name so
adding SAC/DQN later is a matter of populating this file, not editing the
training loop.

Today only PPO is wired up because the env exposes a Discrete action space
without a continuous variant. SAC/TD3 are listed as TODOs to make the
intent explicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Type

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm


@dataclass
class AlgoSpec:
    """Static description of an RL algorithm and its defaults."""

    name: str
    cls: Type[BaseAlgorithm]
    policy: str
    supports_discrete: bool
    supports_continuous: bool
    default_kwargs: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# PPO — the only algorithm currently supported. On-policy, works with the
# discrete action space exposed by AtbEnv.
# --------------------------------------------------------------------------
PPO_SPEC = AlgoSpec(
    name="ppo",
    cls=PPO,
    policy="MlpPolicy",
    supports_discrete=True,
    supports_continuous=True,
    default_kwargs={
        "n_steps": 2_000,
        "batch_size": 200,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
        "learning_rate": 3e-4,
        "max_grad_norm": 0.5,
    },
)


ALGOS: dict[str, AlgoSpec] = {
    "ppo": PPO_SPEC,
    # "sac": ...   # requires continuous action space on the Rust side
    # "dqn": ...   # straightforward to add once we have an eval baseline
}


def get_algo(name: str) -> AlgoSpec:
    """Look up an algorithm spec by name with a helpful error message."""
    try:
        return ALGOS[name.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(ALGOS))
        raise ValueError(
            f"Unknown algorithm '{name}'. Available: {available}"
        ) from exc
