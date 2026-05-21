"""Algorithm registry.

Each entry binds an SB3 algorithm class to a default policy string and a
config dataclass. `train.py` and `tune.py` resolve algorithms by name so
adding SAC/DQN later is a matter of populating this file, not editing the
training loop.

Today only PPO is wired up because the env exposes a Discrete action space
without a continuous variant. SAC/TD3 are listed as TODOs to make the
intent explicit.

Policy choice
-------------
`policy="CnnPolicy"` is paired with `ATB_CNN_POLICY_KWARGS` (and its
`normalize_images=False` flag) in policy.py. Switching back to the legacy
flat-vector pipeline requires:
  - here:        policy="MlpPolicy"
  - policy.py:   ATB_POLICY_KWARGS = ATB_MLP_POLICY_KWARGS
  - env.rs:      import obs::build_obs / reward::compute_reward
  - atb_env.py:  swap observation_space back to Box(shape=(obs_dim,))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Type, runtime_checkable

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule


# ---------------------------------------------------------------------------
# Protocol — the constructor surface that train.py actually calls.
#
# Every algorithm we might add (PPO, SAC, DQN …) accepts at least these
# kwargs. The Protocol lets the type checker validate call sites without
# forcing AlgoSpec.cls to be typed as a concrete class.
# ---------------------------------------------------------------------------

@runtime_checkable
class SB3AlgoConstructor(Protocol):
    """Structural type for any SB3 on- or off-policy algorithm constructor."""

    def __call__(
        self,
        policy: str | type,
        env: GymEnv,
        *,
        learning_rate: float | Schedule = 3e-4,
        policy_kwargs: dict[str, Any] | None = None,
        tensorboard_log: str | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        **kwargs: Any,           # algorithm-specific params (n_steps, etc.)
    ) -> BaseAlgorithm: ...


@dataclass
class AlgoSpec:
    """Static description of an RL algorithm."""

    name: str
    cls: Type[BaseAlgorithm]
    policy: str
    supports_discrete: bool
    supports_continuous: bool

    @property
    def constructor(self) -> SB3AlgoConstructor:
        """
        Return ``cls`` cast to ``SB3AlgoConstructor``.

        Use this at call sites that pass algorithm-specific kwargs (e.g.
        ``n_steps``, ``clip_range``) so the type checker validates the
        shared surface while ``**kwargs`` absorbs the algo-specific ones
        without spurious "Unexpected argument" warnings.
        """
        return self.cls  # type: ignore[return-value]


# --------------------------------------------------------------------------
# PPO — the only algorithm currently supported. On-policy, works with the
# discrete action space exposed by AtbEnv.
# --------------------------------------------------------------------------
PPO_SPEC = AlgoSpec(
    name="ppo",
    cls=PPO,
    policy="CnnPolicy",
    supports_discrete=True,
    supports_continuous=True,
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
