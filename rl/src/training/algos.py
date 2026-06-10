"""Algorithm registry.

Each entry binds an SB3 algorithm class to a default policy string and a
config dataclass. `train.py` and `tune.py` resolve algorithms by name so
adding SAC/DQN later is a matter of populating this file, not editing the
training loop.

Algorithms
----------
ppo           : Standard PPO — no action masking.
maskable_ppo  : PPO with per-step action masking (sb3-contrib). This is the
                DEFAULT algo (configs/train.yaml). The env exposes action_masks()
                (AtbEnv / BatchVecEnv) and MaskablePPO consumes them via
                env_method, so provably-useless actions (empty regions, base when
                empty-handed) are forbidden; Wait is never masked.

Policy choice
-------------
Both algos use "MlpPolicy": the observation space is flat (10222,) and
AtbCnnExtractor handles the CNN internally — SB3 routes the policy string through
the custom features_extractor_class set in the policy_kwargs (see network/policy.py).

Switching back to the legacy flat-vector pipeline:
  - policy.py:   ATB_POLICY_KWARGS = ATB_MLP_POLICY_KWARGS
  - atb_env.py / batch_vec_env.py: revert obs shape
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Type, runtime_checkable

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

try:
    from sb3_contrib import MaskablePPO

    _MASKABLE_AVAILABLE = True
except ImportError:
    _MASKABLE_AVAILABLE = False
    MaskablePPO = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Protocol — the constructor surface that train.py actually calls.
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
            **kwargs: Any,
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
        return self.cls  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# PPO — standard, no masking.
# ---------------------------------------------------------------------------

PPO_SPEC = AlgoSpec(
    name="ppo",
    cls=PPO,
    policy="MlpPolicy",  # flat obs → AtbCnnExtractor handles CNN internally
    supports_discrete=True,
    supports_continuous=True,
)


# ---------------------------------------------------------------------------
# MaskablePPO — action masking via sb3-contrib.
#
# Install: pip install sb3-contrib
#
# The env must implement action_masks() returning np.ndarray[bool] of shape
# (action_size,) for AtbEnv / (n_envs, action_size) for BatchVecEnv. The mask is
# computed in Rust (SimCore::action_mask) and surfaced through the env wrappers.
# ---------------------------------------------------------------------------


def _make_maskable_spec() -> AlgoSpec:
    assert MaskablePPO is not None  # only called when _MASKABLE_AVAILABLE
    return AlgoSpec(
        name="maskable_ppo",
        cls=MaskablePPO,
        policy="MlpPolicy",
        supports_discrete=True,
        supports_continuous=False,
    )


ALGOS: dict[str, AlgoSpec] = {
    "ppo": PPO_SPEC,
    **({"maskable_ppo": _make_maskable_spec()} if _MASKABLE_AVAILABLE else {}),
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
