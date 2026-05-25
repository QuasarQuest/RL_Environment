"""Algorithm registry.

Each entry binds an SB3 algorithm class to a default policy string and a
config dataclass. `train.py` and `tune.py` resolve algorithms by name so
adding SAC/DQN later is a matter of populating this file, not editing the
training loop.

Algorithms
----------
ppo          : Standard PPO — no action masking.
maskable_ppo : PPO with per-step action masking (sb3-contrib).
               Requires the env to expose an `action_masks()` method.
               Use this for all stages — masks are stage-aware and fall back
               to all-valid in stage 6, so there is no downside to always
               using it.

Policy choice
-------------
Both PPO variants use "MlpPolicy" here because our observation space is flat
(8272,) and AtbCnnExtractor handles the CNN internally. SB3 routes "MlpPolicy"
through the custom features_extractor_class in ATB_POLICY_KWARGS.

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

    name:                str
    cls:                 Type[BaseAlgorithm]
    policy:              str
    supports_discrete:   bool
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
    policy="MlpPolicy",   # flat obs → AtbCnnExtractor handles CNN internally
    supports_discrete=True,
    supports_continuous=True,
)

# ---------------------------------------------------------------------------
# MaskablePPO — action masking via sb3-contrib.
#
# Install: pip install sb3-contrib
#
# The env must implement action_masks() returning np.ndarray[bool] of shape
# (action_size,) for AtbEnv / (n_envs, action_size) for BatchVecEnv.
# Stage-aware masks are defined in env/action_masks.py.
# ---------------------------------------------------------------------------


def _make_maskable_spec() -> AlgoSpec:
    if not _MASKABLE_AVAILABLE or MaskablePPO is None:
        raise ImportError(
            "MaskablePPO requires sb3-contrib. Install with:\n"
            "  pip install sb3-contrib"
        )
    return AlgoSpec(
        name="maskable_ppo",
        cls=MaskablePPO,
        policy="MlpPolicy",
        supports_discrete=True,
        supports_continuous=False,
    )


ALGOS: dict[str, AlgoSpec] = {
    "ppo": PPO_SPEC,
    **( {"maskable_ppo": _make_maskable_spec()} if _MASKABLE_AVAILABLE else {} ),
}


def get_algo(name: str) -> AlgoSpec:
    """Look up an algorithm spec by name with a helpful error message."""
    try:
        spec = ALGOS[name.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(ALGOS))
        raise ValueError(
            f"Unknown algorithm '{name}'. Available: {available}"
        ) from exc

    # Guard: if maskable_ppo was requested but sb3-contrib is missing, raise now.
    if name.lower() == "maskable_ppo" and not _MASKABLE_AVAILABLE:
        raise ImportError(
            "MaskablePPO requires sb3-contrib. Install with:\n"
            "  pip install sb3-contrib"
        )
    return spec
