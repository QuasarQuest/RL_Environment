"""Policy network definitions.

Two feature extractors live here:

* `AtbMlpExtractor` — legacy, for 1D observations. Kept so the previous
  flat-vector pipeline can be revived by switching `ATB_POLICY_KWARGS` back
  to `ATB_MLP_POLICY_KWARGS`.

* `AtbCnnExtractor` — current, for (C, H, W) grid observations. A small
  conv stack with no spatial pooling because our grid is already tiny
  (25×25). Output: `features_dim` floats.

`ATB_POLICY_KWARGS` is the canonical kwargs dict consumed by `train.py`.
Today it points at the CNN extractor.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ---------------------------------------------------------------------------
# MLP extractor (legacy, kept for A/B comparison)
# ---------------------------------------------------------------------------
class AtbMlpExtractor(BaseFeaturesExtractor):
    """LayerNorm-stabilised MLP feature extractor for flat observations."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) != 1:
            raise ValueError(
                f"AtbMlpExtractor requires a 1D observation, got "
                f"{observation_space.shape}. Use AtbCnnExtractor for images."
            )
        obs_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ---------------------------------------------------------------------------
# CNN extractor (current, for spatial grid observations)
# ---------------------------------------------------------------------------
class AtbCnnExtractor(BaseFeaturesExtractor):
    """Small CNN feature extractor for (C, H, W) grid observations.

    Design notes
    ------------
    - No pooling: 25×25 is already small, max-pooling would discard signal.
    - 3×3 convs with padding=1 preserve spatial size, so the model can in
      principle reason about object position across all H×W cells.
    - The final Linear collapses the spatial map to a flat feature vector
      that PPO's policy/value heads can consume as usual.
    - The conv depth (32→64→64) is intentionally modest. With 5 input
      channels and a 25×25 footprint, a deeper stack overfits the toy task.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) != 3:
            raise ValueError(
                f"AtbCnnExtractor requires a 3D observation (C, H, W), got "
                f"{observation_space.shape}. Use AtbMlpExtractor for flat obs."
            )
        c, h, w = observation_space.shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flat dim with a dry run instead of hard-coding it —
        # if we change crop size later, this still works.
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_dim = self.conv(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(flat_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(obs))


# ---------------------------------------------------------------------------
# Policy kwargs
# ---------------------------------------------------------------------------
# MLP kwargs (legacy — preserved for switch-back ability)
ATB_MLP_POLICY_KWARGS = dict(
    features_extractor_class=AtbMlpExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64], vf=[64]),
)

# CNN kwargs (current)
# normalize_images=False is CRITICAL — SB3's CnnPolicy assumes uint8 [0,255]
# and divides inputs by 255. Our observations are already in [0, 1], so
# letting that happen would silently corrupt training.
ATB_CNN_POLICY_KWARGS = dict(
    features_extractor_class=AtbCnnExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64], vf=[64]),
    normalize_images=False,
)

# Canonical kwargs consumed by train.py. Swap the RHS to switch back to MLP.
ATB_POLICY_KWARGS = ATB_CNN_POLICY_KWARGS


# Re-exported for callers that want the policy class directly.
AtbPolicy = ActorCriticPolicy