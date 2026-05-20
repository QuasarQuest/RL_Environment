"""Policy network definitions.

`AtbMlpExtractor` is the current production extractor and feeds an
`MlpPolicy`. When the Rust side exposes an image-shaped observation, add
`AtbCnnExtractor` alongside it and a matching `ATB_CNN_POLICY_KWARGS` dict;
the training loop already resolves the policy string from config.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AtbMlpExtractor(BaseFeaturesExtractor):
    """LayerNorm-stabilised MLP feature extractor for flat observations."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) != 1:
            raise ValueError(
                f"AtbMlpExtractor requires a 1D observation, got "
                f"{observation_space.shape}. Use a CnnExtractor for images."
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


ATB_POLICY_KWARGS = dict(
    features_extractor_class=AtbMlpExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64], vf=[64]),
)


# Re-exported for callers that want the policy class directly.
AtbPolicy = ActorCriticPolicy
