from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AtbMlpExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
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

AtbPolicy = ActorCriticPolicy