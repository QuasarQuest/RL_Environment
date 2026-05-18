# src/model/policy.py
#
# Custom policy network for SB3 PPO.
# Architecture is defined here and plugged into PPO via policy_kwargs.
#
# Changing the net:
#   - Edit AtbMlpExtractor to change hidden layers / activation.
#   - Add a CNN or attention block here if you extend the obs space.
#   - The rest of the training pipeline (train.py, export.py) doesn't change.
#
# SB3 docs on custom policies:
# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym


class AtbMlpExtractor(BaseFeaturesExtractor):
    """
    Shared MLP trunk used by both actor and critic.

    Input  : obs vector (53 floats)
    Output : feature vector (features_dim floats)

    Architecture (default):
        Linear(53 → 256) → LayerNorm → ReLU
        Linear(256 → 256) → LayerNorm → ReLU
        Linear(256 → 128) → ReLU

    LayerNorm stabilizes training with VecNormalize and sparse rewards.
    Swap ReLU for ELU or Tanh here if you want to experiment.
    """

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


# Policy kwargs to pass to PPO(policy_kwargs=ATB_POLICY_KWARGS)
ATB_POLICY_KWARGS = dict(
    features_extractor_class  = AtbMlpExtractor,
    features_extractor_kwargs = dict(features_dim=128),
    # Separate actor/critic heads after the shared trunk
    net_arch = dict(pi=[64], vf=[64]),
)

# Type alias — swap this to a different SB3 policy class if needed
AtbPolicy = ActorCriticPolicy