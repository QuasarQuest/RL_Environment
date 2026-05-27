"""Feature extractor networks for ATB observations.

Both extractors consume a flat float32 buffer of shape (OBS_TOTAL,) = (9629,):
  [0       : 8750)   main egocentric crop  — (14, 25, 25) logically
  [8750    : 9617)   minimap               — ( 3, 17, 17) logically
  [9617    : 9629)   cluster features      — (12,) = 4 clusters × (dx, dy, count)

Channel layout (sim/src/rl/obs.rs is authoritative):
  Spatial:
    0  OOB           out-of-bounds padding
    1  BASE          own base tile
    2  GOLD          gold item
    3  OBSTACLE      impassable wall
    4  ENEMY         enemy position (binary)
    5  ITEMS         consumable items, float-encoded
    6  ENEMY_HP      enemy HP at enemy cell, normalised [0,1]
    7  ENEMY_AMMO    enemy ammo at enemy cell, normalised [0,1]
   13  ENEMY_PATH    planned enemy path, decaying [0,1]
  Broadcast:
    8  CARRYING      gold carried (broadcast)
    9  HEALTH        agent HP (broadcast)
   10  AMMO          agent ammo (broadcast)
   11  BASE_DX       direction to own base X (broadcast)
   12  BASE_DY       direction to own base Y (broadcast)

  Minimap channels (3, 17, 17):
    0  MM_OBSTACLE
    1  MM_ENEMY
    2  MM_GOLD

  Cluster features (12 floats = 4 × [dx_norm, dy_norm, count_norm]):
    One entry per cluster slot, zero-padded when fewer than 4 clusters exist.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ── Observation layout (must match sim/src/rl/obs.rs) ────────────────────────

OBS_CHANNELS = 14
OBS_CROP_H   = 25
OBS_CROP_W   = 25
OBS_CROP_DIM = OBS_CHANNELS * OBS_CROP_H * OBS_CROP_W  # 8750

MM_CHANNELS = 3
MM_H        = 17
MM_W        = 17
MM_DIM      = MM_CHANNELS * MM_H * MM_W  # 867

CLUSTER_FEATURES = 12  # 4 clusters × 3 floats

OBS_TOTAL = OBS_CROP_DIM + MM_DIM + CLUSTER_FEATURES  # 9629


# ── MLP extractor (legacy — flat 1-D observations) ───────────────────────────

class AtbMlpExtractor(BaseFeaturesExtractor):
    """LayerNorm-stabilised MLP. Kept for A/B comparison against the CNN."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),     nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, features_dim),               nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ── CNN + minimap + cluster extractor (current) ───────────────────────────────

class AtbCnnExtractor(BaseFeaturesExtractor):
    """Three-branch extractor for egocentric crop, global minimap, and cluster features.

    Input: flat (OBS_TOTAL,) = (9629,) buffer — split internally.

    Crop branch  (14, 25, 25):
      Conv(14→32, 3×3, pad=1) → ReLU               # (32, 25, 25)
      Conv(32→64, 3×3, pad=1) → ReLU               # (64, 25, 25)
      Conv(64→64, 3×3, stride=2, pad=1) → ReLU     # (64, 13, 13)
      Flatten → Linear(10816→1024) → ReLU
             → Linear(1024→512)   → ReLU
             → Linear(512→256)    → LayerNorm → ReLU

    Minimap branch (3, 17, 17):
      Conv(3→16,  3×3, pad=1) → ReLU               # (16, 17, 17)
      Conv(16→32, 3×3)        → ReLU               # (32, 15, 15)
      Flatten → Linear(7200→128) → LayerNorm → ReLU

    Cluster branch (12,):
      Linear(12→32) → ReLU

    Fusion:
      cat([256, 128, 32]) → Linear(416→features_dim) → LayerNorm → ReLU
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # ── Crop CNN ──────────────────────────────────────────────────────────
        self.crop_cnn = nn.Sequential(
            nn.Conv2d(OBS_CHANNELS, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            crop_flat = self.crop_cnn(
                torch.zeros(1, OBS_CHANNELS, OBS_CROP_H, OBS_CROP_W)
            ).shape[1]
        self.crop_head = nn.Sequential(
            nn.Linear(crop_flat, 1024), nn.ReLU(),
            nn.Linear(1024, 512),       nn.ReLU(),
            nn.Linear(512, 256),        nn.LayerNorm(256), nn.ReLU(),
        )

        # ── Minimap CNN ───────────────────────────────────────────────────────
        self.mm_cnn = nn.Sequential(
            nn.Conv2d(MM_CHANNELS, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),                      nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            mm_flat = self.mm_cnn(
                torch.zeros(1, MM_CHANNELS, MM_H, MM_W)
            ).shape[1]
        self.mm_head = nn.Sequential(
            nn.Linear(mm_flat, 128), nn.LayerNorm(128), nn.ReLU(),
        )

        # ── Cluster MLP ───────────────────────────────────────────────────────
        self.cluster_head = nn.Sequential(
            nn.Linear(CLUSTER_FEATURES, 32), nn.ReLU(),
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 32, features_dim), nn.LayerNorm(features_dim), nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        crop = obs[:, :OBS_CROP_DIM].reshape(-1, OBS_CHANNELS, OBS_CROP_H, OBS_CROP_W)
        mm   = obs[:, OBS_CROP_DIM:OBS_CROP_DIM + MM_DIM].reshape(-1, MM_CHANNELS, MM_H, MM_W)
        cl   = obs[:, OBS_CROP_DIM + MM_DIM:]

        crop_feat = self.crop_head(self.crop_cnn(crop))
        mm_feat   = self.mm_head(self.mm_cnn(mm))
        cl_feat   = self.cluster_head(cl)

        return self.fusion(torch.cat([crop_feat, mm_feat, cl_feat], dim=1))
