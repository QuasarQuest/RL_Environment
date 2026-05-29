"""Feature extractor networks for ATB observations.

Both extractors consume a flat float32 buffer of shape (OBS_TOTAL,) = (11504,):
  [0      : 10625)   main egocentric crop  — (17, 25, 25) logically
  [10625  : 11492)   minimap               — ( 3, 17, 17) logically
  [11492  : 11504)   cluster features      — (12,) = 4 clusters × (dx, dy, count)

Channel layout (sim/src/rl/obs.rs is authoritative):
  Spatial:
    0  OOB           out-of-bounds padding
    1  BASE          own base tile
    2  GOLD          gold item
    3  OBSTACLE      impassable wall
    4  ENEMY         enemy position (binary)
    5  ITEM_HEALTH   health pickup (one-hot)
    6  ITEM_AMMO     ammo pickup   (one-hot)
    7  ITEM_SPEED    speed pickup  (one-hot)
    8  ENEMY_HP      enemy HP at enemy cell, normalised [0,1]
    9  ENEMY_AMMO    enemy ammo at enemy cell, normalised [0,1]
   10  ENEMY_PATH    planned enemy path, decaying [0,1]
  Broadcast:
   11  CARRYING      gold carried (broadcast)
   12  HEALTH        agent HP (broadcast)
   13  AMMO          agent ammo (broadcast)
   14  BASE_DX       direction to own base X (broadcast)
   15  BASE_DY       direction to own base Y (broadcast)
   16  TIME_REMAINING  fraction of match left ∈ [0,1] (broadcast)

  Minimap channels (3, 17, 17):
    0  MM_OBSTACLE
    1  MM_ENEMY
    2  MM_GOLD

  Cluster features (12 floats = 4 × [dx_norm, dy_norm, count_norm]):
    One entry per cluster slot, zero-padded when fewer than 4 clusters exist.

Slim architecture (v2)
----------------------
Original had 13.9M parameters — 83% in a single FC stack (10816→1024→512→256).
For a 50×50 1v1 gold-rush with 11 discrete actions this is ~9× oversized vs
comparable SOTA work (Griddly 1v1: ~500k–1M, Neural MMO 128-agent: ~2–4M).

Key changes vs original:
  crop_cnn  : added a 4th Conv(64→64, stride=2, pad=1)
              spatial: 25×25 → 13×13 → 7×7  (flat: 10816 → 3136)
  crop_head : 10816→1024→512→256  →  3136→256  (single Linear + LN)
  mm_cnn    : added stride=2 to 2nd conv
              spatial: 17×17 → 17×17 → 8×8  (flat: 7200 → 2048)
  mm_head   : 7200→128  →  2048→128  (same output, cheaper input)
  lstm      : 256 → 128  (set in default.yaml)

Parameter counts (features_dim=256):
  Component    Original     Slim       Factor
  crop_cnn       59,488     96,416     (4th conv added)
  crop_head  11,733,248    803,584     14.6×
  mm_cnn          5,088      5,088     —
  mm_head       921,984    262,528      3.5×
  cluster           416        416     —
  fusion        107,264    107,264     —
  lstm (256)    525,312        —
  lstm (128)        —      197,120      2.7×
  ─────────────────────────────────────────
  TOTAL      13,352,800  1,472,416      9.1×

Expected FPS: ~560 → ~2000–3000 (GTX 1650 Ti, 48 envs, n_steps=128).
Architecture is fixed across all 6 curriculum stages — sized for stage 6
self-play, which is still well within the 1–2M param SOTA sweet spot.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ── Observation layout (must match sim/src/rl/obs.rs) ────────────────────────

OBS_CHANNELS = 17
OBS_CROP_H = 25
OBS_CROP_W = 25
OBS_CROP_DIM = OBS_CHANNELS * OBS_CROP_H * OBS_CROP_W  # 10625

MM_CHANNELS = 3
MM_H = 17
MM_W = 17
MM_DIM = MM_CHANNELS * MM_H * MM_W  # 867

CLUSTER_FEATURES = 12  # 4 clusters × 3 floats

OBS_TOTAL = OBS_CROP_DIM + MM_DIM + CLUSTER_FEATURES  # 11504

# Action space size — defined here so env files don't need to import action_masks.
# Must match ACTION_SIZE in src/rl/action.rs.
ACTION_SIZE = 11


# ── MLP extractor (legacy — flat 1-D observations) ───────────────────────────

class AtbMlpExtractor(BaseFeaturesExtractor):
    """LayerNorm-stabilised MLP. Kept for A/B comparison against the CNN."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, features_dim), nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ── CNN + minimap + cluster extractor (slim v2) ───────────────────────────────

class AtbCnnExtractor(BaseFeaturesExtractor):
    """Three-branch extractor for egocentric crop, global minimap, and cluster features.

    Input: flat (OBS_TOTAL,) = (11504,) buffer — split internally.

    Crop branch  (17, 25, 25):
      Conv(17→32, 3×3, pad=1, s=1) → ReLU   # (32, 25, 25)
      Conv(32→64, 3×3, pad=1, s=1) → ReLU   # (64, 25, 25)
      Conv(64→64, 3×3, pad=1, s=2) → ReLU   # (64, 13, 13)
      Conv(64→64, 3×3, pad=1, s=2) → ReLU   # (64,  7,  7)  ← new
      Flatten → Linear(3136→256) → LayerNorm(256) → ReLU

    Minimap branch (3, 17, 17):
      Conv(3→16,  3×3, pad=1, s=1) → ReLU   # (16, 17, 17)
      Conv(16→32, 3×3, s=2)        → ReLU   # (32,  8,  8)  ← stride-2 added
      Flatten → Linear(2048→128) → LayerNorm(128) → ReLU

    Cluster branch (12,):
      Linear(12→32) → ReLU

    Fusion:
      cat([256, 128, 32]) → Linear(416→features_dim) → LayerNorm → ReLU
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # ── Crop CNN ──────────────────────────────────────────────────────────
        # 4 convs: 25×25 → 25×25 → 25×25 → 13×13 → 7×7
        # flat output: 64 × 7 × 7 = 3136
        self.crop_cnn = nn.Sequential(
            nn.Conv2d(OBS_CHANNELS, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            crop_flat = self.crop_cnn(
                torch.zeros(1, OBS_CHANNELS, OBS_CROP_H, OBS_CROP_W)
            ).shape[1]
        # Single FC layer: 3136 → features_dim — no intermediate bottleneck needed
        self.crop_head = nn.Sequential(
            nn.Linear(crop_flat, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

        # ── Minimap CNN ───────────────────────────────────────────────────────
        # stride=2 on 2nd conv: 17×17 → 17×17 → 8×8, flat = 32×8×8 = 2048
        self.mm_cnn = nn.Sequential(
            nn.Conv2d(MM_CHANNELS, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),
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
            nn.Linear(features_dim + 128 + 32, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        crop = obs[:, :OBS_CROP_DIM].reshape(-1, OBS_CHANNELS, OBS_CROP_H, OBS_CROP_W)
        mm = obs[:, OBS_CROP_DIM:OBS_CROP_DIM + MM_DIM].reshape(-1, MM_CHANNELS, MM_H, MM_W)
        cl = obs[:, OBS_CROP_DIM + MM_DIM:]

        crop_feat = self.crop_head(self.crop_cnn(crop))
        mm_feat = self.mm_head(self.mm_cnn(mm))
        cl_feat = self.cluster_head(cl)

        return self.fusion(torch.cat([crop_feat, mm_feat, cl_feat], dim=1))
