"""Stage-aware action masks for the 11-action POI-based action space.

Action space layout (ACTION_SIZE = 11, defined in src/rl/action.rs):
  0– 3   NavigateToCluster(0..3)  — top-4 gold clusters by density
  4      NavigateToBase           — return to base and deposit
  5      NavigateToHealth         — nearest health pickup
  6      NavigateToAmmo           — nearest ammo pickup
  7      NavigateToEnemy          — nearest enemy (for hunting)
  8      MeleeAttack              — auto-target lowest-HP enemy in melee range
  9      RangedAttack             — auto-target lowest-HP enemy in ranged range
  10     Wait

Stage progression
-----------------
  Stage 1–2 : gold + base only — health/ammo/combat masked.
  Stage 3   : health + ammo items added — nav to pickups unlocked.
  Stage 4   : SimpleChaser enemy — navigate to enemy + melee unlocked.
  Stage 5+  : BehaviorTree enemy + ammo items — ranged attack unlocked.

Note: these masks are provided for optional use with MaskablePPO.
Default training uses RecurrentPPO without masking — the agent learns
that invalid actions (e.g. ranged attack with no ammo) return zero reward.
"""
from __future__ import annotations

import numpy as np

# ── Action indices (must match src/rl/action.rs) ──────────────────────────────

ACTION_SIZE = 11

_CLUSTER_ACTIONS = list(range(0, 4))   # NavigateToCluster 0-3
_NAV_BASE        = 4
_NAV_HEALTH      = 5
_NAV_AMMO        = 6
_NAV_ENEMY       = 7
_MELEE           = 8
_RANGED          = 9
_WAIT            = 10

# ── Per-stage valid action sets ───────────────────────────────────────────────

STAGE_VALID_ACTIONS: dict[int, list[int]] = {
    # Stage 1: open map, gold only — navigate to clusters/base, wait.
    1: _CLUSTER_ACTIONS + [_NAV_BASE, _WAIT],

    # Stage 2: obstacles + speed boost — same nav set.
    2: _CLUSTER_ACTIONS + [_NAV_BASE, _WAIT],

    # Stage 3: health + ammo pickups added.
    3: _CLUSTER_ACTIONS + [_NAV_BASE, _NAV_HEALTH, _NAV_AMMO, _WAIT],

    # Stage 4: SimpleChaser enemy — hunting and melee unlocked.
    4: _CLUSTER_ACTIONS + [_NAV_BASE, _NAV_HEALTH, _NAV_AMMO, _NAV_ENEMY, _MELEE, _WAIT],

    # Stage 5+: A* enemy, full items — ranged attack unlocked.
    5: list(range(ACTION_SIZE)),
    6: list(range(ACTION_SIZE)),
}


def get_action_mask(stage: int) -> np.ndarray:
    """Return a boolean mask of shape (ACTION_SIZE,) for the given stage."""
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    valid = STAGE_VALID_ACTIONS.get(stage, list(range(ACTION_SIZE)))
    mask[valid] = True
    return mask


def get_action_mask_batch(stage: int, n_envs: int) -> np.ndarray:
    """Return a boolean mask of shape (n_envs, ACTION_SIZE) for BatchVecEnv."""
    single = get_action_mask(stage)
    return np.broadcast_to(single, (n_envs, ACTION_SIZE)).copy()
