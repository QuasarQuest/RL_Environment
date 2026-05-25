"""Stage-aware action masks for MaskablePPO.

Action space layout (ACTION_SIZE = 26, defined in src/rl/action.rs):
  0– 7   Move          (N S E W NE NW SE SW)
  8–15   Attack        (N S E W NE NW SE SW)  — melee, no ammo cost
  16–23  RangedAttack  (N S E W NE NW SE SW)  — costs 1 ammo
  24     Drop
  25     Wait

Stage progression
-----------------
  Stage 1–3 : no enemy → Attack and RangedAttack are no-ops.
              Masking them reduces effective action space 26 → 10,
              which accelerates exploration significantly.
  Stage 4   : SimpleChaser enemy appears → melee Attack unlocked.
              RangedAttack still masked (agent has no ammo reward yet).
  Stage 5   : A* enemy + full item set → RangedAttack unlocked.
  Stage 6   : self-play, all actions valid.

Drop (24) is available from stage 3 onward where gold carrying matters,
but harmless to allow earlier (it does nothing when not carrying gold).
"""
from __future__ import annotations

import numpy as np

# ── Action index ranges (must match src/rl/action.rs) ────────────────────────

ACTION_SIZE = 26

_MOVE_START = 0  # 0–7
_MOVE_END = 8
_ATTACK_START = 8  # 8–15
_ATTACK_END = 16
_RANGED_START = 16  # 16–23
_RANGED_END = 24
_DROP = 24
_WAIT = 25

# ── Per-stage valid action sets ───────────────────────────────────────────────

_MOVE_ACTIONS = list(range(_MOVE_START, _MOVE_END))  # [0..7]
_ATTACK_ACTIONS = list(range(_ATTACK_START, _ATTACK_END))  # [8..15]
_RANGED_ACTIONS = list(range(_RANGED_START, _RANGED_END))  # [16..23]

STAGE_VALID_ACTIONS: dict[int, list[int]] = {
    # Stage 1: navigation only — no enemy, no combat items.
    1: _MOVE_ACTIONS + [_WAIT],

    # Stage 2: obstacles added — same action set as stage 1.
    2: _MOVE_ACTIONS + [_WAIT],

    # Stage 3: health/ammo/speed items added — Drop now meaningful.
    3: _MOVE_ACTIONS + [_DROP, _WAIT],

    # Stage 4: SimpleChaser enemy — melee Attack unlocked.
    4: _MOVE_ACTIONS + _ATTACK_ACTIONS + [_DROP, _WAIT],

    # Stage 5: A* enemy + ammo items — RangedAttack unlocked.
    5: _MOVE_ACTIONS + _ATTACK_ACTIONS + _RANGED_ACTIONS + [_DROP, _WAIT],

    # Stage 6: self-play — all actions valid.
    6: list(range(ACTION_SIZE)),
}


def get_action_mask(stage: int) -> np.ndarray:
    """Return a boolean mask of shape (ACTION_SIZE,) for the given stage.

    True  = action is valid and will be sampled.
    False = action is masked out (logit set to -inf by MaskablePPO).

    Unknown stages default to all-valid (conservative fallback).
    """
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    valid = STAGE_VALID_ACTIONS.get(stage, list(range(ACTION_SIZE)))
    mask[valid] = True
    return mask


def get_action_mask_batch(stage: int, n_envs: int) -> np.ndarray:
    """Return a boolean mask of shape (n_envs, ACTION_SIZE) for BatchVecEnv."""
    single = get_action_mask(stage)
    return np.broadcast_to(single, (n_envs, ACTION_SIZE)).copy()
