// src/rl/obs.rs
//
// Observation space constants — single source of truth for the CNN input layout.
// build_obs_into lives in engine/obs.rs; Python consumers use these via pyo3.rs.

pub const OBS_CHANNELS: usize = 11;

// ── Spatial channels (one pixel = one tile in the agent's crop view) ──────────

pub const CH_OOB:      usize = 0; // out-of-bounds: tile is beyond grid edge
pub const CH_BASE:     usize = 1; // own base tile
pub const CH_GOLD:     usize = 2; // gold item
pub const CH_OBSTACLE: usize = 3; // impassable wall
pub const CH_ENEMY:    usize = 4; // enemy agent position

// Shared item channel — float-encoded by kind so a single channel covers all
// consumable items. CNN learns distinct values per item type.
pub const CH_ITEMS:    usize = 5;

// ── Broadcast channels (filled uniformly — agent-level scalars) ───────────────

pub const CH_CARRYING: usize = 6; // gold_carried / AGENT_MAX_GOLD ∈ [0, 1]
pub const CH_HEALTH:   usize = 7; // hearts / AGENT_MAX_HEARTS ∈ [0, 1]
pub const CH_AMMO:     usize = 8; // ammo  / AGENT_MAX_AMMO   ∈ [0, 1]

// ── Broadcast channels (navigation — global, works on any map size) ───────────

// Signed direction from agent to current goal (nearest gold when not full,
// own base when full), normalised by map dimensions → roughly ∈ [-1, 1].
// Allows navigation even when the goal is outside the local crop window.
pub const CH_GOAL_DX:  usize = 9;
pub const CH_GOAL_DY:  usize = 10;

// ── CH_ITEMS encoding ─────────────────────────────────────────────────────────

pub const ITEM_HEALTH: f32 = 1.0 / 3.0;
pub const ITEM_AMMO:   f32 = 2.0 / 3.0;
pub const ITEM_SPEED:  f32 = 1.0;

// ── Spatial dimensions ────────────────────────────────────────────────────────

pub const OBS_CROP_SIZE: usize = 25;

pub const OBS_DIM:   usize                  = OBS_CHANNELS * OBS_CROP_SIZE * OBS_CROP_SIZE;
pub const OBS_TOTAL: usize                  = OBS_DIM;
pub const OBS_SHAPE: (usize, usize, usize)  = (OBS_CHANNELS, OBS_CROP_SIZE, OBS_CROP_SIZE);
