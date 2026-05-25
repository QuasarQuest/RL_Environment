// src/rl/obs.rs
//
// Observation space constants — single source of truth for the CNN input layout.
// build_obs_into lives in engine/obs.rs; Python consumers use these via pyo3.rs.
//
// Channel layout (13 channels total):
//
//  Spatial channels — one value per tile in the egocentric 25×25 crop:
//    0  OOB       out-of-bounds padding (1.0 = beyond grid edge)
//    1  BASE      own base tile
//    2  GOLD      gold item
//    3  OBSTACLE  impassable wall
//    4  ENEMY     enemy position (1.0)
//    5  ITEMS     consumable items, float-encoded by kind
//    6  ENEMY_HP  enemy HP at enemy position, normalised [0,1] (0 elsewhere)
//    7  ENEMY_AMMO enemy ammo at enemy position, normalised [0,1] (0 elsewhere)
//
//  Broadcast channels — agent-level scalars, uniform across all 625 pixels:
//    8  CARRYING  gold_carried / AGENT_MAX_GOLD ∈ [0,1]
//    9  HEALTH    hearts / AGENT_MAX_HEARTS ∈ [0,1]
//   10  AMMO      ammo / AGENT_MAX_AMMO ∈ [0,1]
//
//  Broadcast channels — global navigation, map-size independent:
//   11  BASE_DX   (base_pos.x - agent.x) / grid_width  ∈ [-1,1]
//   12  BASE_DY   (base_pos.y - agent.y) / grid_height ∈ [-1,1]
//
//  Minimap — separate small tensor, NOT part of OBS_DIM:
//    (MM_CHANNELS, MM_SIZE, MM_SIZE) = (3, 7, 7) = 147 floats
//    Appended after OBS_DIM in the flat buffer → OBS_TOTAL = OBS_DIM + MM_DIM
//    Channels: 0=obstacles, 1=enemy, 2=gold (all max-pooled over ~3.5×3.5 tiles)

pub const OBS_CHANNELS: usize = 13;

// ── Spatial channels ──────────────────────────────────────────────────────────

pub const CH_OOB:        usize = 0;
pub const CH_BASE:       usize = 1;
pub const CH_GOLD:       usize = 2;
pub const CH_OBSTACLE:   usize = 3;
pub const CH_ENEMY:      usize = 4;
pub const CH_ITEMS:      usize = 5;
pub const CH_ENEMY_HP:   usize = 6;
pub const CH_ENEMY_AMMO: usize = 7;

// ── Broadcast channels — agent scalars ────────────────────────────────────────

pub const CH_CARRYING: usize = 8;
pub const CH_HEALTH:   usize = 9;
pub const CH_AMMO:     usize = 10;

// ── Broadcast channels — global navigation ───────────────────────────────────

/// Signed direction from agent to own base, normalised by map width/height.
/// Always points to base — gold navigation comes from CH_GOLD in the crop.
pub const CH_BASE_DX: usize = 11;
pub const CH_BASE_DY: usize = 12;

// ── CH_ITEMS encoding ─────────────────────────────────────────────────────────

pub const ITEM_HEALTH: f32 = 1.0 / 3.0;
pub const ITEM_AMMO:   f32 = 2.0 / 3.0;
pub const ITEM_SPEED:  f32 = 1.0;

// ── Spatial crop dimensions ───────────────────────────────────────────────────

pub const OBS_CROP_SIZE: usize = 25;

pub const OBS_DIM:   usize                 = OBS_CHANNELS * OBS_CROP_SIZE * OBS_CROP_SIZE;
pub const OBS_SHAPE: (usize, usize, usize) = (OBS_CHANNELS, OBS_CROP_SIZE, OBS_CROP_SIZE);

// ── Minimap (appended after OBS_DIM in the flat buffer) ──────────────────────

pub const MM_CHANNELS: usize = 3;
pub const MM_SIZE:     usize = 7;

pub const MM_CH_OBSTACLE: usize = 0;
pub const MM_CH_ENEMY:    usize = 1;
pub const MM_CH_GOLD:     usize = 2;

pub const MM_DIM:   usize                 = MM_CHANNELS * MM_SIZE * MM_SIZE; // 147
pub const MM_SHAPE: (usize, usize, usize) = (MM_CHANNELS, MM_SIZE, MM_SIZE);

// ── Total flat buffer size ────────────────────────────────────────────────────

/// OBS_DIM (main crop) + MM_DIM (minimap) — written by build_obs_into.
pub const OBS_TOTAL: usize = OBS_DIM + MM_DIM;