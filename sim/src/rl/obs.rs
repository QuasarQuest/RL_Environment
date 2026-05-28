// src/rl/obs.rs
//
// Observation space constants — single source of truth for the CNN input layout.
// build_obs_into lives in engine/obs.rs; Python consumers use these via pyo3.rs.
//
// Channel layout (17 channels total):
//
//  Spatial channels — one value per tile in the egocentric 25×25 crop:
//    0  OOB          out-of-bounds padding (1.0 = beyond grid edge)
//    1  BASE         own base tile
//    2  GOLD         gold item
//    3  OBSTACLE     impassable wall
//    4  ENEMY        enemy position (1.0)
//    5  ITEM_HEALTH  health pickup (one-hot, 1.0)
//    6  ITEM_AMMO    ammo pickup   (one-hot, 1.0)
//    7  ITEM_SPEED   speed pickup  (one-hot, 1.0)
//    8  ENEMY_HP     enemy HP at enemy position, normalised [0,1] (0 elsewhere)
//    9  ENEMY_AMMO   enemy ammo at enemy position, normalised [0,1] (0 elsewhere)
//   10  ENEMY_PATH   planned enemy path — decaying float (1.0=step1 → 0.0=step12)
//
//  Broadcast channels — agent-level scalars, uniform across all crop pixels:
//   11  CARRYING  gold_carried / AGENT_MAX_GOLD ∈ [0,1]
//   12  HEALTH    hearts / AGENT_MAX_HEARTS ∈ [0,1]
//   13  AMMO      ammo / AGENT_MAX_AMMO ∈ [0,1]
//
//  Broadcast channels — global navigation, map-size independent:
//   14  BASE_DX   (base_pos.x - agent.x) / grid_width  ∈ [-1,1]
//   15  BASE_DY   (base_pos.y - agent.y) / grid_height ∈ [-1,1]
//
//  Broadcast channel — episode progress (makes truncation bootstrap correct):
//   16  TIME_REMAINING  (match_ticks - tick) / match_ticks ∈ [0,1]
//
//  Minimap — 17×17 cells, ~3×3 tiles/cell for a 50×50 grid:
//    (MM_CHANNELS, MM_SIZE, MM_SIZE) = (3, 17, 17) = 867 floats
//    Appended after OBS_DIM in the flat buffer.
//    Channels: 0=obstacles, 1=enemy, 2=gold (all max-pooled over ~3×3 tiles)
//
//  Cluster features — 4 clusters × 3 floats = 12 floats:
//    Appended after MM section: [dx_norm, dy_norm, count_norm] per cluster slot.
//    Zero-padded when fewer than CLUSTER_K clusters exist.

pub const OBS_CHANNELS: usize = 17;

// ── Spatial channels ──────────────────────────────────────────────────────────

pub const CH_OOB:         usize = 0;
pub const CH_BASE:        usize = 1;
pub const CH_GOLD:        usize = 2;
pub const CH_OBSTACLE:    usize = 3;
pub const CH_ENEMY:       usize = 4;
// Item pickups — one binary channel per kind (was a single float-encoded channel;
// one-hot avoids forcing the CNN to learn an artificial ordering between kinds).
pub const CH_ITEM_HEALTH: usize = 5;
pub const CH_ITEM_AMMO:   usize = 6;
pub const CH_ITEM_SPEED:  usize = 7;
pub const CH_ENEMY_HP:    usize = 8;
pub const CH_ENEMY_AMMO:  usize = 9;
pub const CH_ENEMY_PATH:  usize = 10;

// ── Broadcast channels — agent scalars ────────────────────────────────────────

pub const CH_CARRYING: usize = 11;
pub const CH_HEALTH:   usize = 12;
pub const CH_AMMO:     usize = 13;

// ── Broadcast channels — global navigation ───────────────────────────────────

pub const CH_BASE_DX: usize = 14;
pub const CH_BASE_DY: usize = 15;

// ── Broadcast channel — episode progress ──────────────────────────────────────
// Fraction of the match still remaining. Makes the value function time-aware so
// the time-limit truncation bootstrap (see batch_vec_env.py) is unbiased.

pub const CH_TIME_REMAINING: usize = 16;

// ── Spatial crop dimensions ───────────────────────────────────────────────────
// 25×25 covers half the 50×50 map in each dimension — sufficient local context.

pub const OBS_CROP_SIZE: usize = 25;

pub const OBS_DIM:   usize                 = OBS_CHANNELS * OBS_CROP_SIZE * OBS_CROP_SIZE;
pub const OBS_SHAPE: (usize, usize, usize) = (OBS_CHANNELS, OBS_CROP_SIZE, OBS_CROP_SIZE);

// ── Minimap ───────────────────────────────────────────────────────────────────
// 17×17 → each cell covers ~3×3 tiles on a 50×50 grid, giving cluster-level detail.

pub const MM_CHANNELS: usize = 3;
pub const MM_SIZE:     usize = 17;

pub const MM_CH_OBSTACLE: usize = 0;
pub const MM_CH_ENEMY:    usize = 1;
pub const MM_CH_GOLD:     usize = 2;

pub const MM_DIM:   usize                 = MM_CHANNELS * MM_SIZE * MM_SIZE; // 867
pub const MM_SHAPE: (usize, usize, usize) = (MM_CHANNELS, MM_SIZE, MM_SIZE);

// ── Cluster features (appended after minimap) ─────────────────────────────────
// 4 clusters × (dx_norm, dy_norm, count_norm) = 12 floats.
// Matches CLUSTER_K in rl/action.rs and engine/clusters.rs.

pub const CLUSTER_K:        usize = 4;
pub const CLUSTER_FEATURES: usize = CLUSTER_K * 3; // 12 floats

// ── Total flat buffer size ────────────────────────────────────────────────────

/// OBS_DIM (main crop) + MM_DIM (minimap) + CLUSTER_FEATURES — written by build_obs_into.
pub const OBS_TOTAL: usize = OBS_DIM + MM_DIM + CLUSTER_FEATURES;
