// src/rl/obs.rs
//
// Observation space constants — single source of truth for the CNN input layout.
// build_obs_into lives in engine/obs.rs; Python consumers use these via pyo3.rs.
//
// Single-agent gold rush only: no enemies, no combat, no health/ammo/speed items.
// (Those channels were removed; combat lives in a separate future game mode.)
//
// Channel layout (8 channels total):
//
//  Spatial channels — one value per tile in the egocentric 25×25 crop:
//    0  OOB          out-of-bounds padding (1.0 = beyond grid edge)
//    1  BASE         own base tile
//    2  GOLD         gold item
//    3  OBSTACLE     impassable wall
//
//  Broadcast channels — agent-level scalars, uniform across all crop pixels:
//    4  CARRYING  gold_carried / AGENT_MAX_GOLD ∈ [0,1]
//
//  Broadcast channels — global navigation, map-size independent:
//    5  BASE_DX   (base_pos.x - agent.x) / grid_width  ∈ [-1,1]
//    6  BASE_DY   (base_pos.y - agent.y) / grid_height ∈ [-1,1]
//
//  Broadcast channel — episode progress (makes truncation bootstrap correct):
//    7  TIME_REMAINING  (match_ticks - tick) / match_ticks ∈ [0,1]
//
//  Minimap — 17×17 cells, ~3×3 tiles/cell for a 50×50 grid:
//    (MM_CHANNELS, MM_SIZE, MM_SIZE) = (2, 17, 17) = 578 floats
//    Appended after OBS_DIM in the flat buffer.
//    Channels: 0=obstacles, 1=gold (max-pooled over ~3×3 tiles)
//
//  Cluster features — 9 regions × 3 floats = 27 floats:
//    Appended after MM section: [dx_norm, dy_norm, count_norm] per region slot.
//    Region k is a fixed 3×3 grid cell (stable spatial ID, see engine/clusters.rs).
//    Zero for regions that currently hold no gold.

pub const OBS_CHANNELS: usize = 8;

// ── Spatial channels ──────────────────────────────────────────────────────────

pub const CH_OOB:      usize = 0;
pub const CH_BASE:     usize = 1;
pub const CH_GOLD:     usize = 2;
pub const CH_OBSTACLE: usize = 3;

// ── Broadcast channels — agent scalars ────────────────────────────────────────

pub const CH_CARRYING: usize = 4;

// ── Broadcast channels — global navigation ───────────────────────────────────

pub const CH_BASE_DX: usize = 5;
pub const CH_BASE_DY: usize = 6;

// ── Broadcast channel — episode progress ──────────────────────────────────────
// Fraction of the match still remaining. Makes the value function time-aware so
// the time-limit truncation bootstrap (see batch_vec_env.py) is unbiased.

pub const CH_TIME_REMAINING: usize = 7;

// ── Spatial crop dimensions ───────────────────────────────────────────────────
// 25×25 covers half the 50×50 map in each dimension — sufficient local context.

pub const OBS_CROP_SIZE: usize = 25;

pub const OBS_DIM:   usize                 = OBS_CHANNELS * OBS_CROP_SIZE * OBS_CROP_SIZE;
pub const OBS_SHAPE: (usize, usize, usize) = (OBS_CHANNELS, OBS_CROP_SIZE, OBS_CROP_SIZE);

// ── Minimap ───────────────────────────────────────────────────────────────────
// 17×17 → each cell covers ~3×3 tiles on a 50×50 grid, giving cluster-level detail.

pub const MM_CHANNELS: usize = 2;
pub const MM_SIZE:     usize = 17;

pub const MM_CH_OBSTACLE: usize = 0;
pub const MM_CH_GOLD:     usize = 1;

pub const MM_DIM:   usize                 = MM_CHANNELS * MM_SIZE * MM_SIZE; // 578
pub const MM_SHAPE: (usize, usize, usize) = (MM_CHANNELS, MM_SIZE, MM_SIZE);

// ── Cluster features (appended after minimap) ─────────────────────────────────
// 9 fixed regions × (dx_norm, dy_norm, count_norm) = 27 floats.
// Matches CLUSTER_K in rl/action.rs and engine/clusters.rs (3×3 region grid).

pub const CLUSTER_K:        usize = 9;
pub const CLUSTER_FEATURES: usize = CLUSTER_K * 3; // 27 floats

// ── Total flat buffer size ────────────────────────────────────────────────────

/// OBS_DIM (main crop) + MM_DIM (minimap) + CLUSTER_FEATURES — written by build_obs_into.
/// 5000 + 578 + 27 = 5605.
pub const OBS_TOTAL: usize = OBS_DIM + MM_DIM + CLUSTER_FEATURES;
