// src/rl/obs.rs
//
// Observation space constants — single source of truth for the CNN input layout.
// build_obs_into lives in engine/obs.rs; Python consumers mirror these in
// rl/network/extractor.py (keep the two in sync).
//
// Single-agent gold rush: no enemies, no combat. The map carries gold plus the
// flavour items the policy can see and navigate toward: the speed boost and the
// score multiplier. (The old Slow and Trap hazards were both removed — avoidance
// of a hazard is not learnable under the option/A* design.)
//
// Channel layout (12 channels total):
//
//  Spatial channels — one value per tile in the egocentric 25×25 crop:
//    0  OOB        out-of-bounds padding (1.0 = beyond grid edge)
//    1  BASE       own base tile
//    2  GOLD       gold item
//    3  OBSTACLE   impassable wall
//    4  SPEED      speed-boost item (1.0 where present)
//    5  MULT       score-multiplier item
//
//  Broadcast channels — agent-level scalars, uniform across all crop pixels:
//    6  CARRYING        gold_carried / AGENT_MAX_GOLD ∈ [0,1]
//    7  BASE_DX         (base_pos.x - agent.x) / grid_width  ∈ [-1,1]
//    8  BASE_DY         (base_pos.y - agent.y) / grid_height ∈ [-1,1]
//    9  TIME_REMAINING  (match_ticks - tick) / match_ticks ∈ [0,1]
//   10  SPEED_REMAINING speed_buff / SPEED_BUFF_MAX ∈ [0,1]
//   11  MULT_CHARGE     1.0 while a multiplier charge is held (consumed at deposit)
//
//  Minimap — 17×17 cells, ~3×3 tiles/cell for a 50×50 grid (4 channels):
//    0 obstacle, 1 gold, 2 speed, 3 mult (max-pooled over ~3×3 tiles).
//
//  Cluster features — 9 gold regions × 4 floats = 36 floats:
//    [dx_norm, dy_norm, pathdist_norm, count_norm] per fixed 3×3 region slot.
//    dx/dy point to the region's PATH-nearest gold (BFS around walls, not
//    Chebyshev); pathdist_norm is that true travel distance / (w+h) ∈ [0,1]
//    (1.0 = unreachable). See engine/clusters.rs and engine/nav.rs::dist_field.

pub const OBS_CHANNELS: usize = 12;

// ── Spatial channels ──────────────────────────────────────────────────────────

pub const CH_OOB:      usize = 0;
pub const CH_BASE:     usize = 1;
pub const CH_GOLD:     usize = 2;
pub const CH_OBSTACLE: usize = 3;
pub const CH_SPEED:    usize = 4;
pub const CH_MULT:     usize = 5;

// ── Broadcast channels — agent scalars ────────────────────────────────────────

pub const CH_CARRYING:        usize = 6;
pub const CH_BASE_DX:         usize = 7;
pub const CH_BASE_DY:         usize = 8;
pub const CH_TIME_REMAINING:  usize = 9;
pub const CH_SPEED_REMAINING: usize = 10;
pub const CH_MULT_CHARGE:     usize = 11; // broadcast: 1.0 while a multiplier charge is held

// ── Spatial crop dimensions ───────────────────────────────────────────────────
// 25×25 covers half the 50×50 map in each dimension — sufficient local context.

pub const OBS_CROP_SIZE: usize = 25;

pub const OBS_DIM:   usize                 = OBS_CHANNELS * OBS_CROP_SIZE * OBS_CROP_SIZE;
pub const OBS_SHAPE: (usize, usize, usize) = (OBS_CHANNELS, OBS_CROP_SIZE, OBS_CROP_SIZE);

// ── Minimap ───────────────────────────────────────────────────────────────────
// 17×17 → each cell covers ~3×3 tiles on a 50×50 grid, giving cluster-level detail.

pub const MM_CHANNELS: usize = 4;
pub const MM_SIZE:     usize = 17;

pub const MM_CH_OBSTACLE: usize = 0;
pub const MM_CH_GOLD:     usize = 1;
pub const MM_CH_SPEED:    usize = 2;
pub const MM_CH_MULT:     usize = 3;

pub const MM_DIM:   usize                 = MM_CHANNELS * MM_SIZE * MM_SIZE;
pub const MM_SHAPE: (usize, usize, usize) = (MM_CHANNELS, MM_SIZE, MM_SIZE);

// ── Cluster features (appended after minimap) ─────────────────────────────────
// 9 fixed gold regions × (dx_norm, dy_norm, pathdist_norm, count_norm) = 36 floats.
// Matches CLUSTER_K in rl/action.rs and engine/clusters.rs (3×3 region grid).

pub const CLUSTER_K:           usize = 9;
pub const CLUSTER_FEATURE_DIM: usize = 4; // dx, dy, pathdist, count
pub const CLUSTER_FEATURES:    usize = CLUSTER_K * CLUSTER_FEATURE_DIM; // 36 floats

// ── Total flat buffer size ────────────────────────────────────────────────────

/// OBS_DIM (crop) + MM_DIM (minimap) + CLUSTER_FEATURES — written by build_obs_into.
/// 12·625 + 4·289 + 36 = 7500 + 1156 + 36 = 8692.
pub const OBS_TOTAL: usize = OBS_DIM + MM_DIM + CLUSTER_FEATURES;
