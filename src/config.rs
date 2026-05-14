// src/config.rs

// ── Window ───────────────────────────────────────────────────────────────────
pub const WINDOW_TITLE:  &str = "Algorithm Test Environment";
pub const WINDOW_WIDTH:  u32  = 1000;
pub const WINDOW_HEIGHT: u32  = 1000;

// ── Grid ─────────────────────────────────────────────────────────────────────
pub const GRID_W: usize = 50;
pub const GRID_H: usize = 50;

// ── Rendering ────────────────────────────────────────────────────────────────
pub const TILE_SIZE: f32 = 16.0;
pub const TILE_GAP:  f32 =  1.0;

// ── Simulation ───────────────────────────────────────────────────────────────
pub const DEFAULT_TICKS_PER_SECOND: f32 = 10.0;
pub const MIN_TICKS_PER_SECOND:     f32 =  1.0;
pub const MAX_TICKS_PER_SECOND:     f32 = 60.0;
pub const SPEED_STEP:               f32 =  5.0;

// ── Match ─────────────────────────────────────────────────────────────────────
pub const MATCH_DURATION_SECS: f32 = 90.0;  // 1:30

// ── Agent ────────────────────────────────────────────────────────────────────
pub const AGENT_START_HEALTH: u32 = 100;
pub const AGENT_MAX_GOLD:     u32 =   5;