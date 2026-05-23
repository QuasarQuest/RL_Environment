// src/config.rs

// ── Window ───────────────────────────────────────────────────────────────────
pub const WINDOW_TITLE:  &str = "Algorithm Test Environment";
pub const WINDOW_WIDTH:  u32  = 1000;
pub const WINDOW_HEIGHT: u32  = 1000;

// ── Rendering ────────────────────────────────────────────────────────────────
pub const TILE_SIZE: f32 = 16.0;
pub const TILE_GAP:  f32 =  1.0;

// ── Camera ───────────────────────────────────────────────────────────────────
/// Fraction of extra space around the grid when fitting to window (1.10 = 10% margin).
pub const CAMERA_FIT_MARGIN: f32 = 1.10;
pub const ZOOM_MIN:          f32 = 0.1;
pub const ZOOM_MAX:          f32 = 10.0;
/// Zoom step per scroll tick: fraction of current scale added or removed.
pub const ZOOM_SPEED:        f32 = 0.08;

// ── Simulation ───────────────────────────────────────────────────────────────
pub const DEFAULT_TICKS_PER_SECOND: f32 = 10.0;

// ── Agent ────────────────────────────────────────────────────────────────────
pub const AGENT_MAX_GOLD:      u8  = 5;
pub const AGENT_MAX_HEARTS:    u8  = 3;
pub const AGENT_RESPAWN_TICKS: u8  = 150;
pub const GOLD_CARRY_SPEED:    f32 = 0.9;

// ── Ammo ─────────────────────────────────────────────────────────────────────
pub const AGENT_START_AMMO: u8 = 3;
pub const AGENT_MAX_AMMO:   u8 = 10;
pub const AMMO_PER_PICKUP:  u8 = 3;

// ── Speed boost ───────────────────────────────────────────────────────────────
pub const SPEED_BUFF_TICKS: u8 = 15;

// ── Combat — defaults, all overridable via WorldConfig ───────────────────────
pub const MELEE_RANGE:           u8 = 2;
pub const RANGED_RANGE:          u8 = 6;
pub const KILL_REWARD:           u8 = 15;
pub const MELEE_DAMAGE:          u8 = 1;
pub const RANGED_DAMAGE:         u8 = 1;
pub const MELEE_COOLDOWN_TICKS:  u8 = 4;
pub const RANGED_COOLDOWN_TICKS: u8 = 6;