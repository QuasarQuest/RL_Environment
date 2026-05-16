// src/config.rs

// ── Window ───────────────────────────────────────────────────────────────────
pub const WINDOW_TITLE:  &str = "Algorithm Test Environment";
pub const WINDOW_WIDTH:  u32  = 1000;
pub const WINDOW_HEIGHT: u32  = 1000;

// ── Rendering ────────────────────────────────────────────────────────────────
pub const TILE_SIZE: f32 = 16.0;
pub const TILE_GAP:  f32 =  1.0;

// ── Simulation ───────────────────────────────────────────────────────────────
pub const DEFAULT_TICKS_PER_SECOND: f32 = 10.0;

// ── Agent ────────────────────────────────────────────────────────────────────
pub const AGENT_MAX_GOLD:      u32 = 5;

// ── Health (hearts) ───────────────────────────────────────────────────────────
pub const AGENT_MAX_HEARTS:    u8  = 3;
/// Sim-ticks before a dead agent respawns at base.
pub const AGENT_RESPAWN_TICKS: u8  = 10;

// ── Combat ───────────────────────────────────────────────────────────────────
/// Melee: always available, 1-tile range, no ammo cost.
pub const MELEE_RANGE:         i32 = 1;
/// Ranged: costs 1 ammo, up to RANGED_RANGE tiles (Chebyshev).
pub const RANGED_RANGE:        i32 = 5;

// ── Ammo ─────────────────────────────────────────────────────────────────────
pub const AGENT_START_AMMO:    u8  = 3;
pub const AGENT_MAX_AMMO:      u8  = 10;
/// Ammo added per pickup.
pub const AMMO_PER_PICKUP:     u8  = 3;

// ── Speed boost ───────────────────────────────────────────────────────────────
/// Sim-ticks the speed buff lasts.
pub const SPEED_BUFF_TICKS:    u8  = 15;