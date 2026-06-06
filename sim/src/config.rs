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

// ── Item buffs (ticks) ─────────────────────────────────────────────────────────
// Speed boosts: all give 2× move speed; the three tiers differ only in how long
// the window lasts (rarer tier → longer). Slow halves move speed; Multiplier
// doubles deposit value. SPEED_BUFF_MAX is the normaliser for the speed obs channel.
pub const SPEED1_TICKS: u16 =  40;
pub const SPEED2_TICKS: u16 =  80;
pub const SPEED3_TICKS: u16 = 160;
pub const SLOW_TICKS:   u16 = 200;  // half move-speed window — long enough to clearly bite
pub const MULT_TICKS:   u16 = 100;
/// Trap immobilises the agent completely (0 tiles/tick) for this many ticks — a
/// hazard to avoid, far more punishing than Slow.
pub const TRAP_TICKS:   u16 = 250;

/// Normalisers for the buff-remaining observation channels (longest window per buff).
pub const SPEED_BUFF_MAX: u16 = SPEED3_TICKS;
pub const SLOW_BUFF_MAX:  u16 = SLOW_TICKS;
pub const MULT_BUFF_MAX:  u16 = MULT_TICKS;
pub const TRAP_BUFF_MAX:  u16 = TRAP_TICKS;

/// Score multiplier applied to a deposit while `mult_buff > 0`.
pub const DEPOSIT_MULTIPLIER: u32 = 2;

// ── Viewer-compat constant ──────────────────────────────────────────────────────
// Combat is gone, but the agent's inert `hearts` field (read only by the viewer's
// GOAP fallback) is initialised "full" so it never reads as low-health.
pub const AGENT_MAX_HEARTS: u8 = 3;