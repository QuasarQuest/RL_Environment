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
// the window lasts (rarer tier → longer). SPEED_BUFF_MAX is the normaliser for the
// speed obs channel. (The old Slow hazard was removed.)
pub const SPEED1_TICKS: u16 =  40;
pub const SPEED2_TICKS: u16 =  80;
pub const SPEED3_TICKS: u16 = 160;
/// Trap immobilises the agent completely (0 tiles/tick) for this many ticks. The
/// navigator routes around trap tiles (engine/nav.rs), so this only bites when a
/// trap is unavoidable or spawns under the agent.
pub const TRAP_TICKS:   u16 = 250;

/// Normalisers for the buff-remaining observation channels (longest window per buff).
pub const SPEED_BUFF_MAX: u16 = SPEED3_TICKS;
pub const TRAP_BUFF_MAX:  u16 = TRAP_TICKS;

/// The score multiplier is a consumable charge (not a timed window): picking one up
/// holds a single charge; the next deposit consumes it and is worth this many times
/// its gold. Capacity is intentionally 1 — "use it on your next bank".
pub const DEPOSIT_MULTIPLIER: u32 = 2;
pub const MULT_CHARGE_MAX:    u8  = 1;

/// Traps spawn as connected blobs of this many adjacent tiles (instead of single
/// tiles), so a cluster can seal a corridor and force the navigator to detour.
pub const TRAP_CLUSTER_SIZE: usize = 3;

// ── Viewer-compat constant ──────────────────────────────────────────────────────
// Combat is gone, but the agent's inert `hearts` field (read only by the viewer's
// GOAP fallback) is initialised "full" so it never reads as low-health.
pub const AGENT_MAX_HEARTS: u8 = 3;