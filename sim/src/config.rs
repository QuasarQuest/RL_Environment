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
// Speed boost: a single speed item. While the buff is active the agent moves at the
// full per-tick cadence (a step every tick) instead of the base half-cadence (a step
// every other tick) — a 2× speed-up that never teleports two tiles (see the movement
// cadence constants below). SPEED_BUFF_MAX is the normaliser for the speed obs
// channel. (The old Slow hazard was removed.)
pub const SPEED_TICKS: u16 = 160;

// ── Movement cadence (energy system) ────────────────────────────────────────────
// The agent never moves more than one tile per tick. Each tick it accumulates move
// energy and steps one tile the moment that energy reaches MOVE_ENERGY_STEP, then
// subtracts the threshold. Base gains MOVE_ENERGY_BASE/tick → a step every other
// tick; an active speed buff gains MOVE_ENERGY_SPEED/tick → a step every tick. Same
// 2× advantage as the old two-tile jump, but capped at one tile/tick so a speed move
// can never overshoot a turn or sail onto a trap the path routed around.
pub const MOVE_ENERGY_STEP:  u8 = 2;
pub const MOVE_ENERGY_BASE:  u8 = 1;
pub const MOVE_ENERGY_SPEED: u8 = 2;

/// Trap immobilises the agent completely (0 tiles/tick) for this many ticks. The
/// navigator routes around trap tiles (engine/nav.rs), so this only bites when a
/// trap is unavoidable or spawns under the agent.
pub const TRAP_TICKS:   u16 = 250;

/// Normalisers for the buff-remaining observation channels (longest window per buff).
pub const SPEED_BUFF_MAX: u16 = SPEED_TICKS;
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