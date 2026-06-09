use crate::world::coords::GridPos;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dir { N, S, E, W, NE, NW, SE, SW }

impl Dir {
    pub fn all() -> &'static [Dir; 8] {
        &[Dir::N, Dir::S, Dir::E, Dir::W, Dir::NE, Dir::NW, Dir::SE, Dir::SW]
    }

    pub fn delta(self) -> (i32, i32) {
        match self {
            Dir::N  => ( 0,  1), Dir::S  => ( 0, -1),
            Dir::E  => ( 1,  0), Dir::W  => (-1,  0),
            Dir::NE => ( 1,  1), Dir::NW => (-1,  1),
            Dir::SE => ( 1, -1), Dir::SW => (-1, -1),
        }
    }

    pub fn is_diagonal(self) -> bool {
        matches!(self, Dir::NE | Dir::NW | Dir::SE | Dir::SW)
    }
}

/// Internal movement action used by both the RL agent (via A* goal navigation)
/// Internal movement action used by the RL agent via A* goal navigation.
/// Single-agent gold rush — no combat actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Move one cell in the given direction.
    Move(Dir),
    /// Do nothing this tick.
    Wait,
}

/// Single-agent gold-rush state: gold carry/score plus the temporary buff timers
/// and the held multiplier charge. Combat is gone — there are no enemies, no teams
/// and no damage. `team`/`hearts`/`ammo` are retained as inert single-agent values
/// (team 0, full hearts, no ammo) only so the existing viewer panels keep
/// rendering; the simulation never mutates them.
#[derive(Clone)]
pub struct AgentState {
    pub pos:          GridPos,
    pub gold_carried: u8,
    pub score:        u32,
    /// Ticks of the speed buff remaining (0 = none). While >0 the agent moves at the
    /// full per-tick cadence (one tile every tick) instead of the base half-cadence.
    /// Set by the Speed pickup.
    pub speed_buff:   u16,
    /// Movement-cadence accumulator. Each tick gains MOVE_ENERGY_BASE (or
    /// MOVE_ENERGY_SPEED while speed-buffed); the agent steps one tile and subtracts
    /// MOVE_ENERGY_STEP whenever it reaches that threshold. Caps movement at one
    /// tile/tick so a speed buff is "twice as often", never "twice as far".
    pub move_energy:  u8,
    /// Held score-multiplier charges (0 or MULT_CHARGE_MAX). Set by the Multiplier
    /// pickup, consumed one-per-deposit to double that deposit's value.
    pub mult_charge:  u8,
    pub spawn_pos:    GridPos,
    pub base_pos:     GridPos,

    // ── Inert viewer-compat fields (never mutated by the sim) ──────────────────
    pub team:   u8,
    pub hearts: u8,
    pub ammo:   u8,
}
