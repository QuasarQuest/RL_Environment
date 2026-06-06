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

/// Single-agent gold-rush state: gold carry/score plus three temporary buff
/// timers (ticks remaining). Combat is gone — there are no enemies, no teams and
/// no damage. `team`/`hearts`/`ammo` are retained as inert single-agent values
/// (team 0, full hearts, no ammo) only so the existing viewer panels keep
/// rendering; the simulation never mutates them.
#[derive(Clone)]
pub struct AgentState {
    pub pos:          GridPos,
    pub gold_carried: u8,
    pub score:        u32,
    /// Ticks of 2× move speed remaining (0 = none). Set by Speed{1,2,3} pickups.
    pub speed_buff:   u16,
    /// Ticks of 0.5× move speed remaining (0 = none). Set by the Slow hazard.
    pub slow_buff:    u16,
    /// Ticks of 2× deposit value remaining (0 = none). Set by the Multiplier pickup.
    pub mult_buff:    u16,
    /// Ticks of forced immobility remaining (0 = none). Set by stepping on a Trap.
    pub trap_buff:    u16,
    pub spawn_pos:    GridPos,
    pub base_pos:     GridPos,

    // ── Inert viewer-compat fields (never mutated by the sim) ──────────────────
    pub team:   u8,
    pub hearts: u8,
    pub ammo:   u8,
}
