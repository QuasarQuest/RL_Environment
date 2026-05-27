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
/// and scripted enemies. Combat is handled separately via physics::try_melee/ranged_attack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Move one cell in the given direction.
    Move(Dir),
    /// Do nothing this tick.
    Wait,
}

#[derive(Clone)]
pub struct AgentState {
    pub pos:          GridPos,
    pub team:         u8,
    pub gold_carried: u8,
    pub score:        u32,
    pub hearts:       u8,
    pub ammo:         u8,
    pub speed_buff:   u8,
    pub spawn_pos:    GridPos,
    pub base_pos:     GridPos,
}
