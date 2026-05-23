// src/world/tile.rs

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tile {
    Free,
    Obstacle,
    /// Base(team_id) — centre tile where the owning team deposits gold.
    Base(u8),
    /// SafeZone(team_id) — area around the base; walkable, no obstacles.
    SafeZone(u8),
}

impl Tile {
    pub fn is_walkable(self) -> bool {
        !matches!(self, Tile::Obstacle)
    }

    pub fn team_id(self) -> Option<u8> {
        match self {
            Tile::Base(t) | Tile::SafeZone(t) => Some(t),
            _ => None,
        }
    }
}
