// src/world/tile.rs
// Tile is terrain only. Items are ECS entities — Tile::Gold removed.

use bevy::prelude::Color;
use crate::style::color::team_color;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tile {
    Free,
    Obstacle,
    /// Base(team_id) — only the owning team can drop gold here.
    Base(u8),
}

impl Tile {
    pub fn color(self) -> Color {
        match self {
            Tile::Free     => Color::srgb(0.12, 0.12, 0.12),
            Tile::Obstacle => Color::srgb(0.35, 0.35, 0.35),
            Tile::Base(t)  => team_color(t),
        }
    }

    pub fn is_walkable(self) -> bool {
        !matches!(self, Tile::Obstacle)
    }
}