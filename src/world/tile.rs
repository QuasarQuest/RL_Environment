// src/world/tile.rs

use bevy::prelude::Color;
use bevy::color::Alpha;
use crate::style::color::team_color;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tile {
    Free,
    Obstacle,
    /// Base(team_id) — centre tile where the owning team deposits gold.
    Base(u8),
    /// SafeZone(team_id) — 7×7 area around the base.
    /// Walkable by anyone. The owning team cannot take damage here.
    /// Obstacles never spawn inside this zone.
    SafeZone(u8),
}

impl Tile {
    pub fn color(self) -> Color {
        match self {
            Tile::Free          => Color::srgb(0.12, 0.12, 0.12),
            Tile::Obstacle      => Color::srgb(0.35, 0.35, 0.35),
            Tile::Base(t)       => team_color(t),
            // Subtle tint of the team color — visible but not distracting
            Tile::SafeZone(t)   => team_color(t).with_alpha(0.15),
        }
    }

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