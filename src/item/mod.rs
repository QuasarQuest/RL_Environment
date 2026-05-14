// src/item/mod.rs

pub mod pickup;
pub mod plugin;
pub mod spawner;

pub use plugin::ItemPlugin;

use bevy::prelude::*;
use crate::world::coords::GridPos;
use crate::style::color::GOLD_500;

// ── ItemKind ──────────────────────────────────────────────────────────────────
// Add variants here as the game grows. Nothing else needs to change.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ItemKind {
    Gold,
    // Health,
    // SpeedBoost,
    // Trap,
    // Ammo,
}

impl ItemKind {
    pub fn color(self) -> Color {
        match self {
            ItemKind::Gold => GOLD_500,
        }
    }

    /// Display label for HUD/tooltip.
    pub fn label(self) -> &'static str {
        match self {
            ItemKind::Gold => "Gold",
        }
    }

    /// z-layer for rendering — above tiles (0.0), below agents (1.0).
    pub fn z_layer(self) -> f32 { 0.5 }
}

// ── Item ECS components ───────────────────────────────────────────────────────

/// Marker component on item entities.
#[derive(Component, Clone, Copy, Debug)]
pub struct Item {
    pub kind: ItemKind,
}

/// Bundle spawned for each item entity.
#[derive(Bundle)]
pub struct ItemBundle {
    pub item:       Item,
    pub pos:        GridPos,
    pub sprite:     Sprite,
    pub transform:  Transform,
    pub visibility: Visibility,
}

impl ItemBundle {
    pub fn new(kind: ItemKind, pos: GridPos, tile_size: f32, world_pos: Vec2) -> Self {
        Self {
            item:      Item { kind },
            pos,
            sprite:    Sprite {
                color:       kind.color(),
                custom_size: Some(Vec2::splat(tile_size * 0.6)),
                ..default()
            },
            transform:  Transform::from_xyz(world_pos.x, world_pos.y, kind.z_layer()),
            visibility: Visibility::default(),
        }
    }
}