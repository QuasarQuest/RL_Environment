// src/item/mod.rs

pub mod pickup;
pub mod plugin;
pub mod spawner;

pub use plugin::ItemPlugin;

use bevy::prelude::*;
use crate::world::coords::GridPos;
use crate::style::color::{GOLD_500, GREEN_400, RED_500, BLUE_500};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ItemKind {
    Gold,
    Health,     // restores 1 heart
    Ammo,       // adds AMMO_PER_PICKUP ammo
    SpeedBoost, // grants SPEED_BUFF_TICKS ticks of double movement
}

impl ItemKind {
    pub fn color(self) -> Color {
        match self {
            ItemKind::Gold       => GOLD_500,
            ItemKind::Health     => RED_500,
            ItemKind::Ammo       => BLUE_500,
            ItemKind::SpeedBoost => GREEN_400,
        }
    }

    pub fn z_layer(self) -> f32 { 0.5 }
}

// ── Item ECS components ───────────────────────────────────────────────────────

#[derive(Component, Clone, Copy, Debug)]
pub struct Item {
    pub kind: ItemKind,
}

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