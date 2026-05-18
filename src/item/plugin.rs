// src/item/plugin.rs

use bevy::prelude::*;
use crate::sim::schedule::OnSimTick;
use super::spawner::{build_free_tile_pool, decay_items};
use super::pickup::{pickup_items, deposit_gold, despawn_claimed};

#[cfg(not(feature = "headless"))]
use crate::world::config::WorldConfig;
#[cfg(not(feature = "headless"))]
use super::spawner::{ItemSpawner, spawn_items_periodically};
#[cfg(not(feature = "headless"))]
use super::pickup::sync_item_transforms;

#[cfg(not(feature = "headless"))]
fn init_item_spawner(mut commands: Commands, map: Res<WorldConfig>) {
    commands.insert_resource(ItemSpawner::from_map_config(&map));
}

pub struct ItemPlugin;

impl Plugin for ItemPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(not(feature = "headless"))]
        app.add_systems(Startup, init_item_spawner);

        app
            .add_systems(Startup, build_free_tile_pool)
            .add_systems(OnSimTick, (
                decay_items,
                pickup_items,
                deposit_gold,
                despawn_claimed,
            ).chain());

        #[cfg(not(feature = "headless"))]
        app.add_systems(OnSimTick,
                        spawn_items_periodically.before(decay_items),
        );

        #[cfg(not(feature = "headless"))]
        app.add_systems(Update, sync_item_transforms);
    }
}