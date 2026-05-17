// src/item/plugin.rs

use bevy::prelude::*;
use crate::sim::schedule::OnSimTick;
use crate::world::config::MapConfig;
use super::spawner::{ItemSpawner, build_free_tile_pool, spawn_items_periodically, decay_items};
use super::pickup::{pickup_items, deposit_gold, despawn_claimed, sync_item_transforms};

fn init_item_spawner(mut commands: Commands, map: Res<MapConfig>) {
    commands.insert_resource(ItemSpawner::from_map_config(&map));
}

pub struct ItemPlugin;

impl Plugin for ItemPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_systems(Startup, init_item_spawner)
            .add_systems(Startup, build_free_tile_pool.after(init_item_spawner))
            .add_systems(OnSimTick, (
                spawn_items_periodically,
                decay_items,
                pickup_items,
                deposit_gold,
                despawn_claimed,
            ).chain())
            .add_systems(Update, sync_item_transforms);
    }
}