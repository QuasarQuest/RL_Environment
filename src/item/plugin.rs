// src/item/plugin.rs

use bevy::prelude::*;
use crate::sim::schedule::OnSimTick;
use crate::world::map_config::MapConfig;
use super::spawner::{ItemSpawner, spawn_items_periodically};
use super::pickup::{pickup_items, despawn_claimed, sync_item_transforms};

/// Startup system — runs after WorldPlugin::load_map has inserted MapConfig.
fn init_item_spawner(
    mut commands: Commands,
    map:          Res<MapConfig>,
) {
    commands.insert_resource(ItemSpawner::from_map_config(&map));
}

pub struct ItemPlugin;

impl Plugin for ItemPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_item_spawner)
            .add_systems(OnSimTick, (
                spawn_items_periodically,
                pickup_items,
                despawn_claimed,
            ).chain())
            .add_systems(Update, sync_item_transforms);
    }
}