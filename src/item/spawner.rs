// src/item/spawner.rs

use bevy::prelude::*;
use crate::config;
use crate::sim::config::SimConfig;
use crate::world::coords::GridPos;
use crate::world::map_config::MapConfig;
use crate::world::tile::Tile;
use crate::world::Grid;
use crate::viz::grid_offset::GridOffset;
use super::{Item, ItemBundle, ItemKind};

#[derive(Clone, Debug)]
pub struct ItemSpawnConfig {
    pub kind:           ItemKind,
    pub interval_ticks: u32,
    pub max_on_map:     usize,
}

#[derive(Resource)]
pub struct ItemSpawner {
    pub configs: Vec<ItemSpawnConfig>,
}

impl ItemSpawner {
    pub fn from_map_config(map: &MapConfig) -> Self {
        let configs = map.item_spawners.iter()
            .filter_map(|ron| ron.to_config())
            .collect();
        Self { configs }
    }
}

pub fn spawn_items_periodically(
    mut commands: Commands,
    sim:          Res<SimConfig>,
    spawner:      Res<ItemSpawner>,
    grid:         Res<Grid>,
    offset:       Res<GridOffset>,
    existing:     Query<&Item>,
) {
    if sim.game_over { return; }

    for cfg in &spawner.configs {
        if cfg.interval_ticks == 0 { continue; }
        if sim.tick % cfg.interval_ticks as u64 != 0 { continue; }

        let count = existing.iter().filter(|i| i.kind == cfg.kind).count();
        if count >= cfg.max_on_map { continue; }

        try_spawn_item(&mut commands, cfg.kind, &grid, &offset);
    }
}

pub fn try_spawn_item(
    commands: &mut Commands,
    kind:     ItemKind,
    grid:     &Grid,
    offset:   &GridOffset,
) {
    for _ in 0..200 {
        let x = rand::random_range(0..grid.width  as i32);
        let y = rand::random_range(0..grid.height as i32);
        if grid.get(x, y) == Some(Tile::Free) {
            let pos       = GridPos::new(x, y);
            let world_pos = offset.world_pos(x, y);
            commands.spawn(ItemBundle::new(kind, pos, config::TILE_SIZE, world_pos));
            return;
        }
    }
}