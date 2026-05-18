// src/item/spawner.rs

use bevy::prelude::*;
use crate::sim::config::SimConfig;
use crate::world::coords::GridPos;
use crate::world::config::WorldConfig;
use crate::world::tile::Tile;
use crate::world::Grid;
use super::{Item, ItemKind};

#[cfg(not(feature = "headless"))]
use crate::config;
#[cfg(not(feature = "headless"))]
use crate::viz::grid_offset::GridOffset;
#[cfg(not(feature = "headless"))]
use super::ItemBundle;

// ── Per-item decay ────────────────────────────────────────────────────────────

#[derive(Component, Clone, Copy)]
pub struct DespawnAt(pub u64);

// ── Free-tile pool ────────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct FreeTilePool {
    tiles: Vec<GridPos>,
}

impl FreeTilePool {
    pub fn build(grid: &Grid) -> Self {
        let tiles = grid.iter()
            .filter(|(_, _, tile)| *tile == Tile::Free)
            .map(|(x, y, _)| GridPos::new(x as i32, y as i32))
            .collect();
        Self { tiles }
    }

    pub fn random(&self) -> Option<GridPos> {
        if self.tiles.is_empty() { return None; }
        Some(self.tiles[rand::random_range(0..self.tiles.len())])
    }
}

// ── Config types ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ItemSpawnConfig {
    pub kind:       ItemKind,
    pub max_on_map: usize,
}

#[derive(Clone, Debug)]
pub struct BandConfig {
    pub kind:           ItemKind,
    pub min_on_map:     usize,
    pub max_on_map:     usize,
    pub decay_ticks:    u64,
    pub kp:             f32,
    pub ki:             f32,
    pub ki_clamp:       f32,
    integral:           f32,
    last_sample_tick:   u64,
    sample_interval:    u64,
    last_spawn_tick:    u64,
    effective_cooldown: u64,
    effective_batch:    usize,
}

impl BandConfig {
    pub fn new(kind: ItemKind, min_on_map: usize, max_on_map: usize, decay_ticks: u64) -> Self {
        Self {
            kind, min_on_map, max_on_map, decay_ticks,
            kp: 0.5, ki: 0.02, ki_clamp: 10.0,
            integral: 0.0, last_sample_tick: 0,
            sample_interval: 30, last_spawn_tick: 0,
            effective_cooldown: 20,
            effective_batch: (max_on_map / 4).max(1),
        }
    }

    fn target(&self) -> f32 { ((self.min_on_map + self.max_on_map) / 2) as f32 }

    fn update_pi(&mut self, count: usize) {
        let error     = self.target() - count as f32;
        self.integral = (self.integral + error).clamp(-self.ki_clamp, self.ki_clamp);
        let output    = self.kp * error + self.ki * self.integral;
        let max_out   = self.kp * self.max_on_map as f32 + self.ki * self.ki_clamp;
        let t         = (output / max_out).clamp(-1.0, 1.0);
        self.effective_cooldown =
            (20.0 * (1.0 - 0.75 * t)).clamp(4.0, 60.0) as u64;
        let nom   = (self.max_on_map / 4).max(1) as f32;
        let max_b = (self.max_on_map / 2).max(1) as f32;
        self.effective_batch =
            (nom + t * (max_b - nom)).clamp(1.0, max_b) as usize;
    }
}

// ── Spawner resource ──────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct ItemSpawner {
    pub bands: Vec<BandConfig>,
}

impl ItemSpawner {
    pub fn from_map_config(map: &WorldConfig) -> Self {
        let bands = map.item_spawners.iter()
            .filter_map(|ron| ron.to_config())
            .map(|cfg| BandConfig::new(
                cfg.kind,
                (cfg.max_on_map / 2).max(1),
                cfg.max_on_map,
                300,
            ))
            .collect();
        Self { bands }
    }
}

// ── Systems ───────────────────────────────────────────────────────────────────

pub fn build_free_tile_pool(mut commands: Commands, grid: Res<Grid>) {
    commands.insert_resource(FreeTilePool::build(&grid));
}

/// Periodically spawns items based on PI controller target.
/// In headless mode items are not spawned — the RL agent's obs vector
/// tracks item positions, but item spawning requires GridOffset (viz-only).
/// For RL training, initial items from config.ron are sufficient.
#[cfg(not(feature = "headless"))]
pub fn spawn_items_periodically(
    mut commands: Commands,
    sim:          Res<SimConfig>,
    mut spawner:  ResMut<ItemSpawner>,
    pool:         Res<FreeTilePool>,
    offset:       Res<GridOffset>,
    existing:     Query<&Item>,
) {
    if sim.game_over { return; }

    for band in spawner.bands.iter_mut() {
        let count = existing.iter().filter(|i| i.kind == band.kind).count();

        if sim.tick >= band.last_sample_tick + band.sample_interval {
            band.update_pi(count);
            band.last_sample_tick = sim.tick;
        }

        if count >= band.max_on_map { continue; }
        if sim.tick < band.last_spawn_tick + band.effective_cooldown { continue; }
        if count as f32 >= band.target() { continue; }

        let to_spawn = band.effective_batch.min(band.max_on_map - count);
        for _ in 0..to_spawn {
            spawn_item(&mut commands, band.kind, &pool, &offset, sim.tick, band.decay_ticks);
        }
        band.last_spawn_tick = sim.tick;
    }
}

pub fn decay_items(
    mut commands: Commands,
    sim:          Res<SimConfig>,
    query:        Query<(Entity, &DespawnAt)>,
) {
    for (entity, at) in query.iter() {
        if sim.tick >= at.0 { commands.entity(entity).despawn(); }
    }
}

#[cfg(not(feature = "headless"))]
fn spawn_item(
    commands:     &mut Commands,
    kind:         ItemKind,
    pool:         &FreeTilePool,
    offset:       &GridOffset,
    current_tick: u64,
    decay_ticks:  u64,
) {
    let Some(pos) = pool.random() else { return };
    let world_pos = offset.world_pos(pos.x, pos.y);
    let mut e = commands.spawn(ItemBundle::new(kind, pos, config::TILE_SIZE, world_pos));
    if decay_ticks > 0 { e.insert(DespawnAt(current_tick + decay_ticks)); }
}