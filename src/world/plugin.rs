// src/world/plugin.rs

use bevy::prelude::*;
use super::grid::Grid;
use super::config::{WorldConfig, ObstacleKind, TileKind};
use super::tile::Tile;
use crate::item::Item;
use crate::config;

fn load_map(mut commands: Commands) {
    let cfg = WorldConfig::load("assets/world/config.ron");
    commands.insert_resource(Grid::new(cfg.width, cfg.height));
    commands.insert_resource(cfg);
}

fn spawn_world(
    mut commands: Commands,
    map:          Res<WorldConfig>,
    mut grid:     ResMut<Grid>,
) {
    apply_fixed_tiles(&map, &mut grid);
    regenerate_obstacles(&map, &mut grid);
    spawn_initial_items(&mut commands, &map, &grid);
}

// ── Public — also called by restart ───────────────────────────────────────────

pub fn apply_fixed_tiles(map: &WorldConfig, grid: &mut Grid) {
    // Reset everything to Free
    for y in 0..grid.height as i32 {
        for x in 0..grid.width as i32 {
            grid.set(x, y, Tile::Free);
        }
    }

    // Fixed tiles (bases)
    for fixed in &map.fixed {
        let tile = match fixed.tile {
            TileKind::Free     => Tile::Free,
            TileKind::Obstacle => Tile::Obstacle,
            TileKind::Base     => Tile::Base(0),
            TileKind::BaseRed  => Tile::Base(0),
            TileKind::BaseBlue => Tile::Base(1),
        };
        grid.set(fixed.x as i32, fixed.y as i32, tile);
    }

    // Stamp SafeZone(team_id) in a (2*radius+1)² area around each base.
    // radius=3 → 7×7 zone. The base centre tile stays as Tile::Base.
    let radius = map.base_safe_radius as i32;
    for fixed in &map.fixed {
        let team_id = match fixed.tile {
            TileKind::BaseRed | TileKind::Base => 0u8,
            TileKind::BaseBlue                 => 1u8,
            _                                  => continue,
        };
        let bx = fixed.x as i32;
        let by = fixed.y as i32;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx == 0 && dy == 0 { continue; } // keep Base tile
                let cx = bx + dx;
                let cy = by + dy;
                if grid.in_bounds(cx, cy) {
                    grid.set(cx, cy, Tile::SafeZone(team_id));
                }
            }
        }
    }
}

pub fn regenerate_obstacles(map: &WorldConfig, grid: &mut Grid) {
    for cluster in &map.obstacle_clusters {
        let (w, h)       = cluster.size;
        let mut placed   = 0;
        let mut attempts = 0;
        while placed < cluster.count && attempts < cluster.count * 200 {
            attempts += 1;
            match cluster.kind {
                ObstacleKind::Block => {
                    let max_x = (map.width  as i32 - w as i32 - 1).max(1);
                    let max_y = (map.height as i32 - h as i32 - 1).max(1);
                    let ox = rand::random_range(1..max_x);
                    let oy = rand::random_range(1..max_y);
                    if footprint_is_free(grid, ox, oy, w as i32, h as i32) {
                        for dy in 0..h as i32 {
                            for dx in 0..w as i32 {
                                grid.set(ox + dx, oy + dy, Tile::Obstacle);
                            }
                        }
                        placed += 1;
                    }
                }
                ObstacleKind::Wall => {
                    let length     = w as i32;
                    let ox         = rand::random_range(1..map.width  as i32 - length - 1);
                    let oy         = rand::random_range(1..map.height as i32 - 2);
                    let horizontal = rand::random_range(0..2) == 0;
                    let (ex, ey)   = if horizontal { (ox + length, oy) } else { (ox, oy + length) };
                    if !grid.in_bounds(ex, ey) { continue; }
                    let clear = if horizontal {
                        (ox..=ox+length).all(|x| grid.get(x, oy) == Some(Tile::Free))
                    } else {
                        (oy..=oy+length).all(|y| grid.get(ox, y) == Some(Tile::Free))
                    };
                    if clear {
                        if horizontal {
                            for x in ox..=ox+length { grid.set(x, oy, Tile::Obstacle); }
                        } else {
                            for y in oy..=oy+length { grid.set(ox, y, Tile::Obstacle); }
                        }
                        placed += 1;
                    }
                }
                ObstacleKind::Scatter => {
                    let x = rand::random_range(1..map.width  as i32 - 1);
                    let y = rand::random_range(1..map.height as i32 - 1);
                    if grid.get(x, y) == Some(Tile::Free) {
                        grid.set(x, y, Tile::Obstacle);
                        placed += 1;
                    }
                }
            }
        }
    }

    // Clear spawn perimeters
    const SPAWN_CLEAR_RADIUS: i32 = 2;
    for agent_cfg in &map.agents {
        for dy in -SPAWN_CLEAR_RADIUS..=SPAWN_CLEAR_RADIUS {
            for dx in -SPAWN_CLEAR_RADIUS..=SPAWN_CLEAR_RADIUS {
                let cx = agent_cfg.x + dx;
                let cy = agent_cfg.y + dy;
                if grid.in_bounds(cx, cy) && grid.get(cx, cy) == Some(Tile::Obstacle) {
                    grid.set(cx, cy, Tile::Free);
                }
            }
        }
    }
}

pub fn spawn_initial_items(commands: &mut Commands, map: &WorldConfig, grid: &Grid) {
    for spawner_cfg in &map.item_spawners {
        if spawner_cfg.initial == 0 { continue; }
        let Some(cfg) = spawner_cfg.to_config() else { continue };
        let mut placed   = 0;
        let mut attempts = 0;
        while placed < spawner_cfg.initial && attempts < spawner_cfg.initial * 100 {
            attempts += 1;
            let x = rand::random_range(0..map.width  as i32);
            let y = rand::random_range(0..map.height as i32);
            // Only Free tiles — not SafeZone or Base
            if grid.get(x, y) == Some(Tile::Free) {
                let pos = crate::world::coords::GridPos::new(x, y);
                commands.spawn((
                    Item { kind: cfg.kind },
                    pos,
                    Sprite {
                        color:       cfg.kind.color(),
                        custom_size: Some(Vec2::splat(config::TILE_SIZE * 0.6)),
                        ..default()
                    },
                    Transform::from_xyz(0.0, 0.0, cfg.kind.z_layer()),
                    Visibility::default(),
                ));
                placed += 1;
            }
        }
    }
}

/// A footprint can only be placed on Tile::Free — never SafeZone or Base.
fn footprint_is_free(grid: &Grid, ox: i32, oy: i32, w: i32, h: i32) -> bool {
    for dy in 0..h {
        for dx in 0..w {
            if grid.get(ox + dx, oy + dy) != Some(Tile::Free) { return false; }
        }
    }
    true
}

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_systems(PreStartup, load_map)
            .add_systems(Startup, spawn_world);
    }
}