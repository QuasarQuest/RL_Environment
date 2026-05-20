// src/world/plugin.rs
//
// WorldPlugin — thin orchestrator.
//
// This file contains NO layout math. All coordinate computation lives in
// world/layout.rs. This file only:
//   1. Loads WorldConfig from disk
//   2. Builds the Grid resource
//   3. Calls layout functions to place tiles, obstacles, items
//   4. Exposes reset functions for the headless RL restart path

use bevy::prelude::*;
use super::config::WorldConfig;
use super::grid::Grid;
use super::layout::{
    self, ResolvedLayout, place_obstacles, item_configs_for_free_count,
};
use super::tile::Tile;
use crate::item::Item;
use crate::config as global;
use crate::world::coords::GridPos;

// ── Config path resource ──────────────────────────────────────────────────────

/// Injected before `WorldPlugin` runs — tells `load_map` which file to load.
/// Defaults to the standard project-relative path.
#[derive(Resource)]
pub struct ConfigPath(pub String);

impl Default for ConfigPath {
    fn default() -> Self {
        Self("assets/world/config.ron".to_owned())
    }
}

// ── Startup systems ───────────────────────────────────────────────────────────

fn load_map(path: Res<ConfigPath>, mut commands: Commands) {
    let cfg = WorldConfig::load(&path.0);
    commands.insert_resource(Grid::new(cfg.width, cfg.height));
    commands.insert_resource(cfg);
}

pub fn spawn_world(
    mut commands: Commands,
    cfg:          Res<WorldConfig>,
    mut grid:     ResMut<Grid>,
) {
    let mut layout = build_world_layout(&cfg, &mut grid);

    // Compute item configs now that free tile count is known, store on layout
    // so ItemPlugin::init_item_spawner can read them without recomputation.
    let free_count = layout::count_free_tiles(&grid_to_vec(&grid));
    layout.item_configs = item_configs_for_free_count(&cfg, free_count);

    spawn_initial_items(&mut commands, &cfg, &grid, &layout);

    // Store layout as a Bevy resource — agent spawner, display system,
    // RL restart path all read from it.
    commands.insert_resource(layout);
}

// ── Public API — called by headless restart ───────────────────────────────────

/// Rebuild the grid from scratch: fixed tiles, safe zones, obstacles.
/// Returns the resolved layout so the caller can respawn agents/items.
///
/// This is the only function that should be called on episode reset —
/// it is guaranteed to leave the grid in a consistent state.
pub fn rebuild_grid(cfg: &WorldConfig, grid: &mut Grid) -> ResolvedLayout {
    build_world_layout(cfg, grid)
}


// ── Internal helpers ──────────────────────────────────────────────────────────

/// Core layout builder: clears the grid, places bases + safe zones, then
/// places procedural obstacles.  Returns the resolved layout.
fn build_world_layout(cfg: &WorldConfig, grid: &mut Grid) -> ResolvedLayout {
    // 1. Clear everything to Free
    clear_grid(grid);

    // 2. Resolve bases and agent positions (no randomness needed here)
    let layout = layout::resolve(cfg);

    // 3. Place base tiles and surrounding safe zones
    let safe_radius = cfg.safe_zone_radius();
    for base in &layout.bases {
        // Safe zone first so the base tile overwrites the centre
        stamp_safe_zone(grid, base.x, base.y, base.team, safe_radius);
        grid.set(base.x, base.y, Tile::Base(base.team));
    }

    // 4. Place procedural obstacles into a flat tile buffer, then write back
    //    (layout::place_obstacles works on a Vec<Tile> for borrow reasons)
    let mut tiles = grid_to_vec(grid);
    place_obstacles(&mut tiles, cfg, &layout);
    vec_to_grid(grid, &tiles);

    layout
}

/// Spawn items proportional to free tile count after obstacle placement.
fn spawn_initial_items(
    commands: &mut Commands,
    cfg:      &WorldConfig,
    grid:     &Grid,
    layout:   &ResolvedLayout,  // kept for future: avoid spawning on spawn points
) {
    let _ = layout; // currently unused, reserved for spawn-point exclusion
    let free_count = layout::count_free_tiles(&grid_to_vec(grid));
    let item_cfgs  = item_configs_for_free_count(cfg, free_count);

    for item_cfg in &item_cfgs {
        // Spawn `initial` items = half of max_on_map (see ItemDensityConfig)
        let initial = (item_cfg.max_on_map / 2).max(1);
        let mut placed   = 0usize;
        let mut attempts = 0usize;

        while placed < initial && attempts < initial * 200 {
            attempts += 1;
            let x = rand::random_range(0..cfg.width  as i32);
            let y = rand::random_range(0..cfg.height as i32);
            if grid.get(x, y) == Some(Tile::Free) {
                let pos = GridPos::new(x, y);
                commands.spawn((
                    Item { kind: item_cfg.kind },
                    pos,
                    Sprite {
                        color:       item_cfg.kind.color(),
                        custom_size: Some(Vec2::splat(global::TILE_SIZE * 0.6)),
                        ..default()
                    },
                    Transform::from_xyz(0.0, 0.0, item_cfg.kind.z_layer()),
                    Visibility::default(),
                ));
                placed += 1;
            }
        }
    }

    // Rebuild the free tile pool resource so the item spawner stays accurate.
    // (FreeTilePool is rebuilt by the RL restart path too — keep in sync.)
}

fn stamp_safe_zone(grid: &mut Grid, bx: i32, by: i32, team: u8, radius: i32) {
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx == 0 && dy == 0 { continue; } // base tile stamped separately
            let cx = bx + dx;
            let cy = by + dy;
            if grid.in_bounds(cx, cy) {
                // Only overwrite Free — don't clobber another team's base/safe zone
                if grid.get(cx, cy) == Some(Tile::Free) {
                    grid.set(cx, cy, Tile::SafeZone(team));
                }
            }
        }
    }
}

fn clear_grid(grid: &mut Grid) {
    for y in 0..grid.height as i32 {
        for x in 0..grid.width as i32 {
            grid.set(x, y, Tile::Free);
        }
    }
}

// ── Grid ↔ Vec<Tile> helpers ─────────────────────────────────────────────────
// layout::place_obstacles mutates a Vec<Tile> (avoids borrowing Grid twice).
// These two functions bridge the gap cheaply.

fn grid_to_vec(grid: &Grid) -> Vec<Tile> {
    (0..grid.height).flat_map(|y| {
        (0..grid.width).map(move |x| {
            grid.get(x as i32, y as i32).unwrap_or(Tile::Free)
        })
    }).collect()
}

fn vec_to_grid(grid: &mut Grid, tiles: &[Tile]) {
    for y in 0..grid.height {
        for x in 0..grid.width {
            grid.set(x as i32, y as i32, tiles[y * grid.width + x]);
        }
    }
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        // Insert default config path if the caller hasn't set one.
        if !app.world().contains_resource::<ConfigPath>() {
            app.init_resource::<ConfigPath>();
        }
        app
            .add_systems(PreStartup, load_map)
            .add_systems(Startup,    spawn_world);
    }
}