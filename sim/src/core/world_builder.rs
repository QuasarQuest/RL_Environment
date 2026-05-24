// src/sim_core/world_builder.rs
//
// Constructs a fresh episode: grid, agents, items.
// Called on new() and (item placement only) on reset().
//
// Item placement uses shuffle+take instead of rejection sampling:
//   Old: up to target*200 random attempts per item kind — O(n²) worst case.
//   New: collect all eligible tiles once, shuffle, take first N — O(n).
// This is faster and always places exactly `target` items when enough free
// tiles exist, with no silent under-placement from exhausted retry budgets.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::{SmallRng, SysRng};
use rustc_hash::FxHashSet;

use crate::config;
use crate::world::{
    config::WorldConfig,
    coords::GridPos,
    grid::Grid,
    layout::{self, item_configs_for_free_count, place_obstacles},
    tile::Tile,
};
use super::state::{AgentState, ItemState};

pub struct WorldSnapshot {
    pub grid:   Grid,
    pub agents: Vec<AgentState>,
    pub items:  Vec<ItemState>,
}

/// Full build: used once on SimCore::new().
pub fn build(cfg: &WorldConfig) -> WorldSnapshot {
    let layout   = layout::resolve(cfg);
    let mut grid = Grid::new(cfg.width, cfg.height);
    let safe_r   = cfg.safe_zone_radius();

    // Safe zones then base tiles (base overwrites safe zone centre).
    for base in &layout.bases {
        for dy in -safe_r..=safe_r {
            for dx in -safe_r..=safe_r {
                if dx == 0 && dy == 0 { continue; }
                let (x, y) = (base.x + dx, base.y + dy);
                if grid.in_bounds(x, y) && grid.get(x, y) == Some(Tile::Free) {
                    grid.set(x, y, Tile::SafeZone(base.team));
                }
            }
        }
        grid.set(base.x, base.y, Tile::Base(base.team));
    }

    // Obstacle placement via a flat tile vec (avoids double-borrow of Grid).
    let mut tiles: Vec<Tile> = {
        let g = &grid;
        (0..cfg.height)
            .flat_map(|y| (0..cfg.width).map(move |x| g.get(x as i32, y as i32).unwrap_or(Tile::Free)))
            .collect()
    };
    place_obstacles(&mut tiles, cfg, &layout);
    for y in 0..cfg.height {
        for x in 0..cfg.width {
            grid.set(x as i32, y as i32, tiles[y * cfg.width + x]);
        }
    }

    // Agents sorted by team so agents[0] is always team 0 (the RL agent).
    // base_pos is set to the Base tile for this agent's team by scanning the
    // layout bases — guaranteed to exist after grid construction above.
    let mut agents: Vec<AgentState> = layout.agents.iter().map(|ra| {
        let base = layout.bases.iter()
            .find(|b| b.team == ra.team)
            .map(|b| GridPos::new(b.x, b.y))
            .unwrap_or_else(|| GridPos::new(ra.x, ra.y));
        AgentState {
            pos:          GridPos::new(ra.x, ra.y),
            team:         ra.team,
            gold_carried: 0,
            score:        0,
            hearts:       config::AGENT_MAX_HEARTS,
            ammo:         config::AGENT_START_AMMO,
            speed_buff:   0,
            spawn_pos:    GridPos::new(ra.x, ra.y),
            base_pos:     base,
        }
    }).collect();
    agents.sort_by_key(|a| a.team);

    let mut rng = SmallRng::try_from_rng(&mut SysRng).expect("SysRng failed");
    let items = spawn_items_with_rng(cfg, &grid, &agents.iter().map(|a| a.pos).collect::<Vec<_>>(), &mut rng);
    WorldSnapshot { grid, agents, items }
}

/// Item placement — shuffle+take instead of rejection sampling.
///
/// Collects all eligible free tiles once into a Vec, shuffles it, then takes
/// the first `target` entries. This is O(free_tiles) instead of O(target²)
/// and always succeeds when enough tiles exist — no silent under-placement.
pub fn spawn_items(cfg: &WorldConfig, grid: &Grid, spawn_positions: &[GridPos], rng: &mut SmallRng) -> Vec<ItemState> {
    spawn_items_with_rng(cfg, grid, spawn_positions, rng)
}

fn spawn_items_with_rng(cfg: &WorldConfig, grid: &Grid, spawn_positions: &[GridPos], rng: &mut SmallRng) -> Vec<ItemState> {
    let blocked: FxHashSet<GridPos> = spawn_positions.iter().copied().collect();

    // Collect all eligible positions once — O(width × height).
    let mut free_tiles: Vec<GridPos> = (0..cfg.height as i32)
        .flat_map(|y| (0..cfg.width as i32).map(move |x| GridPos::new(x, y)))
        .filter(|p| grid.get(p.x, p.y) == Some(Tile::Free) && !blocked.contains(p))
        .collect();

    let free_count = free_tiles.len();
    let item_cfgs = item_configs_for_free_count(cfg, free_count);
    let mut items = Vec::new();

    for ic in &item_cfgs {
        let target = (ic.max_on_map / 2).max(1);
        // Shuffle the whole pool and take the first `target` positions.
        // Each item kind gets an independent shuffle so placements don't
        // cluster at the same tiles across kinds.
        free_tiles.shuffle(rng);
        for &pos in free_tiles.iter().take(target) {
            items.push(ItemState { pos, kind: ic.kind });
        }
    }

    items
}