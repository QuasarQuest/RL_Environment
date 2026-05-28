// src/engine/builder.rs
//
// Constructs a fresh episode: grid, agents, items.
// Called once on SimCore::new() and (items only) on SimCore::reset().
//
// Item placement uses shuffle+take instead of rejection sampling:
//   O(free_tiles) instead of O(target²), always places exactly `target` items.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::{SmallRng, SysRng};
use rustc_hash::FxHashSet;

use crate::config;
use crate::entity::{AgentState, ItemState};
use crate::world::{
    config::WorldConfig,
    coords::GridPos,
    grid::Grid,
    layout::{self, item_configs_for_free_count, place_obstacles},
    tile::Tile,
};

pub struct WorldSnapshot {
    pub grid:   Grid,
    pub agents: Vec<AgentState>,
    pub items:  Vec<ItemState>,
}

/// Full build — used once on SimCore::new().
pub fn build(cfg: &WorldConfig) -> WorldSnapshot {
    let layout   = layout::resolve(cfg);
    let mut grid = Grid::new(cfg.width, cfg.height);
    let safe_r   = cfg.safe_zone_radius();

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

    // agents[0] is always team 0 (the RL agent).
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
            melee_cooldown:  0,
            ranged_cooldown: 0,
            respawn_timer:   0,
            kills:           0,
        }
    }).collect();
    agents.sort_by_key(|a| a.team);

    let mut rng = SmallRng::try_from_rng(&mut SysRng).expect("SysRng failed");
    let items = spawn_items_internal(cfg, &grid, &agents.iter().map(|a| a.pos).collect::<Vec<_>>(), &mut rng);
    WorldSnapshot { grid, agents, items }
}

/// Item placement — called on every reset with the env's own RNG for reproducibility.
pub fn spawn_items(cfg: &WorldConfig, grid: &Grid, spawn_positions: &[GridPos], rng: &mut SmallRng) -> Vec<ItemState> {
    spawn_items_internal(cfg, grid, spawn_positions, rng)
}

fn spawn_items_internal(cfg: &WorldConfig, grid: &Grid, spawn_positions: &[GridPos], rng: &mut SmallRng) -> Vec<ItemState> {
    let blocked: FxHashSet<GridPos> = spawn_positions.iter().copied().collect();

    let mut free_tiles: Vec<GridPos> = (0..cfg.height as i32)
        .flat_map(|y| (0..cfg.width as i32).map(move |x| GridPos::new(x, y)))
        .filter(|p| grid.get(p.x, p.y) == Some(Tile::Free) && !blocked.contains(p))
        .collect();

    let free_count = free_tiles.len();
    let item_cfgs  = item_configs_for_free_count(cfg, free_count);
    let mut items  = Vec::new();

    for ic in &item_cfgs {
        let target = (ic.max_on_map / 2).max(1);
        free_tiles.shuffle(rng);
        for &pos in free_tiles.iter().take(target) {
            items.push(ItemState { pos, kind: ic.kind });
        }
    }

    items
}
