// src/sim_core/world_builder.rs
//
// Constructs a fresh episode: grid, agents, items.
// Called on new() and reset().

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
    let mut agents: Vec<AgentState> = layout.agents.iter().map(|ra| {
        AgentState {
            pos:          GridPos::new(ra.x, ra.y),
            team:         ra.team,
            gold_carried: 0,
            score:        0,
            hearts:       config::AGENT_MAX_HEARTS,
            ammo:         config::AGENT_START_AMMO,
            speed_buff:   0,
            spawn_pos:    GridPos::new(ra.x, ra.y),
        }
    }).collect();
    agents.sort_by_key(|a| a.team);

    // Items placed at random free tiles, excluding agent spawn positions.
    let spawn_positions: std::collections::HashSet<GridPos> =
        agents.iter().map(|a| a.pos).collect();
    let free_count = tiles.iter().filter(|&&t| t == Tile::Free).count();
    let item_cfgs  = item_configs_for_free_count(cfg, free_count);
    let mut items  = Vec::new();
    for ic in &item_cfgs {
        let target       = (ic.max_on_map / 2).max(1);
        let mut placed   = 0;
        let mut attempts = 0;
        while placed < target && attempts < target * 200 {
            attempts += 1;
            let x = rand::random_range(0..cfg.width  as i32);
            let y = rand::random_range(0..cfg.height as i32);
            let pos = GridPos::new(x, y);
            if grid.get(x, y) == Some(Tile::Free) && !spawn_positions.contains(&pos) {
                items.push(ItemState { pos, kind: ic.kind });
                placed += 1;
            }
        }
    }

    WorldSnapshot { grid, agents, items }
}
