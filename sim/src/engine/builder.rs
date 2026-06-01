// src/engine/builder.rs
//
// Constructs a fresh episode: grid, agent, items. Called on SimCore::new() and on
// every SimCore::reset(), both via the env RNG so episodes are seed-reproducible.
//
// Single-agent gold rush: the base is placed at a RANDOM tile each episode, kept
// inside a 7×7 spawn pocket that is ≥1 tile from every border and free of
// obstacles. The agent starts on its base. Randomising the base each episode
// stops the policy overfitting to one corner.

use rand::RngExt;
use rand::seq::SliceRandom;
use rand::rngs::SmallRng;
use rustc_hash::FxHashSet;

use crate::config;
use crate::entity::{AgentState, ItemState};
use crate::world::{
    config::{WorldConfig, SPAWN_POCKET_RADIUS},
    coords::GridPos,
    grid::Grid,
    layout::{item_configs_for_free_count, place_obstacles},
    tile::Tile,
};

pub struct WorldSnapshot {
    pub grid:   Grid,
    pub agents: Vec<AgentState>,
    pub items:  Vec<ItemState>,
}

/// Build one full episode (grid + agent + items) with a randomised base.
pub fn build_episode(cfg: &WorldConfig, rng: &mut SmallRng) -> WorldSnapshot {
    let (w, h) = (cfg.width as i32, cfg.height as i32);

    // Base centre: anywhere such that the 7×7 pocket stays ≥1 tile from the border.
    let lo = SPAWN_POCKET_RADIUS + 1;
    let bx = rng.random_range(lo..=(w - 1 - lo));
    let by = rng.random_range(lo..=(h - 1 - lo));
    let base = GridPos::new(bx, by);

    // Obstacles on a scratch tile buffer (pocket around base is protected), then
    // commit to the grid and stamp the base tile on top.
    let mut grid = Grid::new(cfg.width, cfg.height);
    let mut tiles: Vec<Tile> = vec![Tile::Free; cfg.width * cfg.height];
    place_obstacles(&mut tiles, cfg, (bx, by), rng);
    for y in 0..cfg.height {
        for x in 0..cfg.width {
            grid.set(x as i32, y as i32, tiles[y * cfg.width + x]);
        }
    }
    grid.set(bx, by, Tile::Base(0));

    let agent = AgentState {
        pos:          base,
        gold_carried: 0,
        score:        0,
        speed_buff:   0,
        slow_buff:    0,
        mult_buff:    0,
        spawn_pos:    base,
        base_pos:     base,
        // Inert viewer-compat fields (see entity/agent.rs).
        team:   0,
        hearts: config::AGENT_MAX_HEARTS,
        ammo:   0,
    };
    let agents = vec![agent];

    let items = spawn_items_internal(cfg, &grid, &[base], rng);
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

    // Place each kind on its own shuffled prefix of free tiles. Reshuffling per
    // kind lets kinds overlap in candidate space without colliding (swap_remove
    // would be O(n²)); duplicates across kinds are avoided by removing taken tiles.
    let mut taken: FxHashSet<GridPos> = FxHashSet::default();
    for ic in &item_cfgs {
        let target = (ic.max_on_map / 2).max(1);
        free_tiles.shuffle(rng);
        let mut placed = 0;
        for &pos in free_tiles.iter() {
            if placed >= target { break; }
            if taken.contains(&pos) { continue; }
            taken.insert(pos);
            items.push(ItemState { pos, kind: ic.kind });
            placed += 1;
        }
    }

    items
}
