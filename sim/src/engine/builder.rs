// src/engine/builder.rs
//
// Constructs a fresh episode: grid, agent, items. Called on SimCore::new() and on
// every SimCore::reset(), both via the env RNG so episodes are seed-reproducible.
//
// Single-agent gold rush: the base is placed at a RANDOM tile each episode, kept
// inside a 7×7 spawn pocket that is ≥1 tile from every border and free of
// obstacles. The agent starts on its base. Randomising the base each episode
// stops the policy overfitting to one corner.

use std::collections::VecDeque;

use rand::RngExt;
use rand::seq::SliceRandom;
use rand::rngs::SmallRng;
use rustc_hash::FxHashSet;

use crate::config;
use crate::entity::{AgentState, ItemState};
use crate::entity::item::ItemKind;
use crate::engine::spawner;
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
    // guarantee the spawn is never sealed in, then commit and stamp the base.
    let mut grid = Grid::new(cfg.width, cfg.height);
    let mut tiles: Vec<Tile> = vec![Tile::Free; cfg.width * cfg.height];
    place_obstacles(&mut tiles, cfg, (bx, by), rng);
    ensure_escape(&mut tiles, w, h, base);
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
        move_energy:  0,
        mult_charge:  0,
        spawn_pos:    base,
        base_pos:     base,
        // Inert viewer-compat fields (see entity/agent.rs).
        team:   0,
        hearts: config::AGENT_MAX_HEARTS,
        ammo:   0,
    };
    let agents = vec![agent];

    // Items never spawn inside the spawn pocket (the agent always starts on a clear
    // patch of gold-free, item-free tiles around its base).
    let pocket = spawn_pocket(base, w, h);
    let items = spawn_items(cfg, &grid, &pocket, rng);
    WorldSnapshot { grid, agents, items }
}

/// Tiles of the SPAWN_POCKET_RADIUS square around `base`, clamped to the grid.
/// The pocket is kept obstacle-free (layout::place_obstacles) AND item-free
/// (spawn placement + the runtime spawner), so the agent always starts on an
/// open, empty patch around its base.
pub fn spawn_pocket(base: GridPos, w: i32, h: i32) -> Vec<GridPos> {
    let r = SPAWN_POCKET_RADIUS;
    let mut out = Vec::with_capacity(((2 * r + 1) * (2 * r + 1)) as usize);
    for dy in -r..=r {
        for dx in -r..=r {
            let (px, py) = (base.x + dx, base.y + dy);
            if px >= 0 && py >= 0 && px < w && py < h {
                out.push(GridPos::new(px, py));
            }
        }
    }
    out
}

/// Guarantee obstacles never completely seal the spawn in. The pocket itself is
/// already obstacle-free, but a dense ring around it could in principle trap the
/// agent in a tiny region. If a 4-connected flood from the base reaches fewer than
/// half of all walkable tiles, carve a straight corridor from the base to the map
/// centre (clearing any obstacles on it), which reconnects the pocket to the bulk.
fn ensure_escape(tiles: &mut [Tile], w: i32, h: i32, base: GridPos) {
    let reachable = flood_count(tiles, w, h, base);
    let free_total = tiles.iter().filter(|t| t.is_walkable()).count();
    if reachable * 2 >= free_total {
        return; // base is in the majority component — not boxed in
    }
    carve_line(tiles, w, h, base, GridPos::new(w / 2, h / 2));
}

/// Count walkable tiles reachable from `base` via 4-connected moves (a conservative
/// subset of the agent's 8-connected movement — if these are reachable, so are they
/// for the agent).
fn flood_count(tiles: &[Tile], w: i32, h: i32, base: GridPos) -> usize {
    let idx = |x: i32, y: i32| (y as usize) * (w as usize) + x as usize;
    let walkable = |x: i32, y: i32| x >= 0 && y >= 0 && x < w && y < h && tiles[idx(x, y)].is_walkable();
    if !walkable(base.x, base.y) {
        return 0;
    }
    let mut seen = vec![false; (w * h) as usize];
    let mut q = VecDeque::new();
    seen[idx(base.x, base.y)] = true;
    q.push_back(base);
    let mut count = 1;
    while let Some(c) = q.pop_front() {
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, ny) = (c.x + dx, c.y + dy);
            if walkable(nx, ny) && !seen[idx(nx, ny)] {
                seen[idx(nx, ny)] = true;
                count += 1;
                q.push_back(GridPos::new(nx, ny));
            }
        }
    }
    count
}

/// Clear obstacles along a Bresenham line from `a` to `b` (carving a corridor).
fn carve_line(tiles: &mut [Tile], w: i32, h: i32, a: GridPos, b: GridPos) {
    let idx = |x: i32, y: i32| (y as usize) * (w as usize) + x as usize;
    let (mut x, mut y) = (a.x, a.y);
    let dx = (b.x - a.x).abs();
    let dy = -(b.y - a.y).abs();
    let sx = if a.x < b.x { 1 } else { -1 };
    let sy = if a.y < b.y { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x >= 0 && y >= 0 && x < w && y < h {
            tiles[idx(x, y)] = Tile::Free;
        }
        if x == b.x && y == b.y {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn spawn_items(cfg: &WorldConfig, grid: &Grid, spawn_positions: &[GridPos], rng: &mut SmallRng) -> Vec<ItemState> {
    let blocked: FxHashSet<GridPos> = spawn_positions.iter().copied().collect();

    let free_tiles: Vec<GridPos> = (0..cfg.height as i32)
        .flat_map(|y| (0..cfg.width as i32).map(move |x| GridPos::new(x, y)))
        .filter(|p| grid.get(p.x, p.y) == Some(Tile::Free) && !blocked.contains(p))
        .collect();

    let free_count = free_tiles.len();
    let free_set: FxHashSet<GridPos> = free_tiles.iter().copied().collect();
    let item_cfgs  = item_configs_for_free_count(cfg, free_count);
    let mut items  = Vec::new();

    // Place each kind on its own shuffled prefix of free tiles. Reshuffling per
    // kind lets kinds overlap in candidate space without colliding (swap_remove
    // would be O(n²)); duplicates across kinds are avoided by skipping taken tiles.
    let mut taken: FxHashSet<GridPos> = FxHashSet::default();
    for ic in &item_cfgs {
        let target = (ic.max_on_map / 2).max(1);
        if ic.kind == ItemKind::Gold && cfg.item_density.gold_clustering.enabled() {
            place_gold_clustered(&mut items, target, &free_set, &mut taken, cfg, rng);
            continue;
        }
        // Non-gold (speed/multiplier): distribute around the map, far from gold clusters
        // and each other (farthest-point sampling; see spawner::pick_distributed).
        let avail: Vec<GridPos> = free_tiles.iter().copied().filter(|p| !taken.contains(p)).collect();
        let avoid: Vec<GridPos> = items.iter().map(|i| i.pos).collect();
        for pos in spawner::pick_distributed(&avail, &avoid, target, rng) {
            taken.insert(pos);
            items.push(ItemState { pos, kind: ic.kind });
        }
    }

    items
}

/// Place `target` gold as small clumps plus scattered singletons (see GoldClusterConfig).
/// `free_set` is the set of placeable tiles (free, outside the pocket); `taken` is the
/// running set of already-occupied tiles so nothing ever stacks. Never places more than
/// `target` gold and silently places fewer if the map runs out of room.
fn place_gold_clustered(
    items:    &mut Vec<ItemState>,
    target:   usize,
    free_set: &FxHashSet<GridPos>,
    taken:    &mut FxHashSet<GridPos>,
    cfg:      &WorldConfig,
    rng:      &mut SmallRng,
) {
    let gc = &cfg.item_density.gold_clustering;
    let (lo, hi) = gc.clamped_clump();
    let n_clustered = ((target as f32) * gc.clustered_fraction.clamp(0.0, 1.0)).round() as usize;

    // Shuffled pool of all placeable tiles — clump centres and scattered singletons are
    // both drawn from its front, skipping any tile already taken.
    let mut pool: Vec<GridPos> = free_set.iter().copied().collect();
    pool.shuffle(rng);
    let mut cursor = 0usize;
    let next_free = |cursor: &mut usize, taken: &FxHashSet<GridPos>| -> Option<GridPos> {
        while *cursor < pool.len() {
            let p = pool[*cursor];
            *cursor += 1;
            if !taken.contains(&p) { return Some(p); }
        }
        None
    };

    // ── Clumps ──
    let mut placed = 0usize;
    while placed < n_clustered {
        let Some(centre) = next_free(&mut cursor, taken) else { break };
        // Keep distinct clumps from merging into an oversized blob: reject a centre that
        // sits within 2*radius of existing gold (clump members live within `radius` of a
        // centre, so 2*radius of clearance guarantees two clumps never touch).
        if items.iter().any(|i| i.kind == ItemKind::Gold
            && (i.pos.x - centre.x).abs().max((i.pos.y - centre.y).abs()) <= 2 * gc.radius) {
            continue;
        }
        taken.insert(centre);
        items.push(ItemState { pos: centre, kind: ItemKind::Gold });
        placed += 1;

        // Remaining members of this clump, drawn from free tiles within `radius`.
        let want = rng.random_range(lo..=hi).min(n_clustered - placed + 1);
        let mut neigh: Vec<GridPos> = (-gc.radius..=gc.radius)
            .flat_map(|dy| (-gc.radius..=gc.radius).map(move |dx| (dx, dy)))
            .filter(|&(dx, dy)| dx != 0 || dy != 0)
            .map(|(dx, dy)| GridPos::new(centre.x + dx, centre.y + dy))
            .filter(|p| free_set.contains(p) && !taken.contains(p))
            .collect();
        neigh.shuffle(rng);
        for p in neigh.into_iter().take(want.saturating_sub(1)) {
            if placed >= n_clustered { break; }
            taken.insert(p);
            items.push(ItemState { pos: p, kind: ItemKind::Gold });
            placed += 1;
        }
    }

    // ── Scattered singletons ── (the remaining budget)
    while placed < target {
        let Some(p) = next_free(&mut cursor, taken) else { break };
        taken.insert(p);
        items.push(ItemState { pos: p, kind: ItemKind::Gold });
        placed += 1;
    }
}

