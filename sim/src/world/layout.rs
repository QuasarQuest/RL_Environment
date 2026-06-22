// src/world/layout.rs
//
// Obstacle generation + item-config resolution for a single episode. Base/agent
// placement is now procedural and randomised per episode (engine/builder.rs); the
// only spatial input here is the base centre, around which a spawn pocket is kept
// obstacle-free. All randomness uses the env RNG so episodes are seed-reproducible.

use rand::RngExt;
use rand::rngs::SmallRng;

use crate::entity::item::ItemSpawnConfig;
use super::config::{ObstacleCluster, ObstacleKind, ObstacleProfile, WorldConfig, SPAWN_POCKET_RADIUS};
use super::tile::Tile;

// ── Item configs ──────────────────────────────────────────────────────────────

pub fn item_configs_for_free_count(cfg: &WorldConfig, free_tiles: usize) -> Vec<ItemSpawnConfig> {
    cfg.item_density.to_spawn_configs(free_tiles)
}

// ── Protected pocket ────────────────────────────────────────────────────────────

/// Mark the SPAWN_POCKET_RADIUS square around `base` as protected (no obstacles).
fn mark_pocket(protected: &mut [bool], cfg: &WorldConfig, base: (i32, i32)) {
    let (w, h) = (cfg.width as i32, cfg.height as i32);
    let idx = |x: i32, y: i32| -> usize { y as usize * cfg.width + x as usize };
    for dy in -SPAWN_POCKET_RADIUS..=SPAWN_POCKET_RADIUS {
        for dx in -SPAWN_POCKET_RADIUS..=SPAWN_POCKET_RADIUS {
            let (px, py) = (base.0 + dx, base.1 + dy);
            if px >= 0 && px < w && py >= 0 && py < h {
                protected[idx(px, py)] = true;
            }
        }
    }
}

// ── Obstacle generation ───────────────────────────────────────────────────────

pub fn place_obstacles(
    tiles: &mut [Tile],
    cfg:   &WorldConfig,
    base:  (i32, i32),
    rng:   &mut SmallRng,
) {
    if !cfg.obstacle_clusters.is_empty() {
        let clusters = cfg.obstacle_clusters.clone();
        place_obstacle_clusters(tiles, cfg, base, &clusters, rng);
        return;
    }
    if matches!(cfg.obstacles.profile, ObstacleProfile::None) { return; }

    let w     = cfg.width  as i32;
    let h     = cfg.height as i32;
    let total = (w * h) as usize;
    let target = ((total as f32) * cfg.obstacles.density.clamp(0.0, 0.4)) as usize;
    if target == 0 { return; }

    let short     = cfg.short_side() as f32;
    let max_block = ((short * cfg.obstacles.max_block_fraction).round() as i32).max(1);
    let max_wall  = ((short * cfg.obstacles.max_wall_fraction ).round() as i32).max(1);

    let idx = |x: i32, y: i32| -> usize { y as usize * cfg.width + x as usize };

    let mut protected = vec![false; total];
    mark_pocket(&mut protected, cfg, base);

    let max_attempts = target * 500;
    let mut attempts = 0usize;
    // Running obstacle count, incremented as tiles are stamped — avoids an
    // O(width·height) rescan of the whole grid on every placement attempt.
    let mut current = 0usize;

    loop {
        if current >= target || attempts >= max_attempts { break; }
        attempts += 1;

        match pick_shape(cfg.obstacles.profile, rng) {
            Shape::Block => {
                let bw = rng.random_range(1..=max_block);
                let bh = rng.random_range(1..=max_block);
                if w - bw < 2 || h - bh < 2 { continue; }
                let ox = rng.random_range(1..w - bw - 1);
                let oy = rng.random_range(1..h - bh - 1);
                let clear = (0..bh).all(|dy|
                    (0..bw).all(|dx| {
                        let i = idx(ox + dx, oy + dy);
                        tiles[i] == Tile::Free && !protected[i]
                    })
                );
                if clear {
                    for dy in 0..bh {
                        for dx in 0..bw {
                            tiles[idx(ox + dx, oy + dy)] = Tile::Obstacle;
                            current += 1;
                        }
                    }
                }
            }

            Shape::Wall => {
                let length = rng.random_range(2..=max_wall);
                let horiz  = rng.random_range(0..2) == 0;
                let (ox, oy, end_x, end_y) = if horiz {
                    if w - length < 2 { continue; }
                    let ox = rng.random_range(1..w - length - 1);
                    let oy = rng.random_range(1..h - 2);
                    (ox, oy, ox + length, oy)
                } else {
                    if h - length < 2 { continue; }
                    let ox = rng.random_range(1..w - 2);
                    let oy = rng.random_range(1..h - length - 1);
                    (ox, oy, ox, oy + length)
                };
                if end_x >= w || end_y >= h { continue; }
                let clear = if horiz {
                    (ox..=end_x).all(|x| { let i = idx(x, oy);  tiles[i] == Tile::Free && !protected[i] })
                } else {
                    (oy..=end_y).all(|y| { let i = idx(ox, y);  tiles[i] == Tile::Free && !protected[i] })
                };
                if clear {
                    if horiz { for x in ox..=end_x { tiles[idx(x, oy)] = Tile::Obstacle; current += 1; } }
                    else      { for y in oy..=end_y { tiles[idx(ox, y)] = Tile::Obstacle; current += 1; } }
                }
            }

            Shape::Scatter => {
                let x = rng.random_range(1..w - 1);
                let y = rng.random_range(1..h - 1);
                let i = idx(x, y);
                if tiles[i] == Tile::Free && !protected[i] {
                    tiles[i] = Tile::Obstacle;
                    current += 1;
                }
            }
        }
    }
}

// ── Cluster-based placement ───────────────────────────────────────────────────

fn place_obstacle_clusters(
    tiles:    &mut [Tile],
    cfg:      &WorldConfig,
    base:     (i32, i32),
    clusters: &[ObstacleCluster],
    rng:      &mut SmallRng,
) {
    let w = cfg.width  as i32;
    let h = cfg.height as i32;
    let total = (w * h) as usize;
    let idx   = |x: i32, y: i32| -> usize { y as usize * cfg.width + x as usize };

    let mut protected = vec![false; total];
    mark_pocket(&mut protected, cfg, base);

    for cluster in clusters {
        let (bw, bh) = (cluster.size.0 as i32, cluster.size.1 as i32);
        let max_attempts = cluster.count * 200;
        let mut placed   = 0;
        let mut attempts = 0;

        while placed < cluster.count && attempts < max_attempts {
            attempts += 1;

            match cluster.kind {
                ObstacleKind::Block => {
                    if w - bw < 2 || h - bh < 2 { continue; }
                    let ox = rng.random_range(1..w - bw - 1);
                    let oy = rng.random_range(1..h - bh - 1);
                    let clear = (0..bh).all(|dy|
                        (0..bw).all(|dx| {
                            let i = idx(ox + dx, oy + dy);
                            tiles[i] == Tile::Free && !protected[i]
                        })
                    );
                    if clear {
                        for dy in 0..bh { for dx in 0..bw { tiles[idx(ox + dx, oy + dy)] = Tile::Obstacle; } }
                        placed += 1;
                    }
                }
                ObstacleKind::Wall => {
                    let horiz = rng.random_range(0..2) == 0;
                    let (len, perp) = if horiz { (bw, bh) } else { (bh, bw) };
                    if horiz && w - len < 2 { continue; }
                    if !horiz && h - len < 2 { continue; }
                    let ox = rng.random_range(1..w - len.max(perp));
                    let oy = rng.random_range(1..h - len.max(perp));
                    let clear = if horiz {
                        (ox..ox + len).all(|x| { let i = idx(x, oy); tiles[i] == Tile::Free && !protected[i] })
                    } else {
                        (oy..oy + len).all(|y| { let i = idx(ox, y); tiles[i] == Tile::Free && !protected[i] })
                    };
                    if clear {
                        if horiz { for x in ox..ox + len { tiles[idx(x, oy)] = Tile::Obstacle; } }
                        else      { for y in oy..oy + len { tiles[idx(ox, y)] = Tile::Obstacle; } }
                        placed += 1;
                    }
                }
                ObstacleKind::Scatter => {
                    let x = rng.random_range(1..w - 1);
                    let y = rng.random_range(1..h - 1);
                    let i = idx(x, y);
                    if tiles[i] == Tile::Free && !protected[i] {
                        tiles[i] = Tile::Obstacle;
                        placed += 1;
                    }
                }
            }
        }
    }
}

enum Shape { Block, Wall, Scatter }

fn pick_shape(profile: ObstacleProfile, rng: &mut SmallRng) -> Shape {
    match profile {
        ObstacleProfile::Blocks  => Shape::Block,
        ObstacleProfile::Walls   => Shape::Wall,
        ObstacleProfile::Scatter => Shape::Scatter,
        ObstacleProfile::None    => Shape::Scatter,
        ObstacleProfile::Mixed   => match rng.random_range(0u8..10) {
            0..=2 => Shape::Block,
            3..=6 => Shape::Wall,
            _     => Shape::Scatter,
        },
    }
}
