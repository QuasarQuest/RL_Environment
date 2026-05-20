// src/world/layout.rs
//
// Layout resolution — converts declarative WorldConfig into concrete tile
// coordinates and spawn positions.
//
// Rules:
//   - No Bevy types except the Resource derive (needed so ResolvedLayout
//     can be stored as a Bevy resource for other systems to read).
//   - All randomness uses rand::random_range — no caller-supplied RNG needed
//     because reproducibility is handled at the config/seed level.
//   - This module is the ONLY place that converts fractions → tile coords.
//     world/plugin.rs calls these functions; nothing else should.

use bevy::prelude::Resource;
use crate::agent::strategy::StrategyKind;
use crate::agent::planner::PlannerKind;
use crate::item::spawner::ItemSpawnConfig;
use super::config::{AgentConfig, ObstacleProfile, SpawnIntent, WorldConfig};
use super::tile::Tile;

// ── Resolved types ────────────────────────────────────────────────────────────

/// A base that has been placed on the grid.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedBase {
    pub team: u8,
    pub x:    i32,
    pub y:    i32,
}

/// An agent ready to spawn — position already resolved, strategy chosen.
#[derive(Debug, Clone)]
pub struct ResolvedAgent {
    pub team:     u8,
    pub x:        i32,
    pub y:        i32,
    pub strategy: StrategyKind,
    pub planner:  PlannerKind,
}

/// Complete resolved layout — stored as a Bevy resource so agent spawners,
/// the factory display system, and the RL restart path can all read it
/// without recomputing positions.
#[derive(Debug, Clone, Resource)]
pub struct ResolvedLayout {
    pub bases:        Vec<ResolvedBase>,
    pub agents:       Vec<ResolvedAgent>,
    /// Populated after obstacle placement (call item_configs_for_free_count).
    pub item_configs: Vec<ItemSpawnConfig>,
}

// ── Public entry points ───────────────────────────────────────────────────────

/// Resolve bases and agent spawn positions. Item configs are empty — call
/// `item_configs_for_free_count` after obstacle placement to fill them.
pub fn resolve(cfg: &WorldConfig) -> ResolvedLayout {
    let bases  = resolve_bases(cfg);
    let agents = resolve_agents(cfg, &bases);
    ResolvedLayout { bases, agents, item_configs: Vec::new() }
}

/// Compute item spawn configs once the free tile count is known.
pub fn item_configs_for_free_count(cfg: &WorldConfig, free_tiles: usize) -> Vec<ItemSpawnConfig> {
    cfg.item_density.to_spawn_configs(free_tiles)
}

// ── Bases ─────────────────────────────────────────────────────────────────────

fn resolve_bases(cfg: &WorldConfig) -> Vec<ResolvedBase> {
    cfg.bases.iter().map(|base_cfg| {
        let (x, y) = base_cfg.corner.resolve(cfg.width, cfg.height, base_cfg.margin);
        ResolvedBase { team: base_cfg.team, x, y }
    }).collect()
}

// ── Agents ────────────────────────────────────────────────────────────────────

fn resolve_agents(cfg: &WorldConfig, bases: &[ResolvedBase]) -> Vec<ResolvedAgent> {
    cfg.agents.iter().map(|agent_cfg| {
        let (x, y) = resolve_spawn(cfg, agent_cfg, bases);
        ResolvedAgent {
            team:     agent_cfg.team,
            x,
            y,
            strategy: agent_cfg.strategy,
            planner:  agent_cfg.planner,
        }
    }).collect()
}

fn resolve_spawn(cfg: &WorldConfig, agent: &AgentConfig, bases: &[ResolvedBase]) -> (i32, i32) {
    match agent.spawn {
        SpawnIntent::NearBase => {
            let base = bases.iter()
                .find(|b| b.team == agent.team)
                .expect("resolve_spawn: no base for team — validate() should have caught this");

            let offset = ((cfg.short_side() as f32) * agent.spawn_offset)
                .round() as i32;
            let offset = offset.max(1);

            let cx = cfg.width  as i32 / 2;
            let cy = cfg.height as i32 / 2;
            let dx = (cx - base.x).signum();
            let dy = (cy - base.y).signum();

            let x = (base.x + dx * offset).clamp(0, cfg.width  as i32 - 1);
            let y = (base.y + dy * offset).clamp(0, cfg.height as i32 - 1);
            (x, y)
        }
        SpawnIntent::Centre => {
            (cfg.width as i32 / 2, cfg.height as i32 / 2)
        }
    }
}

// ── Obstacle generation ───────────────────────────────────────────────────────

/// Place obstacles into `tiles` (row-major, width × height) until `density`
/// fraction of tiles are occupied or attempts are exhausted.
///
/// Protected tiles (base areas, spawn perimeters) are never overwritten.
pub fn place_obstacles(tiles: &mut Vec<Tile>, cfg: &WorldConfig, layout: &ResolvedLayout) {
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

    // ── Protection mask ───────────────────────────────────────────────────────
    let safe_radius = cfg.safe_zone_radius();
    let mut protected = vec![false; total];

    for base in &layout.bases {
        for dy in -safe_radius..=safe_radius {
            for dx in -safe_radius..=safe_radius {
                let px = base.x + dx;
                let py = base.y + dy;
                if px >= 0 && px < w && py >= 0 && py < h {
                    protected[idx(px, py)] = true;
                }
            }
        }
    }
    const SPAWN_CLEAR: i32 = 2;
    for agent in &layout.agents {
        for dy in -SPAWN_CLEAR..=SPAWN_CLEAR {
            for dx in -SPAWN_CLEAR..=SPAWN_CLEAR {
                let px = agent.x + dx;
                let py = agent.y + dy;
                if px >= 0 && px < w && py >= 0 && py < h {
                    protected[idx(px, py)] = true;
                }
            }
        }
    }

    // ── Placement loop ────────────────────────────────────────────────────────
    // Count by scanning rather than a closure so there is no immutable borrow
    // alive when we mutate the slice.
    let max_attempts = target * 500;
    let mut attempts = 0usize;

    loop {
        let current = tiles.iter().filter(|&&t| t == Tile::Obstacle).count();
        if current >= target || attempts >= max_attempts { break; }
        attempts += 1;

        match pick_shape(cfg.obstacles.profile) {
            Shape::Block => {
                let bw = rand::random_range(1..=max_block);
                let bh = rand::random_range(1..=max_block);
                if w - bw < 2 || h - bh < 2 { continue; }
                let ox = rand::random_range(1..w - bw - 1);
                let oy = rand::random_range(1..h - bh - 1);
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
                        }
                    }
                }
            }

            Shape::Wall => {
                let length = rand::random_range(2..=max_wall);
                let horiz  = rand::random_range(0..2) == 0;
                let (ox, oy, end_x, end_y) = if horiz {
                    if w - length < 2 { continue; }
                    let ox = rand::random_range(1..w - length - 1);
                    let oy = rand::random_range(1..h - 2);
                    (ox, oy, ox + length, oy)
                } else {
                    if h - length < 2 { continue; }
                    let ox = rand::random_range(1..w - 2);
                    let oy = rand::random_range(1..h - length - 1);
                    (ox, oy, ox, oy + length)
                };
                if end_x >= w || end_y >= h { continue; }
                let clear = if horiz {
                    (ox..=end_x).all(|x| { let i = idx(x, oy);  tiles[i] == Tile::Free && !protected[i] })
                } else {
                    (oy..=end_y).all(|y| { let i = idx(ox, y);  tiles[i] == Tile::Free && !protected[i] })
                };
                if clear {
                    if horiz { for x in ox..=end_x { tiles[idx(x, oy)] = Tile::Obstacle; } }
                    else      { for y in oy..=end_y { tiles[idx(ox, y)] = Tile::Obstacle; } }
                }
            }

            Shape::Scatter => {
                let x = rand::random_range(1..w - 1);
                let y = rand::random_range(1..h - 1);
                let i = idx(x, y);
                if tiles[i] == Tile::Free && !protected[i] {
                    tiles[i] = Tile::Obstacle;
                }
            }
        }
    }
}

enum Shape { Block, Wall, Scatter }

fn pick_shape(profile: ObstacleProfile) -> Shape {
    match profile {
        ObstacleProfile::Blocks  => Shape::Block,
        ObstacleProfile::Walls   => Shape::Wall,
        ObstacleProfile::Scatter => Shape::Scatter,
        ObstacleProfile::None    => Shape::Scatter,
        ObstacleProfile::Mixed   => match rand::random_range(0u8..10) {
            0..=2 => Shape::Block,
            3..=6 => Shape::Wall,
            _     => Shape::Scatter,
        },
    }
}

// ── Free tile count ───────────────────────────────────────────────────────────

pub fn count_free_tiles(tiles: &[Tile]) -> usize {
    tiles.iter().filter(|&&t| t == Tile::Free).count()
}