// src/rl/obs.rs
//
// Builds the 53-float observation vector for the RL agent each tick.
// Reads directly from the ECS World — called by RlEnv::step().
//
// Borrow discipline:
//   world.resource::<T>() holds a shared borrow on World for its lifetime.
//   world.query*() needs &mut World. Therefore all resource reads must be
//   cloned / copied into locals BEFORE any query call. Queries must also
//   fully resolve (collect results into owned values) before the next query.
//
// Vector layout (indices):
//  [0..9]   self       (9 floats)
//  [9..16]  enemy      (7 floats)
//  [16..24] obstacles  (8 floats — raycasts in 8 directions)
//  [24..33] gold x3    (9 floats — 3 nearest, padded)
//  [33..39] health x2  (6 floats — 2 nearest, padded)
//  [39..45] ammo x2    (6 floats — 2 nearest, padded)
//  [45..48] speedboost (3 floats — 1 nearest, padded)
//  [48..50] own base   (2 floats)
//  [50..53] RESERVED   (3 floats, zeroed)
//
// Total: 53 floats. Shape is fixed regardless of game state.

use bevy::prelude::*;
use crate::agent::action::Dir;
use crate::agent::components::{Ammo, GoldCarried, Hearts, RespawnIn, SpeedBuff};
use crate::item::{Item, ItemKind};
use crate::sim::config::SimConfig;
use crate::team::Team;
use crate::world::config::WorldConfig;
use crate::world::coords::GridPos;
use crate::world::tile::Tile;
use crate::world::Grid;
use super::marker::RlAgent;

pub const OBS_SIZE: usize = 53;

// ── Intermediate owned structs — avoids holding World borrows across queries ──

struct MapParams { w: f32, h: f32, melee_range: i32, ranged_range: i32 }
struct SimParams  { tick: u64, match_duration_ticks: u64 }

struct AgentSnap {
    entity:    Entity,
    pos:       GridPos,
    hearts:    u8,
    ammo:      u8,
    gold:      u8,
    has_speed: bool,
    is_dead:   bool,
    team:      u8,
}

struct EnemySnap {
    pos:    GridPos,
    hearts: u8,
    ammo:   u8,
    is_dead: bool,
}

// ── Public entry point ────────────────────────────────────────────────────────

pub fn build_obs(world: &mut World) -> Vec<f32> {
    let mut obs = vec![0.0f32; OBS_SIZE];

    // ── 1. Clone resource values out before any query ─────────────────────────
    let map = {
        let r = world.resource::<WorldConfig>();
        MapParams {
            w:            r.width  as f32,
            h:            r.height as f32,
            melee_range:  r.melee_range,
            ranged_range: r.ranged_range,
        }
    };
    let sim = {
        let r = world.resource::<SimConfig>();
        SimParams { tick: r.tick, match_duration_ticks: r.match_duration_ticks }
    };

    // ── 2. Query RlAgent — collect into owned AgentSnap ───────────────────────
    let agent: Option<AgentSnap> = {
        let mut q = world.query_filtered::<(
            Entity, &GridPos, &Hearts, &Ammo, &GoldCarried, &Team,
            Has<SpeedBuff>, Has<RespawnIn>,
        ), With<RlAgent>>();
        q.single(world).ok().map(|(e, pos, hearts, ammo, gold, team, spd, dead)| {
            AgentSnap {
                entity:    e,
                pos:       *pos,
                hearts:    hearts.0,
                ammo:      ammo.0,
                gold:      gold.0,
                has_speed: spd,
                is_dead:   dead,
                team:      team.0,
            }
        })
    };

    let Some(agent) = agent else { return obs };

    // ── 3. Read tile under agent — clone Grid tile only ───────────────────────
    let on_own_base = {
        let grid = world.resource::<Grid>();
        grid.get(agent.pos.x, agent.pos.y) == Some(Tile::Base(agent.team))
    };

    // ── 4. [0..9] Self ────────────────────────────────────────────────────────
    obs[0] = agent.pos.x as f32 / map.w;
    obs[1] = agent.pos.y as f32 / map.h;
    obs[2] = agent.hearts as f32 / crate::config::AGENT_MAX_HEARTS as f32;
    obs[3] = agent.ammo   as f32 / crate::config::AGENT_MAX_AMMO   as f32;
    obs[4] = agent.gold   as f32 / crate::config::AGENT_MAX_GOLD   as f32;
    obs[5] = if agent.has_speed { 1.0 } else { 0.0 };
    obs[6] = sim.tick as f32 / sim.match_duration_ticks as f32;
    obs[7] = if agent.is_dead { 1.0 } else { 0.0 };
    obs[8] = if agent.gold > 0 && on_own_base { 1.0 } else { 0.0 };

    // ── 5. Query all agents — collect into Vec before dropping query ──────────
    let enemy: Option<EnemySnap> = {
        let mut q = world.query::<(Entity, &GridPos, &Hearts, &Ammo, &Team, Has<RespawnIn>)>();
        q.iter(world)
            .filter(|(e, _, _, _, team, _)| *e != agent.entity && team.0 != agent.team)
            .min_by_key(|(_, pos, _, _, _, _)| pos.dist_sq(agent.pos))
            .map(|(_, pos, hearts, ammo, _, dead)| EnemySnap {
                pos:     *pos,
                hearts:  hearts.0,
                ammo:    ammo.0,
                is_dead: dead,
            })
    };

    // ── 6. [9..16] Enemy ─────────────────────────────────────────────────────
    if let Some(e) = enemy {
        obs[9]  = (e.pos.x - agent.pos.x) as f32 / map.w;
        obs[10] = (e.pos.y - agent.pos.y) as f32 / map.h;
        obs[11] = e.hearts as f32 / crate::config::AGENT_MAX_HEARTS as f32;
        obs[12] = e.ammo   as f32 / crate::config::AGENT_MAX_AMMO   as f32;
        let dist = chebyshev(agent.pos, e.pos);
        obs[13] = if dist <= map.melee_range  { 1.0 } else { 0.0 };
        obs[14] = if dist <= map.ranged_range { 1.0 } else { 0.0 };
        obs[15] = if e.is_dead { 1.0 } else { 0.0 };
    }

    // ── 7. [16..24] Raycasts — clone grid snapshot ────────────────────────────
    {
        let max_ray = map.w.max(map.h) as i32;
        // Clone just the tiles we need via a local grid read.
        // Grid::get takes &self so we can borrow immutably here — no query needed.
        let grid = world.resource::<Grid>();
        for (i, dir) in Dir::all().iter().enumerate() {
            obs[16 + i] = raycast(grid, agent.pos, *dir, max_ray);
        }
    }

    // ── 8. Collect item positions — one query, fully owned ────────────────────
    let item_positions: Vec<(GridPos, ItemKind)> = {
        let mut q = world.query::<(&GridPos, &Item)>();
        q.iter(world).map(|(pos, item)| (*pos, item.kind)).collect()
    };

    fill_items(&item_positions, agent.pos, ItemKind::Gold,       3, map.w, map.h, &mut obs[24..33]);
    fill_items(&item_positions, agent.pos, ItemKind::Health,     2, map.w, map.h, &mut obs[33..39]);
    fill_items(&item_positions, agent.pos, ItemKind::Ammo,       2, map.w, map.h, &mut obs[39..45]);
    fill_items(&item_positions, agent.pos, ItemKind::SpeedBoost, 1, map.w, map.h, &mut obs[45..48]);

    // ── 9. [48..50] Own base — grid borrow, no query ──────────────────────────
    {
        let grid = world.resource::<Grid>();
        let base = grid.iter()
            .filter(|(_, _, tile)| *tile == Tile::Base(agent.team))
            .map(|(x, y, _)| GridPos::new(x as i32, y as i32))
            .min_by_key(|p| p.dist_sq(agent.pos));
        if let Some(base_pos) = base {
            obs[48] = (base_pos.x - agent.pos.x) as f32 / map.w;
            obs[49] = (base_pos.y - agent.pos.y) as f32 / map.h;
        }
    }

    // [50..53] reserved — zeroed by default
    obs
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn chebyshev(a: GridPos, b: GridPos) -> i32 {
    (a.x - b.x).abs().max((a.y - b.y).abs())
}

fn raycast(grid: &Grid, origin: GridPos, dir: Dir, max_len: i32) -> f32 {
    let (dx, dy) = dir.delta();
    for dist in 1..=max_len {
        let x = origin.x + dx * dist;
        let y = origin.y + dy * dist;
        match grid.get(x, y) {
            None                 => return dist as f32 / max_len as f32,
            Some(Tile::Obstacle) => return dist as f32 / max_len as f32,
            _                    => {}
        }
    }
    1.0
}

/// Fill `k` nearest items of `kind` into `slot` (3 floats each: rel_x, rel_y, exists).
/// Takes a pre-collected owned Vec — no World borrow needed.
fn fill_items(
    items:  &[(GridPos, ItemKind)],
    origin: GridPos,
    kind:   ItemKind,
    k:      usize,
    map_w:  f32,
    map_h:  f32,
    slot:   &mut [f32],
) {
    let mut positions: Vec<GridPos> = items.iter()
        .filter(|(_, k2)| *k2 == kind)
        .map(|(pos, _)| *pos)
        .collect();
    positions.sort_by_key(|p| p.dist_sq(origin));

    for i in 0..k {
        let base = i * 3;
        if let Some(pos) = positions.get(i) {
            slot[base]     = (pos.x - origin.x) as f32 / map_w;
            slot[base + 1] = (pos.y - origin.y) as f32 / map_h;
            slot[base + 2] = 1.0;
        }
    }
}