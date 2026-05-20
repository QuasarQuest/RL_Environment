// src/rl/obs.rs
//
// Builds the 55-float observation vector for the RL agent each tick.
// Reads directly from the ECS World — called by RlEnv::step().
//
// Borrow discipline:
//   world.resource::<T>() holds a shared borrow on World for its lifetime.
//   world.query*() needs &mut World. Therefore all resource reads must be
//   cloned / copied into locals BEFORE any query call. Queries must also
//   fully resolve (collect results into owned values) before the next query.
//
// Vector layout (indices):
//  [0..11]  self        (11 floats)
//  [11..19] enemy       (8 floats)
//  [19..27] obstacles   (8 floats — raycasts in 8 directions)
//  [27..36] gold x3     (9 floats — 3 nearest, padded)
//  [36..42] health x2   (6 floats — 2 nearest, padded)
//  [42..48] ammo x2     (6 floats — 2 nearest, padded)
//  [48..51] speedboost  (3 floats — 1 nearest, padded)
//  [51..53] own base    (2 floats)
//  [53..55] RESERVED    (2 floats, zeroed)
//
// Total: 55 floats. Shape is fixed regardless of game state.
//
// Changes from v1 (53 floats):
//   Self block:    added dist_to_base_norm [9], dist_to_nearest_gold_norm [10]
//   Enemy block:   added enemy gold_carried [18]
//   Reserved:      reduced from 3 to 2 (net +2 floats)

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

pub const OBS_SIZE: usize = 55;

// ── Intermediate owned structs ────────────────────────────────────────────────

struct MapParams {
    w:            f32,
    h:            f32,
    diagonal:     f32,
    melee_range:  i32,
    ranged_range: i32,
}

struct SimParams {
    tick:                 u64,
    match_duration_ticks: u64,
}

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
    gold:   u8,
    is_dead: bool,
}

// ── Public entry point ────────────────────────────────────────────────────────

pub fn build_obs(world: &mut World) -> Vec<f32> {
    let mut obs = vec![0.0f32; OBS_SIZE];

    // ── 1. Clone resource values before any query ──────────────────────────
    let map = {
        let r = world.resource::<WorldConfig>();
        let w = r.width  as f32;
        let h = r.height as f32;
        MapParams {
            w,
            h,
            diagonal:     (w * w + h * h).sqrt(),
            melee_range:  r.melee_range as i32,
            ranged_range: r.ranged_range as i32,
        }
    };
    let sim = {
        let r = world.resource::<SimConfig>();
        SimParams {
            tick:                 r.tick,
            match_duration_ticks: r.match_duration_ticks,
        }
    };

    // ── 2. Query RlAgent ───────────────────────────────────────────────────
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

    // ── 3. Tile under agent ────────────────────────────────────────────────
    let on_own_base = {
        let grid = world.resource::<Grid>();
        grid.get(agent.pos.x, agent.pos.y) == Some(Tile::Base(agent.team))
    };

    // ── 4. Nearest base position (needed for dist_to_base) ────────────────
    let base_pos: Option<GridPos> = {
        let grid = world.resource::<Grid>();
        grid.iter()
            .filter(|(_, _, tile)| *tile == Tile::Base(agent.team))
            .map(|(x, y, _)| GridPos::new(x as i32, y as i32))
            .min_by_key(|p| p.dist_sq(agent.pos))
    };

    // ── 5. [0..11] Self block ──────────────────────────────────────────────
    obs[0] = agent.pos.x as f32 / map.w;
    obs[1] = agent.pos.y as f32 / map.h;
    obs[2] = agent.hearts as f32 / crate::config::AGENT_MAX_HEARTS as f32;
    obs[3] = agent.ammo   as f32 / crate::config::AGENT_MAX_AMMO   as f32;
    obs[4] = agent.gold   as f32 / crate::config::AGENT_MAX_GOLD   as f32;
    obs[5] = if agent.has_speed { 1.0 } else { 0.0 };
    obs[6] = sim.tick as f32 / sim.match_duration_ticks as f32;
    obs[7] = if agent.is_dead { 1.0 } else { 0.0 };
    obs[8] = if agent.gold > 0 && on_own_base { 1.0 } else { 0.0 };
    // [9]  normalised Chebyshev distance to own base (0 = at base, 1 = far)
    obs[9] = if let Some(bp) = base_pos {
        chebyshev(agent.pos, bp) as f32 / map.diagonal
    } else {
        1.0
    };
    // [10] carrying-gold flag × dist_to_base — joint signal for "carry urgency"
    obs[10] = if agent.gold > 0 { obs[9] } else { 0.0 };

    // ── 6. Query all agents for nearest enemy ──────────────────────────────
    let enemy: Option<EnemySnap> = {
        let mut q = world.query::<(
            Entity, &GridPos, &Hearts, &Ammo, &GoldCarried, &Team, Has<RespawnIn>,
        )>();
        q.iter(world)
            .filter(|(e, _, _, _, _, team, _)| {
                *e != agent.entity && team.0 != agent.team
            })
            .min_by_key(|(_, pos, _, _, _, _, _)| pos.dist_sq(agent.pos))
            .map(|(_, pos, hearts, ammo, gold, _, dead)| EnemySnap {
                pos:     *pos,
                hearts:  hearts.0,
                ammo:    ammo.0,
                gold:    gold.0,
                is_dead: dead,
            })
    };

    // ── 7. [11..19] Enemy block ────────────────────────────────────────────
    if let Some(e) = enemy {
        obs[11] = (e.pos.x - agent.pos.x) as f32 / map.w;
        obs[12] = (e.pos.y - agent.pos.y) as f32 / map.h;
        obs[13] = e.hearts as f32 / crate::config::AGENT_MAX_HEARTS as f32;
        obs[14] = e.ammo   as f32 / crate::config::AGENT_MAX_AMMO   as f32;
        let dist = chebyshev(agent.pos, e.pos);
        obs[15] = if dist <= map.melee_range  { 1.0 } else { 0.0 };
        obs[16] = if dist <= map.ranged_range { 1.0 } else { 0.0 };
        obs[17] = if e.is_dead { 1.0 } else { 0.0 };
        // [18] enemy gold carried — signals whether enemy is a delivery threat
        obs[18] = e.gold as f32 / crate::config::AGENT_MAX_GOLD as f32;
    }

    // ── 8. [19..27] Raycasts ───────────────────────────────────────────────
    {
        let max_ray = map.w.max(map.h) as i32;
        let grid = world.resource::<Grid>();
        for (i, dir) in Dir::all().iter().enumerate() {
            obs[19 + i] = raycast(grid, agent.pos, *dir, max_ray);
        }
    }

    // ── 9. Collect items ───────────────────────────────────────────────────
    let item_positions: Vec<(GridPos, ItemKind)> = {
        let mut q = world.query::<(&GridPos, &Item)>();
        q.iter(world).map(|(pos, item)| (*pos, item.kind)).collect()
    };

    fill_items(&item_positions, agent.pos, ItemKind::Gold,       3, map.w, map.h, &mut obs[27..36]);
    fill_items(&item_positions, agent.pos, ItemKind::Health,     2, map.w, map.h, &mut obs[36..42]);
    fill_items(&item_positions, agent.pos, ItemKind::Ammo,       2, map.w, map.h, &mut obs[42..48]);
    fill_items(&item_positions, agent.pos, ItemKind::SpeedBoost, 1, map.w, map.h, &mut obs[48..51]);

    // ── 10. [51..53] Own base vector ───────────────────────────────────────
    if let Some(bp) = base_pos {
        obs[51] = (bp.x - agent.pos.x) as f32 / map.w;
        obs[52] = (bp.y - agent.pos.y) as f32 / map.h;
    }

    // [53..55] reserved — zeroed by default
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