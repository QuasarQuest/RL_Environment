// src/engine/physics.rs
//
// Agent movement, speed buffs, cooldowns, respawn, deposit, and combat.

use crate::config::{AGENT_MAX_HEARTS, AGENT_START_AMMO};
use crate::entity::agent::{Action, AgentState, Dir};
use crate::world::{coords::GridPos, grid::Grid, tile::Tile};

// ── Per-tick passive ticks ────────────────────────────────────────────────────

pub fn tick_speed_buffs(agents: &mut [AgentState]) {
    for a in agents {
        if a.speed_buff > 0 { a.speed_buff -= 1; }
    }
}

pub fn tick_cooldowns(agents: &mut [AgentState]) {
    for a in agents {
        if a.melee_cooldown  > 0 { a.melee_cooldown  -= 1; }
        if a.ranged_cooldown > 0 { a.ranged_cooldown -= 1; }
    }
}

/// Counts down respawn timers; restores full health and returns agent to spawn_pos when ready.
/// Returns a bitmask of agent indices that completed their respawn this tick (for cache clearing).
pub fn tick_respawns(agents: &mut [AgentState]) -> u64 {
    let mut just_respawned: u64 = 0;
    for (i, a) in agents.iter_mut().enumerate() {
        if a.respawn_timer == 0 { continue; }
        a.respawn_timer -= 1;
        if a.respawn_timer == 0 {
            a.hearts = AGENT_MAX_HEARTS;
            a.ammo   = AGENT_START_AMMO;
            a.pos    = a.spawn_pos;
            just_respawned |= 1 << i;
        }
    }
    just_respawned
}

// ── Movement ──────────────────────────────────────────────────────────────────

pub fn apply_action(agents: &mut Vec<AgentState>, grid: &Grid, idx: usize, action: Action, carry_speed: f32) {
    if agents[idx].respawn_timer > 0 { return; }
    match action {
        Action::Move(dir) => apply_move(agents, grid, idx, dir, carry_speed),
        Action::Wait      => {}
    }
}

fn apply_move(agents: &mut Vec<AgentState>, grid: &Grid, idx: usize, dir: Dir, carry_speed: f32) {
    let moves = movement_tiles(&agents[idx], carry_speed);
    if moves == 0 { return; }

    let (dx, dy) = dir.delta();
    for _ in 0..moves {
        let next = agents[idx].pos.apply_delta(dx, dy);
        if !grid.is_walkable(next.x, next.y) { break; }
        // Skip agents that are respawning — they don't block movement.
        if agents.iter().enumerate().any(|(i, a)| i != idx && a.pos == next && a.respawn_timer == 0) { break; }
        agents[idx].pos = next;
    }
}

/// How many tiles the agent moves per tick.
/// Carrying gold reduces effective speed via carry_speed^gold_carried; halts below 0.5.
fn movement_tiles(a: &AgentState, carry_speed: f32) -> u32 {
    if a.gold_carried == 0 {
        return if a.speed_buff > 0 { 2 } else { 1 };
    }
    let eff = carry_speed.powi(a.gold_carried as i32).clamp(0.0, 1.0);
    if eff >= 0.5 {
        if a.speed_buff > 0 { 2 } else { 1 }
    } else {
        0
    }
}

// ── Deposit ───────────────────────────────────────────────────────────────────

/// Explicit Drop action — deposit if standing on own base.
pub fn try_deposit(agents: &mut Vec<AgentState>, grid: &Grid, idx: usize) {
    let (pos, team, gold) = {
        let a = &agents[idx];
        (a.pos, a.team, a.gold_carried)
    };
    if gold == 0 { return; }
    if grid.get(pos.x, pos.y) != Some(Tile::Base(team)) { return; }
    let a = &mut agents[idx];
    a.score        += a.gold_carried as u32;
    a.gold_carried  = 0;
}

/// Auto-deposit — runs every tick for any agent standing on its base.
pub fn auto_deposit(agents: &mut Vec<AgentState>, grid: &Grid) {
    for i in 0..agents.len() {
        let (pos, team, gold) = {
            let a = &agents[i];
            (a.pos, a.team, a.gold_carried)
        };
        if gold == 0 { continue; }
        if grid.get(pos.x, pos.y) != Some(Tile::Base(team)) { continue; }
        let a = &mut agents[i];
        a.score        += a.gold_carried as u32;
        a.gold_carried  = 0;
    }
}

// ── Combat ────────────────────────────────────────────────────────────────────

/// Auto-targeting melee attack: hits the lowest-HP alive enemy within `range` tiles.
/// Respects melee_cooldown; sets it to `cooldown_ticks` on a successful hit.
/// On kill: sets target's respawn_timer, drops its gold, increments attacker's kills.
/// Returns true if an attack landed.
pub fn try_melee_attack(
    agents:         &mut Vec<AgentState>,
    idx:            usize,
    range:          i32,
    damage:         u8,
    cooldown_ticks: u8,
    respawn_ticks:  u8,
) -> bool {
    if agents[idx].melee_cooldown > 0 || agents[idx].respawn_timer > 0 { return false; }
    let attacker_pos  = agents[idx].pos;
    let attacker_team = agents[idx].team;

    let target = (0..agents.len())
        .filter(|&i| {
            i != idx
                && agents[i].team != attacker_team
                && agents[i].hearts > 0
                && agents[i].respawn_timer == 0
                && chebyshev(attacker_pos, agents[i].pos) <= range
        })
        .min_by_key(|&i| agents[i].hearts);

    if let Some(ti) = target {
        agents[idx].melee_cooldown = cooldown_ticks;
        agents[ti].hearts = agents[ti].hearts.saturating_sub(damage);
        if agents[ti].hearts == 0 {
            agents[ti].respawn_timer = respawn_ticks;
            agents[ti].gold_carried  = 0;
            agents[idx].kills       += 1;
        }
        true
    } else {
        false
    }
}

/// Auto-targeting ranged attack: hits the lowest-HP alive enemy within `range` tiles.
/// Costs 1 ammo; does nothing if ammo is 0 or ranged_cooldown > 0.
/// On kill: sets target's respawn_timer, drops its gold, increments attacker's kills.
/// Returns true if an attack landed.
pub fn try_ranged_attack(
    agents:         &mut Vec<AgentState>,
    idx:            usize,
    range:          i32,
    damage:         u8,
    cooldown_ticks: u8,
    respawn_ticks:  u8,
) -> bool {
    if agents[idx].ranged_cooldown > 0 || agents[idx].ammo == 0 || agents[idx].respawn_timer > 0 {
        return false;
    }
    let attacker_pos  = agents[idx].pos;
    let attacker_team = agents[idx].team;

    let target = (0..agents.len())
        .filter(|&i| {
            i != idx
                && agents[i].team != attacker_team
                && agents[i].hearts > 0
                && agents[i].respawn_timer == 0
                && chebyshev(attacker_pos, agents[i].pos) <= range
        })
        .min_by_key(|&i| agents[i].hearts);

    if let Some(ti) = target {
        agents[idx].ranged_cooldown  = cooldown_ticks;
        agents[idx].ammo             = agents[idx].ammo.saturating_sub(1);
        agents[ti].hearts            = agents[ti].hearts.saturating_sub(damage);
        if agents[ti].hearts == 0 {
            agents[ti].respawn_timer = respawn_ticks;
            agents[ti].gold_carried  = 0;
            agents[idx].kills       += 1;
        }
        true
    } else {
        false
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline]
fn chebyshev(a: GridPos, b: GridPos) -> i32 {
    (a.x - b.x).abs().max((a.y - b.y).abs())
}

