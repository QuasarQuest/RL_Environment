// src/engine/physics.rs
//
// Agent movement, speed buffs, deposit, and combat.

use crate::config::GOLD_CARRY_SPEED;
use crate::entity::agent::{Action, AgentState, Dir};
use crate::world::{coords::GridPos, grid::Grid, tile::Tile};

// ── Speed buffs ───────────────────────────────────────────────────────────────

pub fn tick_speed_buffs(agents: &mut [AgentState]) {
    for a in agents {
        if a.speed_buff > 0 { a.speed_buff -= 1; }
    }
}

// ── Movement ──────────────────────────────────────────────────────────────────

pub fn apply_action(agents: &mut Vec<AgentState>, grid: &Grid, idx: usize, action: Action) {
    match action {
        Action::Move(dir) => apply_move(agents, grid, idx, dir),
        Action::Wait      => {}
    }
}

fn apply_move(agents: &mut Vec<AgentState>, grid: &Grid, idx: usize, dir: Dir) {
    let moves = movement_tiles(&agents[idx]);
    if moves == 0 { return; }

    let (dx, dy) = dir.delta();
    for _ in 0..moves {
        let next = agents[idx].pos.apply_delta(dx, dy);
        if !grid.is_walkable(next.x, next.y) { break; }
        if agents.iter().enumerate().any(|(i, a)| i != idx && a.pos == next) { break; }
        agents[idx].pos = next;
    }
}

/// How many tiles the agent moves per tick.
/// Carrying gold reduces effective speed; agent halts when below 0.5 threshold.
fn movement_tiles(a: &AgentState) -> u32 {
    if a.gold_carried == 0 {
        return if a.speed_buff > 0 { 2 } else { 1 };
    }
    let eff = GOLD_CARRY_SPEED.powi(a.gold_carried as i32).clamp(0.0, 1.0);
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

/// Auto-targeting melee attack: hits the lowest-HP enemy within `range` tiles.
/// Returns true if an attack landed.
pub fn try_melee_attack(agents: &mut Vec<AgentState>, idx: usize, range: i32, damage: u8) -> bool {
    let attacker_pos  = agents[idx].pos;
    let attacker_team = agents[idx].team;

    let target = (0..agents.len())
        .filter(|&i| {
            i != idx
                && agents[i].team != attacker_team
                && agents[i].hearts > 0
                && chebyshev(attacker_pos, agents[i].pos) <= range
        })
        .min_by_key(|&i| agents[i].hearts);

    if let Some(ti) = target {
        agents[ti].hearts = agents[ti].hearts.saturating_sub(damage);
        true
    } else {
        false
    }
}

/// Auto-targeting ranged attack: hits the lowest-HP enemy within `range` tiles.
/// Costs 1 ammo; does nothing if ammo is 0. Returns true if an attack landed.
pub fn try_ranged_attack(agents: &mut Vec<AgentState>, idx: usize, range: i32, damage: u8) -> bool {
    if agents[idx].ammo == 0 { return false; }
    let attacker_pos  = agents[idx].pos;
    let attacker_team = agents[idx].team;

    let target = (0..agents.len())
        .filter(|&i| {
            i != idx
                && agents[i].team != attacker_team
                && agents[i].hearts > 0
                && chebyshev(attacker_pos, agents[i].pos) <= range
        })
        .min_by_key(|&i| agents[i].hearts);

    if let Some(ti) = target {
        agents[idx].ammo              = agents[idx].ammo.saturating_sub(1);
        agents[ti].hearts             = agents[ti].hearts.saturating_sub(damage);
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

