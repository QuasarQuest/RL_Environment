// src/engine/physics.rs
//
// Agent movement, buff ticking and deposit. Single-agent gold rush — no combat.

use crate::config::DEPOSIT_MULTIPLIER;
use crate::entity::agent::{Action, AgentState, Dir};
use crate::world::{grid::Grid, tile::Tile};

// ── Per-tick passive buff decay ─────────────────────────────────────────────────

/// Decrement every active buff timer by one tick.
pub fn tick_buffs(agents: &mut [AgentState]) {
    for a in agents {
        if a.speed_buff > 0 { a.speed_buff -= 1; }
        if a.slow_buff  > 0 { a.slow_buff  -= 1; }
        if a.mult_buff  > 0 { a.mult_buff  -= 1; }
    }
}

// ── Movement ──────────────────────────────────────────────────────────────────

pub fn apply_action(
    agents:      &mut Vec<AgentState>,
    grid:        &Grid,
    idx:         usize,
    action:      Action,
    carry_speed: f32,
    tick:        u64,
) {
    match action {
        Action::Move(dir) => apply_move(agents, grid, idx, dir, carry_speed, tick),
        Action::Wait      => {}
    }
}

fn apply_move(
    agents:      &mut Vec<AgentState>,
    grid:        &Grid,
    idx:         usize,
    dir:         Dir,
    carry_speed: f32,
    tick:        u64,
) {
    let moves = movement_tiles(&agents[idx], carry_speed, tick);
    if moves == 0 { return; }

    let (dx, dy) = dir.delta();
    for _ in 0..moves {
        let next = agents[idx].pos.apply_delta(dx, dy);
        if !grid.is_walkable(next.x, next.y) { break; }
        agents[idx].pos = next;
    }
}

/// How many tiles the agent moves this tick.
///   base                 : 1 tile
///   speed_buff active    : 2 tiles
///   slow_buff active     : 0.5 tiles → 1 tile on even ticks, 0 on odd (slow wins over speed)
///   carrying gold        : carry_speed^gold_carried; halts (0) once that drops below 0.5
fn movement_tiles(a: &AgentState, carry_speed: f32, tick: u64) -> u32 {
    // Load check first: a heavily-laden agent can't move regardless of buffs.
    if a.gold_carried > 0 {
        let eff = carry_speed.powi(a.gold_carried as i32).clamp(0.0, 1.0);
        if eff < 0.5 { return 0; }
    }
    if a.slow_buff > 0 {
        // Half speed: move on even ticks only. Slow dominates an active speed buff.
        return if tick % 2 == 0 { 1 } else { 0 };
    }
    if a.speed_buff > 0 { 2 } else { 1 }
}

// ── Deposit ───────────────────────────────────────────────────────────────────

/// Auto-deposit — runs every tick for the agent standing on its base. Banks all
/// carried gold; the Multiplier buff doubles the score gained from this deposit.
pub fn auto_deposit(agents: &mut Vec<AgentState>, grid: &Grid) {
    for i in 0..agents.len() {
        let (pos, gold) = {
            let a = &agents[i];
            (a.pos, a.gold_carried)
        };
        if gold == 0 { continue; }
        if !matches!(grid.get(pos.x, pos.y), Some(Tile::Base(_))) { continue; }
        let a = &mut agents[i];
        let mult = if a.mult_buff > 0 { DEPOSIT_MULTIPLIER } else { 1 };
        a.score        += a.gold_carried as u32 * mult;
        a.gold_carried  = 0;
    }
}
