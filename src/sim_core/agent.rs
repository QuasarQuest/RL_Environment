// src/sim_core/agent.rs
//
// Agent systems: movement, speed buffs, deposit.
// Mirrors the OnSimTick order from the old Bevy pipeline.

use std::collections::HashSet;
use crate::agent::action::{Action, Dir};
use crate::world::{coords::GridPos, grid::Grid, tile::Tile};
use super::state::AgentState;

pub fn tick_speed_buffs(agents: &mut [AgentState]) {
    for a in agents {
        if a.speed_buff > 0 { a.speed_buff -= 1; }
    }
}

pub fn apply_action(agents: &mut Vec<AgentState>, grid: &Grid, idx: usize, action: Action) {
    match action {
        Action::Move(dir) => apply_move(agents, grid, idx, dir),
        Action::Drop      => try_deposit(agents, grid, idx),
        Action::Attack(_) | Action::RangedAttack(_) | Action::Wait => {}
    }
}

fn apply_move(agents: &mut Vec<AgentState>, grid: &Grid, idx: usize, dir: Dir) {
    let moves = movement_tiles(&agents[idx]);

    let occupied: HashSet<GridPos> = agents.iter()
        .enumerate()
        .filter(|(i, _)| *i != idx)
        .map(|(_, a)| a.pos)
        .collect();

    let (dx, dy) = dir.delta();
    for _ in 0..moves {
        let next = agents[idx].pos.apply_delta(dx, dy);
        if grid.is_walkable(next.x, next.y) && !occupied.contains(&next) {
            agents[idx].pos = next;
        } else {
            break;
        }
    }
}

fn movement_tiles(a: &AgentState) -> u32 {
    if a.gold_carried == 0 {
        return if a.speed_buff > 0 { 2 } else { 1 };
    }
    // Speed scales down with gold carried; stochastic sub-tile movement.
    let eff = crate::config::GOLD_CARRY_SPEED
        .powi(a.gold_carried as i32)
        .clamp(0.0, 1.0);
    if eff >= 1.0      { if a.speed_buff > 0 { 2 } else { 1 } }
    else if eff <= 0.0 { 0 }
    else if rand::random::<f32>() < eff { 1 }
    else               { 0 }
}

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

/// Auto-deposit: runs every tick for any agent standing on its base.
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
