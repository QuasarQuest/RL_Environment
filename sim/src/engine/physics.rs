// src/engine/physics.rs
//
// Agent movement, buff ticking and deposit. Single-agent gold rush — no combat.

use crate::config::DEPOSIT_MULTIPLIER;
use crate::entity::agent::{Action, AgentState, Dir};
use crate::world::{coords::GridPos, grid::Grid, tile::Tile};

// ── Per-tick passive buff decay ─────────────────────────────────────────────────

/// Decrement every active buff timer by one tick.
pub fn tick_buffs(agents: &mut [AgentState]) {
    for a in agents {
        if a.speed_buff > 0 { a.speed_buff -= 1; }
        if a.slow_buff  > 0 { a.slow_buff  -= 1; }
        if a.mult_buff  > 0 { a.mult_buff  -= 1; }
        if a.trap_buff  > 0 { a.trap_buff  -= 1; }
    }
}

// ── Movement ──────────────────────────────────────────────────────────────────

/// Apply a movement action. Returns `true` only when the agent walked straight
/// into a wall — a `Move` whose first step is blocked by a non-walkable tile, so
/// the agent made no progress this tick. Stalls for any other reason (slowed this
/// tick, `Wait`) are NOT wall hits, so they incur no wall penalty.
///
/// `stop_at` is the agent's navigation target this tick: a multi-tile (speed-buff)
/// move halts the instant it lands on that tile rather than sailing past it. Without
/// this a 2-tile move steps OVER its goal — pickup/auto_deposit check only the final
/// position, so the agent would skip its gold/base and oscillate around it forever.
pub fn apply_action(
    agents:  &mut Vec<AgentState>,
    grid:    &Grid,
    idx:     usize,
    action:  Action,
    tick:    u64,
    stop_at: Option<GridPos>,
) -> bool {
    match action {
        Action::Move(dir) => apply_move(agents, grid, idx, dir, tick, stop_at),
        Action::Wait      => false,
    }
}

/// Returns `true` iff the agent stepped directly into a wall (first step blocked,
/// no progress made this tick).
fn apply_move(
    agents:  &mut Vec<AgentState>,
    grid:    &Grid,
    idx:     usize,
    dir:     Dir,
    tick:    u64,
    stop_at: Option<GridPos>,
) -> bool {
    let moves = movement_tiles(&agents[idx], tick);
    if moves == 0 { return false; } // couldn't move this tick (e.g. slowed) — not a wall hit

    let (dx, dy) = dir.delta();
    let mut moved = false;
    for _ in 0..moves {
        if stop_at == Some(agents[idx].pos) { break; }
        let next = agents[idx].pos.apply_delta(dx, dy);
        if !grid.is_walkable(next.x, next.y) {
            // Blocked by a wall — a wall hit only if no progress was made at all
            // this tick (the agent stepped straight into the wall).
            return !moved;
        }
        agents[idx].pos = next;
        moved = true;
        if stop_at == Some(next) { break; }
    }
    false
}

/// How many tiles the agent moves this tick.
///   trap_buff active  : 0 tiles — fully immobilised (dominates everything)
///   slow_buff active  : 0.5 tiles → 1 tile on even ticks, 0 on odd (slow wins over speed)
///   speed_buff active : 2 tiles
///   base              : 1 tile
fn movement_tiles(a: &AgentState, tick: u64) -> u32 {
    if a.trap_buff > 0 {
        return 0; // trapped — cannot move at all until the trap window expires
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
