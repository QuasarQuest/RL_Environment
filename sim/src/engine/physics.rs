// src/engine/physics.rs
//
// Agent movement, buff ticking and deposit. Single-agent gold rush — no combat.

use rustc_hash::FxHashSet;

use crate::config::DEPOSIT_MULTIPLIER;
use crate::entity::agent::{Action, AgentState, Dir};
use crate::world::{coords::GridPos, grid::Grid, tile::Tile};

// ── Per-tick passive buff decay ─────────────────────────────────────────────────

/// Decrement every active buff timer by one tick. (The multiplier is a held charge,
/// not a timer, so it does not decay here — it is consumed on deposit.)
pub fn tick_buffs(agents: &mut [AgentState]) {
    for a in agents {
        if a.speed_buff > 0 { a.speed_buff -= 1; }
        if a.trap_buff  > 0 { a.trap_buff  -= 1; }
    }
}

// ── Movement ──────────────────────────────────────────────────────────────────

/// Apply a movement action. Returns `true` only when the agent walked straight
/// into a wall — a `Move` whose first step is blocked by a non-walkable tile, so
/// the agent made no progress this tick. Stalls for any other reason (slowed this
/// tick, `Wait`) are NOT wall hits, so they incur no wall penalty.
///
/// Movement is capped at a single tile per tick — the movement cadence (base vs.
/// speed-buffed) is handled upstream in the engine's move-energy accumulator, so this
/// just executes the one step the engine decided to take this tick. With at most one
/// tile of travel a move can no longer overshoot a turn, so `stop_at`/`blocked` are
/// only light guards: `stop_at` is the navigation goal (we never step off it once
/// reached) and `blocked` is the set of trap tiles the move refuses to enter even if
/// A* briefly points at one (e.g. a trap that just spawned onto the next waypoint).
pub fn apply_action(
    agents:  &mut Vec<AgentState>,
    grid:    &Grid,
    idx:     usize,
    action:  Action,
    _tick:   u64,
    stop_at: Option<GridPos>,
    blocked: &FxHashSet<GridPos>,
) -> bool {
    match action {
        Action::Move(dir) => apply_move(agents, grid, idx, dir, stop_at, blocked),
        Action::Wait      => false,
    }
}

/// Move one tile in `dir`. Returns `true` iff the step was blocked by a wall (the
/// agent stepped straight into it and made no progress).
fn apply_move(
    agents:  &mut Vec<AgentState>,
    grid:    &Grid,
    idx:     usize,
    dir:     Dir,
    stop_at: Option<GridPos>,
    blocked: &FxHashSet<GridPos>,
) -> bool {
    if stop_at == Some(agents[idx].pos) { return false; } // already on the goal

    let (dx, dy) = dir.delta();
    let next = agents[idx].pos.apply_delta(dx, dy);
    if !grid.is_walkable(next.x, next.y) {
        return true; // stepped straight into a wall — no progress this tick
    }
    if blocked.contains(&next) { return false; } // trap ahead — refuse to step on it
    agents[idx].pos = next;
    false
}

// ── Deposit ───────────────────────────────────────────────────────────────────

/// Auto-deposit — runs every tick for the agent standing on its base. Banks all
/// carried gold; a held multiplier charge doubles this deposit's value and is then
/// consumed (one charge per deposit).
pub fn auto_deposit(agents: &mut Vec<AgentState>, grid: &Grid) {
    for i in 0..agents.len() {
        let (pos, gold) = {
            let a = &agents[i];
            (a.pos, a.gold_carried)
        };
        if gold == 0 { continue; }
        if !matches!(grid.get(pos.x, pos.y), Some(Tile::Base(_))) { continue; }
        let a = &mut agents[i];
        let mult = if a.mult_charge > 0 { DEPOSIT_MULTIPLIER } else { 1 };
        a.score        += a.gold_carried as u32 * mult;
        a.gold_carried  = 0;
        a.mult_charge   = a.mult_charge.saturating_sub(1); // consume one charge
    }
}
