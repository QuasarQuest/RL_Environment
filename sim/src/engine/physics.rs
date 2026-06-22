// src/engine/physics.rs
//
// Agent movement, buff ticking and deposit. Single-agent gold rush — no combat.

use crate::config::DEPOSIT_MULTIPLIER;
use crate::entity::agent::{Action, AgentState};
use crate::world::{coords::GridPos, grid::Grid, tile::Tile};

// ── Per-tick passive buff decay ─────────────────────────────────────────────────

/// Decrement every active buff timer by one tick. (The multiplier is a held charge,
/// not a timer, so it does not decay here — it is consumed on deposit.)
pub fn tick_buffs(agents: &mut [AgentState]) {
    for a in agents {
        if a.speed_buff > 0 { a.speed_buff -= 1; }
    }
}

// ── Movement ──────────────────────────────────────────────────────────────────

/// Move one tile. Returns `true` only when the step was blocked by a wall (agent
/// stepped straight into it, no progress) — the only case that incurs a wall penalty;
/// `Wait` and already-on-goal do not. Movement is one tile/tick (the move-energy
/// cadence upstream decides whether to step), so a move can't overshoot a turn.
pub fn apply_action(
    agents:  &mut Vec<AgentState>,
    grid:    &Grid,
    idx:     usize,
    action:  Action,
    stop_at: Option<GridPos>,
) -> bool {
    let Action::Move(dir) = action else { return false };
    if stop_at == Some(agents[idx].pos) { return false; }
    let (dx, dy) = dir.delta();
    let next = agents[idx].pos.apply_delta(dx, dy);
    if !grid.is_walkable(next.x, next.y) { return true; }
    agents[idx].pos = next;
    false
}

// ── Deposit ───────────────────────────────────────────────────────────────────

/// Auto-deposit — runs every tick for the agent standing on its base. Banks all
/// carried gold; a held multiplier charge doubles this deposit's value and is then
/// consumed (one charge per deposit).
pub fn auto_deposit(agents: &mut Vec<AgentState>, grid: &Grid) {
    for a in agents {
        if a.gold_carried == 0 { continue; }
        if !matches!(grid.get(a.pos.x, a.pos.y), Some(Tile::Base(_))) { continue; }
        let mult = if a.mult_charge > 0 { DEPOSIT_MULTIPLIER } else { 1 };
        a.score        += a.gold_carried as u32 * mult;
        a.gold_carried  = 0;
        a.mult_charge   = a.mult_charge.saturating_sub(1); // consume one charge
    }
}
