// src/sim_core/reward.rs
//
// Reward shaping for Stage 1 (gold collection).
//
// Base signals
// ------------
//   TICK     : small negative every step → agent prefers shorter paths
//   PICKUP   : reward on each gold pickup
//   DEPOSIT  : large reward on depositing gold at base
//
// Approach shaping
// ----------------
// A dense guidance signal proportional to the change in Manhattan distance
// to the nearest reachable target (gold when empty, base when carrying).
//
// Δdist = dist_before − dist_after
//   positive → agent moved closer    → positive shaping
//   negative → agent moved away      → negative shaping
//   zero     → agent didn't move     → no shaping
//
// APPROACH increased from 0.01 → 0.05:
//   With 26 actions and only 8 being Move, the expected approach signal per
//   step under a random policy was tiny (~0.01 * 1 * 8/26 ≈ 0.003).  That's
//   barely above the tick penalty noise, giving the policy almost no gradient
//   to climb early in training.  0.05 gives a clearer signal (0.015 expected)
//   while still satisfying potential-based shaping and not dominating DEPOSIT.
//
// TICK softened from -0.001 → -0.0005:
//   The old tick penalty was large enough relative to approach that standing
//   still (reward = -0.001) was as bad as moving away (reward ≈ -0.001-0.01).
//   Softening separates the two signals so the policy can distinguish "did
//   nothing" from "actively moved away".
//
// Scale check (still potential-based):
//   F = γ·Φ(s') − Φ(s) with Φ(s) = −APPROACH · dist
//   Max approach reward per step = APPROACH * max_speed = 0.05 * 2 = 0.10
//   DEPOSIT = 5.0, so approach shaping is at most 2% of a deposit — safe.
//
// Target selection
// ----------------
//   gold_carried == 0  →  nearest Gold item (Manhattan distance)
//   gold_carried  > 0  →  own base tile (agent.base_pos, not spawn_pos)
//
// When no gold items remain the approach term is suppressed (0.0).

use crate::world::coords::GridPos;
use super::state::AgentState;

pub const PICKUP:   f32 =  0.5;
pub const DEPOSIT:  f32 =  5.0;
pub const TICK:     f32 = -0.0005;
pub const APPROACH: f32 =  0.05;

/// Manhattan distance between two grid positions.
#[inline]
fn manhattan(a: GridPos, b: GridPos) -> i32 {
    (a.x - b.x).abs() + (a.y - b.y).abs()
}

/// Distance from `pos` to the nearest gold position.
/// Returns `None` when no gold remains — suppresses approach shaping.
fn dist_to_nearest_gold(pos: GridPos, gold_positions: &[GridPos]) -> Option<i32> {
    gold_positions.iter().map(|&g| manhattan(pos, g)).min()
}

/// Compute the approach shaping term.
///
/// `prev_pos`       — agent position before this tick's action.
/// `agent`          — agent state after action + pickup + deposit.
/// `gold_positions` — pre-filtered gold positions from SimCore.
pub fn approach_shaping(
    prev_pos:       GridPos,
    agent:          &AgentState,
    gold_positions: &[GridPos],
) -> f32 {
    if agent.gold_carried == 0 {
        let d_before = match dist_to_nearest_gold(prev_pos, gold_positions) {
            Some(d) => d,
            None    => return 0.0,
        };
        let d_after = match dist_to_nearest_gold(agent.pos, gold_positions) {
            Some(d) => d,
            None    => return 0.0,
        };
        APPROACH * (d_before - d_after) as f32
    } else {
        // Use base_pos instead of spawn_pos — semantically correct and
        // future-proof for stages where spawn ≠ base.
        let d_before = manhattan(prev_pos, agent.base_pos);
        let d_after  = manhattan(agent.pos, agent.base_pos);
        APPROACH * (d_before - d_after) as f32
    }
}

/// Full per-step reward.
pub fn compute(
    agent:          &AgentState,
    prev_pos:       GridPos,
    prev_gold:      u8,
    prev_score:     u32,
    gold_positions: &[GridPos],
) -> f32 {
    TICK
        + PICKUP  * agent.gold_carried.saturating_sub(prev_gold)  as f32
        + DEPOSIT * agent.score.saturating_sub(prev_score)        as f32
        + approach_shaping(prev_pos, agent, gold_positions)
}