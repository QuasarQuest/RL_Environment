// src/rl/reward.rs
//
// Reward shaping for the gold-collection curriculum.
//
// Base signals
// ------------
//   tick         : small negative every step → agent prefers shorter paths
//   pickup       : reward on each gold pickup
//   deposit      : large reward on depositing gold at base
//
// Approach shaping (potential-based)
// ------------------------------------
// Dense guidance proportional to the change in Manhattan distance to the
// nearest reachable target (gold when empty, own base when carrying).
//
//   Δdist = dist_before − dist_after
//   positive → moved closer  → positive shaping
//   negative → moved away    → negative shaping
//
// Scale check: max approach per step = approach * max_speed = 0.05 * 2 = 0.10
// deposit = 5.0, so approach is ≤ 2% of a deposit — safe.
//
// Target selection
// ----------------
//   gold_carried == 0  →  nearest Gold item (Manhattan distance)
//   gold_carried  > 0  →  own base tile (agent.base_pos)
//
// When no gold items remain the approach term is suppressed (0.0).
//
// Per-stage weights are passed in via `RewardConfig` (world/config.rs) so
// the same function works for all stages without code changes.

use crate::config::AGENT_MAX_GOLD;
use crate::world::coords::GridPos;
use crate::world::config::RewardConfig;
use crate::entity::agent::AgentState;

/// Manhattan distance between two grid positions.
#[inline]
fn manhattan(a: GridPos, b: GridPos) -> i32 {
    (a.x - b.x).abs() + (a.y - b.y).abs()
}

fn dist_to_nearest_gold(pos: GridPos, gold_positions: &[GridPos]) -> Option<i32> {
    gold_positions.iter().map(|&g| manhattan(pos, g)).min()
}

fn approach_shaping(
    cfg:            &RewardConfig,
    prev_pos:       GridPos,
    agent:          &AgentState,
    gold_positions: &[GridPos],
) -> f32 {
    if agent.gold_carried < AGENT_MAX_GOLD {
        // Not yet full — guide toward the nearest gold.
        let d_before = match dist_to_nearest_gold(prev_pos, gold_positions) {
            Some(d) => d,
            None    => return 0.0,
        };
        let d_after = match dist_to_nearest_gold(agent.pos, gold_positions) {
            Some(d) => d,
            None    => return 0.0,
        };
        cfg.approach * (d_before - d_after) as f32
    } else {
        // Inventory full — guide toward base unambiguously.
        let d_before = manhattan(prev_pos, agent.base_pos);
        let d_after  = manhattan(agent.pos, agent.base_pos);
        cfg.approach * (d_before - d_after) as f32
    }
}

/// Full per-step reward using stage-specific weights.
pub fn compute(
    cfg:            &RewardConfig,
    agent:          &AgentState,
    prev_pos:       GridPos,
    prev_gold:      u8,
    prev_score:     u32,
    gold_positions: &[GridPos],
) -> f32 {
    cfg.tick
        + cfg.pickup  * agent.gold_carried.saturating_sub(prev_gold) as f32
        + cfg.deposit * agent.score.saturating_sub(prev_score)       as f32
        + approach_shaping(cfg, prev_pos, agent, gold_positions)
}