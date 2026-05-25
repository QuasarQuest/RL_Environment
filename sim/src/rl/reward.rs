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
// Approach shaping (potential-based, Ng et al. 1999)
// ---------------------------------------------------
// Dense guidance proportional to the change in Euclidean distance to the
// nearest reachable target (gold when empty, own base when carrying).
//
//   Δdist = dist_before − dist_after
//   positive → moved closer  → positive shaping
//   negative → moved away    → negative shaping
//
// WHY EUCLIDEAN (not Manhattan):
//   Manhattan treats Move NE as Δdist=2, but the agent only travels √2≈1.414
//   tiles — a 41% overestimate. This gives diagonals an inflated reward signal
//   that biases the policy toward zigzag paths for the wrong reason.
//   Euclidean is physically correct: diagonal travel is rewarded proportionally
//   to the actual distance covered.
//
// Scale check: max approach per step = approach × √2 ≈ 0.05 × 1.414 = 0.071
// deposit = 5.0, so approach is ≤ 1.4% of a deposit — safe.
//
// Target selection
// ----------------
//   gold_carried < MAX  →  nearest Gold item (Euclidean distance)
//   gold_carried == MAX →  own base tile (Euclidean distance)
//
// When no gold items remain the approach term is suppressed (0.0).
//
// Per-stage weights are passed in via `RewardConfig` (world/config.rs) so
// the same function works for all stages without code changes.

use crate::config::AGENT_MAX_GOLD;
use crate::world::coords::GridPos;
use crate::world::config::RewardConfig;
use crate::entity::agent::AgentState;

/// Euclidean distance between two grid positions.
/// Physically correct: diagonal movement covers √2 ≈ 1.414 tiles, not 2.
#[inline]
fn euclidean(a: GridPos, b: GridPos) -> f32 {
    let dx = (a.x - b.x) as f32;
    let dy = (a.y - b.y) as f32;
    (dx * dx + dy * dy).sqrt()
}

fn dist_to_nearest_gold(pos: GridPos, gold_positions: &[GridPos]) -> Option<f32> {
    gold_positions
        .iter()
        .map(|&g| euclidean(pos, g))
        .reduce(f32::min)
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
        cfg.approach * (d_before - d_after)
    } else {
        // Inventory full — guide toward base unambiguously.
        let d_before = euclidean(prev_pos,  agent.base_pos);
        let d_after  = euclidean(agent.pos, agent.base_pos);
        cfg.approach * (d_before - d_after)
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
    wall_hit:       bool,
) -> f32 {
    cfg.tick
        + cfg.pickup  * agent.gold_carried.saturating_sub(prev_gold) as f32
        + cfg.deposit * agent.score.saturating_sub(prev_score)       as f32
        + approach_shaping(cfg, prev_pos, agent, gold_positions)
        + if wall_hit { cfg.wall_hit } else { 0.0 }
}