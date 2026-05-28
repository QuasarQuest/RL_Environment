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
// agent's chosen navigation goal for this tick.
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
//   NavigateToCluster(k) → nearest gold piece in cluster k
//   NavigateToBase       → own base tile
//   NavigateToHealth/Ammo/Enemy → nearest matching item/agent
//   Wait / Attack        → no shaping (returns 0.0)
//
// The goal is resolved in engine/mod.rs::apply_rl_action and passed here
// as Option<GridPos> so the shaping signal is always aligned with the
// action the policy chose — not with a globally-nearest gold that might
// belong to a completely different cluster.
//
// Per-stage weights are passed in via `RewardConfig` (world/config.rs) so
// the same function works for all stages without code changes.

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

/// Potential-based approach shaping toward the agent's chosen navigation goal.
///
/// Returns 0 when there is no nav goal (Wait / Attack actions).
/// Otherwise rewards moving closer to exactly the target the policy committed to,
/// so the shaping signal is always aligned with the action rather than pulling
/// toward an unrelated nearby gold piece.
fn approach_shaping(
    cfg:      &RewardConfig,
    prev_pos: GridPos,
    agent:    &AgentState,
    nav_goal: Option<GridPos>,
) -> f32 {
    let goal = match nav_goal {
        Some(g) => g,
        None    => return 0.0,
    };
    let d_before = euclidean(prev_pos,   goal);
    let d_after  = euclidean(agent.pos,  goal);
    cfg.approach * (d_before - d_after)
}

/// Full per-step reward using stage-specific weights.
pub fn compute(
    cfg:        &RewardConfig,
    agent:      &AgentState,
    prev_pos:   GridPos,
    prev_gold:  u8,
    prev_score: u32,
    prev_kills: u32,
    nav_goal:   Option<GridPos>,
    wall_hit:   bool,
    just_died:  bool,
) -> f32 {
    cfg.tick
        + cfg.pickup       * agent.gold_carried.saturating_sub(prev_gold) as f32
        + cfg.deposit      * agent.score.saturating_sub(prev_score)       as f32
        + cfg.kill         * agent.kills.saturating_sub(prev_kills)       as f32
        + if wall_hit  { cfg.wall_hit      } else { 0.0 }
        + if just_died { cfg.death_penalty } else { 0.0 }
        // No approach shaping while dead or after a death — agent can't act anyway.
        + if just_died || agent.respawn_timer > 0 { 0.0 }
          else { approach_shaping(cfg, prev_pos, agent, nav_goal) }
}