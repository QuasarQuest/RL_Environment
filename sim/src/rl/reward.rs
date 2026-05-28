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
// Approach shaping (true potential-based shaping, Ng et al. 1999)
// ---------------------------------------------------------------
// Dense guidance from a STATE-ONLY potential Φ(s) = −dist(agent, objective(s)):
//
//   F = approach · (γ·Φ(s') − Φ(s))
//     = approach · (dist_before − γ·dist_after)
//
// WHY STATE-ONLY (not the action's chosen goal):
//   PBRS is policy-invariant only when Φ depends on the STATE alone. The old
//   version keyed the potential off whichever nav goal the action picked this
//   tick. Because A* always steps toward the chosen goal, every navigation
//   action earned ≈ +approach unconditionally — so (a) the agent could "farm"
//   shaping by pacing between two goals, and (b) Wait/Attack (which earn 0)
//   were structurally suppressed even when attacking was correct. Defining the
//   objective from state removes both: the target is the same regardless of the
//   action chosen, so over any closed loop the shaping telescopes to ~0.
//
//   objective(s) = base            if carrying gold   (go deposit)
//                = nearest gold     otherwise          (go collect)
//   Resolved in engine/mod.rs::step from world state and passed here as
//   Option<GridPos>; None (no gold on map and not carrying) → no shaping.
//
// WHY EUCLIDEAN (not Manhattan):
//   Manhattan treats Move NE as Δdist=2, but the agent only travels √2≈1.414
//   tiles — a 41% overestimate that biases the policy toward zigzag paths.
//   Euclidean rewards diagonal travel proportionally to distance covered.
//
// WHY γ:
//   γ·Φ(s') − Φ(s) is the exact PBRS form; γ must match the PPO discount
//   (RewardConfig::shaping_gamma) for the shaping to leave the optimal policy
//   unchanged. Omitting it (the old code) makes the shaping a small but
//   non-invariant bias.
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

/// True potential-based approach shaping toward a state-defined objective.
///
/// `objective` is resolved from world state (base if carrying, else nearest
/// gold) — NOT from the action — so the shaping is policy-invariant and cannot
/// be farmed by switching goals each tick. Returns 0 when there is no objective
/// (no gold on the map and nothing being carried).
///
/// F = approach · (γ·Φ(s') − Φ(s)) with Φ(s) = −dist(agent, objective).
fn approach_shaping(
    cfg:       &RewardConfig,
    prev_pos:  GridPos,
    agent:     &AgentState,
    objective: Option<GridPos>,
) -> f32 {
    let goal = match objective {
        Some(g) => g,
        None    => return 0.0,
    };
    let d_before = euclidean(prev_pos,  goal);
    let d_after  = euclidean(agent.pos, goal);
    cfg.approach * (d_before - cfg.shaping_gamma * d_after)
}

/// Full per-step reward using stage-specific weights.
pub fn compute(
    cfg:        &RewardConfig,
    agent:      &AgentState,
    prev_pos:   GridPos,
    prev_gold:  u8,
    prev_score: u32,
    prev_kills: u32,
    objective:  Option<GridPos>,
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
          else { approach_shaping(cfg, prev_pos, agent, objective) }
}