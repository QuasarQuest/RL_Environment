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
// Reward is intentionally MINIMAL and event-based: the agent learns purely from
// game outcomes — gold picked up, gold banked, kills — plus a tiny per-step time
// cost and the wall_hit/death penalties. There is deliberately NO approach or
// navigation shaping.
//
// An earlier version added potential-based shaping toward the nearest gold / base
// (Φ(s) = −dist(agent, objective)). That is a "go here" hint baked into the reward
// and it required the sim to pick an objective (nearest gold) for the agent every
// tick. Both were removed: the policy must now discover *where* to go from the
// observation alone, with no decision logic in the sim.
//
// Per-stage weights are passed in via `RewardConfig` (world/config.rs) so the same
// function works for all stages. (RewardConfig still carries the now-unused
// `approach`/`shaping_gamma` fields so existing config files keep parsing.)

use crate::world::coords::GridPos;
use crate::world::config::RewardConfig;
use crate::entity::agent::AgentState;

/// Full per-step reward using stage-specific weights.
///
/// Event-based only: time cost + gold pickup + gold deposit + kills, with
/// wall_hit / death penalties. No navigation shaping (see module docs).
pub fn compute(
    cfg:        &RewardConfig,
    agent:      &AgentState,
    _prev_pos:  GridPos,
    prev_gold:  u8,
    prev_score: u32,
    prev_kills: u32,
    wall_hit:   bool,
    just_died:  bool,
) -> f32 {
    cfg.tick
        + cfg.pickup       * agent.gold_carried.saturating_sub(prev_gold) as f32
        + cfg.deposit      * agent.score.saturating_sub(prev_score)       as f32
        + cfg.kill         * agent.kills.saturating_sub(prev_kills)       as f32
        + if wall_hit  { cfg.wall_hit      } else { 0.0 }
        + if just_died { cfg.death_penalty } else { 0.0 }
}