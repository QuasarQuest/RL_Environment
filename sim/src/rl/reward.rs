// src/rl/reward.rs
//
// Reward shaping for the single-agent gold-collection task.
//
// Signals
// -------
//   tick     : small negative every step → agent prefers shorter paths
//   pickup   : reward on each gold pickup
//   deposit  : large reward on depositing gold at base
//   wall_hit : small penalty when a Move is blocked (no position change)
//
// Reward is intentionally MINIMAL and event-based: the agent learns purely from
// game outcomes — gold picked up, gold banked — plus a tiny per-step time cost
// and the wall_hit penalty. There is deliberately NO approach/navigation shaping
// (the policy discovers where to go from the observation alone). With the
// temporally-extended options in SimCore::step_option each pickup/deposit lands
// at an option boundary, so credit assignment is already short.
//
// Combat (kill / death) was removed — this agent is gold rush only.
//
// Per-task weights come from `RewardConfig` (world/config.rs).

use crate::world::config::RewardConfig;
use crate::entity::agent::AgentState;

/// Full per-step reward using config weights.
/// Event-based only: time cost + gold pickup + gold deposit, with a wall_hit penalty.
pub fn compute(
    cfg:        &RewardConfig,
    agent:      &AgentState,
    prev_gold:  u8,
    prev_score: u32,
    wall_hit:   bool,
) -> f32 {
    cfg.tick
        + cfg.pickup  * agent.gold_carried.saturating_sub(prev_gold) as f32
        + cfg.deposit * agent.score.saturating_sub(prev_score)       as f32
        + if wall_hit { cfg.wall_hit } else { 0.0 }
}
