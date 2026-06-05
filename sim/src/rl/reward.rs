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

/// Per-step reward split into its individual signals. `total()` is the value the
/// sim actually uses; the components are kept separate so the trace recorder can
/// log exactly which signal fired each tick (see SimCore::TickRecord).
#[derive(Clone, Copy, Debug, Default)]
pub struct RewardBreakdown {
    pub tick:    f32,
    pub pickup:  f32,
    pub deposit: f32,
    pub wall:    f32,
}

impl RewardBreakdown {
    /// The summed per-step reward — what the agent actually receives this tick.
    pub fn total(&self) -> f32 {
        self.tick + self.pickup + self.deposit + self.wall
    }
}

/// Full per-step reward using config weights.
/// Event-based only: time cost + gold pickup + gold deposit, with a wall_hit penalty.
pub fn compute(
    cfg:        &RewardConfig,
    agent:      &AgentState,
    prev_gold:  u8,
    prev_score: u32,
    wall_hit:   bool,
) -> f32 {
    compute_components(cfg, agent, prev_gold, prev_score, wall_hit).total()
}

/// Same as [`compute`] but returns each signal separately. The sum equals
/// [`compute`]'s result exactly — `compute` delegates here.
pub fn compute_components(
    cfg:        &RewardConfig,
    agent:      &AgentState,
    prev_gold:  u8,
    prev_score: u32,
    wall_hit:   bool,
) -> RewardBreakdown {
    RewardBreakdown {
        tick:    cfg.tick,
        pickup:  cfg.pickup  * agent.gold_carried.saturating_sub(prev_gold) as f32,
        deposit: cfg.deposit * agent.score.saturating_sub(prev_score)       as f32,
        wall:    if wall_hit { cfg.wall_hit } else { 0.0 },
    }
}
