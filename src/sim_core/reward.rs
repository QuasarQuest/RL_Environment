// src/sim_core/reward.rs

use super::state::AgentState;

pub const PICKUP:  f32 =  0.5;
pub const DEPOSIT: f32 =  5.0;
pub const TICK:    f32 = -0.001;

pub fn compute(agent: &AgentState, prev_gold: u8, prev_score: u32) -> f32 {
    TICK
        + PICKUP  * agent.gold_carried.saturating_sub(prev_gold)           as f32
        + DEPOSIT * agent.score.saturating_sub(prev_score)                 as f32
}
