// src/rl/reward.rs
//
// Minimal reward for the gold-collection task. No combat, no death, no
// survival bonus — just enough signal to learn "find gold, bring it back".
//
// Reward formula (per tick):
//   +PICKUP_REWARD * (gold_carried_now - gold_carried_prev)   if positive
//   +DELIVERY_REWARD * (score_now - score_prev)                if positive
//   +TICK_PENALTY                                              always (negative)
//
// Why two separate deltas rather than just score-delta:
//   Score only fires on delivery. The agent needs an earlier reinforcement
//   signal for the act of picking gold up at all, otherwise the gradient
//   from "do nothing" to "carry one gold" is zero.

use bevy::prelude::*;
use crate::agent::components::{GoldCarried, Score};
use super::marker::RlAgent;

// ── Reward constants ──────────────────────────────────────────────────────────

/// Per gold picked up (gold_carried increase from one tick to the next).
const PICKUP_REWARD:   f32 =  0.5;

/// Per unit of score gained. With no combat, score == gold delivered, so
/// this is effectively per-gold-delivered reward.
const DELIVERY_REWARD: f32 =  5.0;

/// Small constant time pressure — discourages standing still forever.
const TICK_PENALTY:    f32 = -0.001;

// ── Prev-state snapshot ───────────────────────────────────────────────────────

#[derive(Default, Clone, Copy)]
pub struct PrevState {
    pub gold_carried: u8,
    pub score:        u32,
}

impl PrevState {
    pub fn read(world: &mut World) -> Self {
        let mut q = world.query_filtered::<
            (&GoldCarried, &Score),
            With<RlAgent>,
        >();
        match q.single(world) {
            Ok((gold, score)) => Self { gold_carried: gold.0, score: score.0 },
            Err(_)            => Self::default(),
        }
    }
}

// ── Reward ────────────────────────────────────────────────────────────────────

/// Compute the reward for the tick that just ran.
/// Call AFTER `app.update()` / `OnSimTick`, BEFORE refreshing `prev`.
pub fn compute_reward(world: &mut World, prev: &PrevState) -> f32 {
    let mut q = world.query_filtered::<
        (&GoldCarried, &Score),
        With<RlAgent>,
    >();
    let Ok((gold, score)) = q.single(world) else { return 0.0 };

    let mut reward = TICK_PENALTY;

    // Pickup: gold_carried went up.
    // Note: gold_carried also drops when the agent dies (gold scatters) or
    // when it delivers (score goes up, gold_carried resets to 0). Using
    // saturating_sub means those events contribute 0 to pickup_delta, which
    // is the correct behaviour.
    let pickup_delta = gold.0.saturating_sub(prev.gold_carried) as f32;
    reward += PICKUP_REWARD * pickup_delta;

    // Delivery: score went up. In this task that means gold was deposited.
    let score_delta = score.0.saturating_sub(prev.score) as f32;
    reward += DELIVERY_REWARD * score_delta;

    reward
}