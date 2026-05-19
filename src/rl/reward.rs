// src/rl/reward.rs
//
// Computes the per-tick scalar reward for the RL agent.
//
// Design:
//   +score_delta     gold delivery and kills already flow through Score,
//                    so one delta captures both without separate tracking.
//   -death_penalty   discourages dying — separate from score loss because
//                    dying has strategic cost beyond losing gold.
//   +survival_bonus  tiny per-tick reward to encourage staying alive and
//                    acting rather than waiting in a corner.
//
// Keep shaping minimal for now — get the pipeline working first,
// tune magnitudes once training is confirmed to run end-to-end.

use bevy::prelude::*;
use crate::agent::components::{RespawnIn, Score};
use super::marker::RlAgent;

// ── Reward constants ──────────────────────────────────────────────────────────

/// Awarded per point of score gained this tick (gold delivery = +1 per gold,
/// kill = +kill_reward from MapConfig). Keep at 1.0 — score is already scaled.
const SCORE_SCALE:      f32 = 1.0;

/// Penalty on death. Tune upward if agent plays recklessly.
const DEATH_PENALTY:    f32 = -5.0;

/// Tiny per-tick bonus for being alive. Encourages action over hiding.
const SURVIVAL_BONUS:   f32 = 0.01;

// ── State carried between ticks ───────────────────────────────────────────────

/// Snapshot of agent state from the previous tick.
/// RlEnv holds one of these and passes it into compute_reward each tick.
#[derive(Default, Clone, Copy)]
pub struct PrevAgentState {
    pub score:    u32,
    pub was_dead: bool,
}

impl PrevAgentState {
    pub fn read(world: &mut World) -> Self {
        let mut q = world.query_filtered::<(&Score, Has<RespawnIn>), With<RlAgent>>();
        match q.single(world) {
            Ok((score, is_dead)) => Self { score: score.0, was_dead: is_dead },
            Err(_)               => Self::default(),
        }
    }
}

// ── Reward ────────────────────────────────────────────────────────────────────

/// Compute the reward for the tick that just ran.
/// Call after `app.update()`, before updating `prev`.
pub fn compute_reward(world: &mut World, prev: &PrevAgentState) -> f32 {
    let mut q = world.query_filtered::<(&Score, Has<RespawnIn>), With<RlAgent>>();
    let Ok((score, is_dead)) = q.single(world) else { return 0.0 };

    let mut reward = 0.0;

    // Score delta — captures gold delivery and kills
    let score_delta = score.0.saturating_sub(prev.score) as f32;
    reward += score_delta * SCORE_SCALE;

    // Death penalty — fires on the tick the agent transitions to dead
    if is_dead && !prev.was_dead {
        reward += DEATH_PENALTY;
    }

    // Survival bonus — only while alive
    if !is_dead {
        reward += SURVIVAL_BONUS;
    }

    reward
}