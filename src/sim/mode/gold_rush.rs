// src/sim/mode/gold_rush.rs
//
// GoldRush — collect gold, deliver to base, first team to WIN_SCORE wins.
//
// Reward shaping:
//   +SCORE_SCALE per point of score gained (gold delivery)
//   +SURVIVAL_BONUS per tick alive
//   +DEATH_PENALTY on the tick the agent dies

use bevy::prelude::*;
use crate::agent::components::{RespawnIn, Score};
use crate::team::{Team, TeamScore};
use super::{GameMode, PrevModeState};

// ── Constants ─────────────────────────────────────────────────────────────────

const SCORE_SCALE:    f32 = 1.0;
const DEATH_PENALTY:  f32 = -5.0;
const SURVIVAL_BONUS: f32 = 0.01;
const WIN_SCORE:      u32 = 10;

// ── GoldRush ──────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct GoldRush;

impl GameMode for GoldRush {
    fn reward(&self, world: &mut World, prev: &PrevModeState) -> f32 {
        let mut q = world.query::<(&Score, Has<RespawnIn>)>();
        let Ok((score, is_dead)) = q.single(world) else { return 0.0 };

        let mut reward = 0.0;

        let score_delta = score.0.saturating_sub(prev.score) as f32;
        reward += score_delta * SCORE_SCALE;

        if is_dead && !prev.was_dead {
            reward += DEATH_PENALTY;
        }

        if !is_dead {
            reward += SURVIVAL_BONUS;
        }

        reward
    }

    fn is_terminal(&self, world: &mut World) -> bool {
        let scores = world.resource::<TeamScore>();
        (0u8..=3).any(|t| scores.get(Team(t)) >= WIN_SCORE)
    }

    fn on_reset(&self, _world: &mut World) {}

    fn snapshot(&self, world: &mut World) -> PrevModeState {
        let mut q = world.query::<(&Score, Has<RespawnIn>)>();
        match q.single(world) {
            Ok((score, is_dead)) => PrevModeState {
                score:    score.0,
                was_dead: is_dead,
                gold:     0,
            },
            Err(_) => PrevModeState::default(),
        }
    }
}