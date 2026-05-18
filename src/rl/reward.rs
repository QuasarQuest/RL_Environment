// src/rl/reward.rs
//
// Reward shaping with potential-based carry guidance.
//
// Reward components:
//   +GOLD_PICKUP          on gold pickup
//   +DELIVERY_SCALE       on gold delivery (score delta)
//   +KILL_SCALE           on kill (score delta >= KILL_THRESHOLD)
//   -DEATH_PENALTY        on death transition
//   -SURVIVAL_PENALTY     per tick alive (forces action)
//   +potential shaping    γΦ(s') - Φ(s) while carrying gold
//                         Φ(s) = -dist_to_base (Chebyshev)
//                         Getting closer to base = positive reward every step.
//                         Guaranteed policy-invariant by Ng et al. 1999.

use bevy::prelude::*;
use crate::agent::components::{GoldCarried, RespawnIn, Score};
use crate::world::Grid;
use crate::world::coords::GridPos;
use crate::world::tile::Tile;
use super::marker::RlAgent;

// ── Constants ─────────────────────────────────────────────────────────────────

const GOLD_PICKUP:      f32 =  2.0;
const DELIVERY_SCALE:   f32 = 10.0;
const KILL_SCALE:       f32 =  0.5;
const KILL_THRESHOLD:   u32 = 10;
const DEATH_PENALTY:    f32 = -5.0;
const SURVIVAL_PENALTY: f32 = -0.001;

/// Discount factor for potential-based shaping.
/// Must match the PPO gamma (0.99) for theoretical guarantees.
const GAMMA:            f32 = 0.99;

/// Scale factor on the potential function.
/// Φ(s) = -dist_to_base / MAP_DIAGONAL * POTENTIAL_SCALE
/// With MAP_DIAGONAL ≈ 70 (50×50 grid), each step toward base ≈ +0.014
const POTENTIAL_SCALE:  f32 = 1.0;

// ── State ─────────────────────────────────────────────────────────────────────

#[derive(Default, Clone, Copy)]
pub struct PrevAgentState {
    pub score:           u32,
    pub gold_carried:    u8,
    pub was_dead:        bool,
    /// Chebyshev distance to own base. u32::MAX when dead or base not found.
    pub dist_to_base:    u32,
}

impl PrevAgentState {
    pub fn read(world: &mut World) -> Self {
        // Query agent
        let snap = {
            let mut q = world.query_filtered::<
                (&Score, &GoldCarried, &GridPos, &crate::team::Team, Has<RespawnIn>),
                With<RlAgent>
            >();
            q.single(world).ok().map(|(score, gold, pos, team, dead)| {
                (score.0, gold.0, *pos, team.0, dead)
            })
        };

        let Some((score, gold, pos, team_id, is_dead)) = snap else {
            return Self::default();
        };

        let dist_to_base = if is_dead {
            u32::MAX
        } else {
            find_base_dist(world, pos, team_id)
        };

        Self { score, gold_carried: gold, was_dead: is_dead, dist_to_base }
    }
}

// ── Reward ────────────────────────────────────────────────────────────────────

pub fn compute_reward(world: &mut World, prev: &PrevAgentState) -> f32 {
    let snap = {
        let mut q = world.query_filtered::<
            (&Score, &GoldCarried, &GridPos, &crate::team::Team, Has<RespawnIn>),
            With<RlAgent>
        >();
        q.single(world).ok().map(|(score, gold, pos, team, dead)| {
            (score.0, gold.0, *pos, team.0, dead)
        })
    };

    let Some((score, gold, pos, team_id, is_dead)) = snap else { return 0.0 };

    let mut reward = 0.0;

    // Gold pickup
    if gold > prev.gold_carried {
        reward += (gold - prev.gold_carried) as f32 * GOLD_PICKUP;
    }

    // Score delta — delivery or kill
    let score_delta = score.saturating_sub(prev.score);
    if score_delta > 0 {
        if score_delta >= KILL_THRESHOLD {
            reward += score_delta as f32 * KILL_SCALE;
        } else {
            reward += score_delta as f32 * DELIVERY_SCALE;
        }
    }

    // Death penalty
    if is_dead && !prev.was_dead {
        reward += DEATH_PENALTY;
    }

    // Survival penalty
    if !is_dead {
        reward += SURVIVAL_PENALTY;
    }

    // ── Potential-based carry shaping ─────────────────────────────────────────
    // Only active while carrying gold and alive both ticks.
    // Φ(s) = -dist_to_base * POTENTIAL_SCALE
    // shaping = γΦ(s') - Φ(s)
    //         = γ(-d') - (-d)
    //         = d - γd'
    // If d' < d (closer): positive. If d' > d (further): negative.
    if gold > 0 && !is_dead && !prev.was_dead {
        let curr_dist = find_base_dist(world, pos, team_id);
        if curr_dist != u32::MAX && prev.dist_to_base != u32::MAX {
            let phi_prev = -(prev.dist_to_base as f32) * POTENTIAL_SCALE;
            let phi_curr = -(curr_dist         as f32) * POTENTIAL_SCALE;
            reward += GAMMA * phi_curr - phi_prev;
        }
    }

    reward
}

// ── Helper ────────────────────────────────────────────────────────────────────

fn find_base_dist(world: &World, pos: GridPos, team_id: u8) -> u32 {
    let grid = world.resource::<Grid>();
    grid.iter()
        .filter(|(_, _, tile)| *tile == Tile::Base(team_id))
        .map(|(x, y, _)| {
            let dx = (x as i32 - pos.x).abs();
            let dy = (y as i32 - pos.y).abs();
            dx.max(dy) as u32
        })
        .min()
        .unwrap_or(u32::MAX)
}