// src/rl/reward.rs
//
// Reward shaping with potential-based guidance for both gold collection
// and delivery.
//
// Reward components:
//   +GOLD_PICKUP          on each unit of gold picked up
//   +DELIVERY_SCALE       on gold delivery (score delta, non-kill)
//   +KILL_SCALE           on kill (score delta >= KILL_THRESHOLD)
//   -DEATH_PENALTY        on death transition (alive → dead)
//   -SURVIVAL_PENALTY     per tick alive (forces urgency)
//
// Potential-based shaping (Ng et al. 1999 — policy-invariant):
//
//   Phase A — NOT carrying gold:
//     Φ_A(s) = -dist_to_nearest_gold * GOLD_APPROACH_SCALE
//     Guides agent toward gold.
//
//   Phase B — carrying gold:
//     Φ_B(s) = -dist_to_base * POTENTIAL_SCALE
//     Guides agent back to base for delivery.
//
//   shaping = γΦ(s') - Φ(s)
//   Both phases are mutually exclusive — no double-counting.

use bevy::prelude::*;
use crate::agent::components::{GoldCarried, RespawnIn, Score};
use crate::item::{Item, ItemKind};
use crate::world::Grid;
use crate::world::coords::GridPos;
use crate::world::tile::Tile;
use super::marker::RlAgent;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Reward per unit of gold picked up.
const GOLD_PICKUP: f32 = 2.0;

/// Reward multiplier for score delta caused by gold delivery.
const DELIVERY_SCALE: f32 = 20.0;

/// Reward multiplier for score delta caused by kills.
const KILL_SCALE: f32 = 0.5;

/// Score delta threshold to distinguish kill from delivery.
const KILL_THRESHOLD: u32 = 10;

/// Penalty applied on the tick the agent transitions from alive to dead.
const DEATH_PENALTY: f32 = -5.0;

/// Small per-tick penalty while alive to encourage urgency.
/// At 10k ticks/episode this totals -50 baseline — forces the agent to act.
const SURVIVAL_PENALTY: f32 = -0.005;

/// Discount factor for potential-based shaping.
/// Must match PPO gamma (0.99) for theoretical policy-invariance guarantees.
const GAMMA: f32 = 0.99;

/// Scale on the carry-to-base potential.
/// Chebyshev dist on a ~50×50 grid → each step closer ≈ +GAMMA*POTENTIAL_SCALE.
const POTENTIAL_SCALE: f32 = 2.0;

/// Scale on the approach-to-gold potential (Phase A).
/// Lower than POTENTIAL_SCALE so delivery still dominates approach.
const GOLD_APPROACH_SCALE: f32 = 0.8;

// ── Prev state ────────────────────────────────────────────────────────────────

/// Snapshot of agent state at the previous tick, used to compute deltas.
#[derive(Default, Clone, Copy)]
pub struct PrevAgentState {
    pub score:         u32,
    pub gold_carried:  u8,
    pub was_dead:      bool,
    /// Chebyshev distance to own base tile. u32::MAX when dead or base absent.
    pub dist_to_base:  u32,
    /// Chebyshev distance to nearest gold item. u32::MAX when dead or no gold.
    pub dist_to_gold:  u32,
}

impl PrevAgentState {
    pub fn read(world: &mut World) -> Self {
        // ── Query agent snapshot ───────────────────────────────────────────
        let snap = {
            let mut q = world.query_filtered::<
                (&Score, &GoldCarried, &GridPos, &crate::team::Team, Has<RespawnIn>),
                With<RlAgent>,
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

        let dist_to_gold = if is_dead || gold > 0 {
            // Phase B (carrying) — gold approach shaping inactive.
            u32::MAX
        } else {
            find_nearest_item_dist(world, pos, ItemKind::Gold)
        };

        Self {
            score,
            gold_carried: gold,
            was_dead: is_dead,
            dist_to_base,
            dist_to_gold,
        }
    }
}

// ── Reward ────────────────────────────────────────────────────────────────────

pub fn compute_reward(world: &mut World, prev: &PrevAgentState) -> f32 {
    // ── Query current agent state ──────────────────────────────────────────
    let snap = {
        let mut q = world.query_filtered::<
            (&Score, &GoldCarried, &GridPos, &crate::team::Team, Has<RespawnIn>),
            With<RlAgent>,
        >();
        q.single(world).ok().map(|(score, gold, pos, team, dead)| {
            (score.0, gold.0, *pos, team.0, dead)
        })
    };

    let Some((score, gold, pos, team_id, is_dead)) = snap else {
        return 0.0;
    };

    let mut reward = 0.0;

    // ── Gold pickup ────────────────────────────────────────────────────────
    if gold > prev.gold_carried {
        reward += (gold - prev.gold_carried) as f32 * GOLD_PICKUP;
    }

    // ── Score delta: delivery or kill ──────────────────────────────────────
    let score_delta = score.saturating_sub(prev.score);
    if score_delta > 0 {
        if score_delta >= KILL_THRESHOLD {
            reward += score_delta as f32 * KILL_SCALE;
        } else {
            reward += score_delta as f32 * DELIVERY_SCALE;
        }
    }

    // ── Death penalty ──────────────────────────────────────────────────────
    if is_dead && !prev.was_dead {
        reward += DEATH_PENALTY;
    }

    // ── Survival penalty ───────────────────────────────────────────────────
    if !is_dead {
        reward += SURVIVAL_PENALTY;
    }

    // ── Potential-based shaping ────────────────────────────────────────────
    // Only active when alive both this tick and last tick.
    if !is_dead && !prev.was_dead {
        if gold > 0 {
            // Phase B: carrying gold → shape toward base.
            let curr_dist = find_base_dist(world, pos, team_id);
            if curr_dist != u32::MAX && prev.dist_to_base != u32::MAX {
                let phi_prev = -(prev.dist_to_base as f32) * POTENTIAL_SCALE;
                let phi_curr = -(curr_dist as f32) * POTENTIAL_SCALE;
                reward += GAMMA * phi_curr - phi_prev;
            }
        } else {
            // Phase A: not carrying → shape toward nearest gold.
            let curr_dist_gold = find_nearest_item_dist(world, pos, ItemKind::Gold);
            if curr_dist_gold != u32::MAX && prev.dist_to_gold != u32::MAX {
                let phi_prev = -(prev.dist_to_gold as f32) * GOLD_APPROACH_SCALE;
                let phi_curr = -(curr_dist_gold as f32) * GOLD_APPROACH_SCALE;
                reward += GAMMA * phi_curr - phi_prev;
            }
        }
    }

    reward
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Chebyshev distance from `pos` to the nearest Base tile owned by `team_id`.
/// Returns u32::MAX when no base tile exists.
fn find_base_dist(world: &World, pos: GridPos, team_id: u8) -> u32 {
    let grid = world.resource::<Grid>();
    grid.iter()
        .filter(|(_, _, tile)| *tile == Tile::Base(team_id))
        .map(|(x, y, _)| chebyshev_raw(x as i32, y as i32, pos))
        .min()
        .unwrap_or(u32::MAX)
}

/// Chebyshev distance from `pos` to the nearest item of `kind` on the map.
/// Returns u32::MAX when no such item exists.
fn find_nearest_item_dist(world: &mut World, pos: GridPos, kind: ItemKind) -> u32 {
    let mut q = world.query::<(&GridPos, &Item)>();
    q.iter(world)
        .filter(|(_, item)| item.kind == kind)
        .map(|(ipos, _)| chebyshev_raw(ipos.x, ipos.y, pos))
        .min()
        .unwrap_or(u32::MAX)
}

#[inline]
fn chebyshev_raw(x: i32, y: i32, origin: GridPos) -> u32 {
    ((x - origin.x).abs().max((y - origin.y).abs())) as u32
}