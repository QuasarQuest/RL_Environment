// src/engine/spawner.rs
//
// Probabilistic item spawner.
//
// Each tick, for every missing item slot below `target`, the slot fires
// independently with probability `spawn_prob`. This gives a geometric
// distribution over spawn timing — the agent cannot learn to predict when
// items will appear because each tick is memoryless.
//
// Extendable to multi-kind budgets for stage 2+ (health, ammo, speed) and
// team-balanced gold spawning in stage 3.

use rand::seq::SliceRandom;
use rand::RngExt;
use rand::rngs::SmallRng;
use rustc_hash::FxHashSet;

use crate::entity::agent::AgentState;
use crate::entity::item::{ItemKind, ItemState};
use crate::world::{coords::GridPos, grid::Grid, tile::Tile};

/// Per-tick probability a single missing slot fires. Geometric distribution
/// keeps spawn timing unpredictable so the agent cannot time arrivals.
pub const DEFAULT_SPAWN_PROB: f32 = 0.02;

/// Fraction of the best (farthest) min-distance a candidate must still reach to be in the
/// random pick pool — see `pick_distributed`. Lower ⇒ buffs hug the gold-sparse extremes
/// harder and more predictably; higher ⇒ more spread/randomised placement.
const DISTRIBUTE_TOP_FRACTION: f32 = 0.7;

/// Choose up to `n` tiles from `candidates` spread far from `avoid` and from each other,
/// via randomised farthest-point sampling: each pick maximises the Chebyshev distance to
/// the nearest already-avoided-or-chosen tile, drawn at random from the farthest
/// `DISTRIBUTE_TOP_FRACTION` so placement isn't perfectly predictable. Used for buff
/// items (speed/multiplier) so they land out in the open, gold-sparse regions and stay
/// distributed around the map instead of sitting on top of a gold cluster. Returns fewer
/// than `n` only if candidates run out.
pub fn pick_distributed(
    candidates: &[GridPos], avoid: &[GridPos], n: usize, rng: &mut SmallRng,
) -> Vec<GridPos> {
    let min_dist = |p: GridPos, pts: &[GridPos]| pts.iter()
        .map(|q| (q.x - p.x).abs().max((q.y - p.y).abs()))
        .min()
        .unwrap_or(i32::MAX); // nothing to avoid ⇒ treat as maximally far

    let mut avoid: Vec<GridPos> = avoid.to_vec();
    let mut pool:  Vec<GridPos> = candidates.to_vec();
    let mut chosen: Vec<GridPos> = Vec::with_capacity(n);

    while chosen.len() < n && !pool.is_empty() {
        let scored: Vec<(i32, usize)> = pool.iter().enumerate()
            .map(|(i, &p)| (min_dist(p, &avoid), i))
            .collect();
        let best = scored.iter().map(|(d, _)| *d).max().unwrap_or(0);
        let thresh = (best as f32 * DISTRIBUTE_TOP_FRACTION).floor() as i32;
        let top: Vec<usize> = scored.iter()
            .filter(|(d, _)| *d >= thresh).map(|(_, i)| *i).collect();

        let idx = top[rng.random_range(0..top.len())];
        let p = pool.swap_remove(idx);
        avoid.push(p);
        chosen.push(p);
    }
    chosen
}

pub struct SpawnBudget {
    pub kind:       ItemKind,
    pub spawn_prob: f32,
    pub target:     usize,
    /// When `Some((radius, bias))`, each new item of this kind lands within Chebyshev
    /// `radius` of an existing same-kind item with probability `bias`, and anywhere free
    /// otherwise. This holds the layout at its initial clumped-vs-sparse mix (see
    /// GoldClusterConfig): always clustering would grow clumps into ever-larger blobs,
    /// never clustering would erode them to uniform. Falls back to anywhere free when no
    /// same-kind item remains on the map.
    pub cluster: Option<(i32, f32)>,
}

/// Called every tick from SimCore::step. May push new items into `items`.
/// `blocked` is the spawn pocket around the base — items never spawn there.
pub fn tick_spawns(
    items:   &mut Vec<ItemState>,
    agents:  &[AgentState],
    grid:    &Grid,
    budgets: &[SpawnBudget],
    blocked: &FxHashSet<GridPos>,
    rng:     &mut SmallRng,
) {
    for budget in budgets {
        let current = items.iter().filter(|i| i.kind == budget.kind).count();
        let missing = budget.target.saturating_sub(current);
        if missing == 0 { continue; }

        let n_spawn = (0..missing)
            .filter(|_| rng.random_bool(budget.spawn_prob as f64))
            .count();
        if n_spawn == 0 { continue; }

        // Collect all eligible free tiles once, then shuffle and take N.
        // Excludes tiles occupied by existing items or agents (so nothing ever stacks).
        let mut candidates: Vec<GridPos> = (0..grid.height as i32)
            .flat_map(|y| (0..grid.width as i32).map(move |x| GridPos::new(x, y)))
            .filter(|p| {
                grid.get(p.x, p.y) == Some(Tile::Free)
                    && !blocked.contains(p)
                    && !items.iter().any(|i| i.pos == *p)
                    && !agents.iter().any(|a| a.pos == *p)
            })
            .collect();

        candidates.shuffle(rng);

        match budget.cluster {
            // ── Clustered kind (gold): partition into tiles near an existing same-kind
            // item vs the rest, and pick each spawn near-or-scattered by `bias`. ──
            Some((r, bias)) => {
                let (mut near, mut rest): (Vec<GridPos>, Vec<GridPos>) = (Vec::new(), Vec::new());
                for p in candidates {
                    if items.iter().any(|i| i.kind == budget.kind
                        && (i.pos.x - p.x).abs().max((i.pos.y - p.y).abs()) <= r) {
                        near.push(p);
                    } else {
                        rest.push(p);
                    }
                }
                for _ in 0..n_spawn {
                    let use_near = !near.is_empty()
                        && (rest.is_empty() || rng.random_bool(bias as f64));
                    match if use_near { near.pop() } else { rest.pop() } {
                        Some(pos) => items.push(ItemState { pos, kind: budget.kind }),
                        None => break, // both pools exhausted
                    }
                }
            }
            // ── Unclustered kind (speed/multiplier): distribute far from gold and every
            // other item, out in the open instead of next to a cluster. ──
            None => {
                let avoid: Vec<GridPos> = items.iter().map(|i| i.pos).collect();
                for pos in pick_distributed(&candidates, &avoid, n_spawn, rng) {
                    items.push(ItemState { pos, kind: budget.kind });
                }
            }
        }
    }
}
