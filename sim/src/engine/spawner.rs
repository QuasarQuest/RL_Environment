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

use crate::entity::agent::AgentState;
use crate::entity::item::{ItemKind, ItemState};
use crate::world::{coords::GridPos, grid::Grid, tile::Tile};

pub struct SpawnBudget {
    pub kind:       ItemKind,
    pub spawn_prob: f32,
    pub target:     usize,
}

/// Called every tick from SimCore::step. May push new items into `items`.
pub fn tick_spawns(
    items:   &mut Vec<ItemState>,
    agents:  &[AgentState],
    grid:    &Grid,
    budgets: &[SpawnBudget],
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
        // Excludes tiles occupied by existing items or agents.
        let mut candidates: Vec<GridPos> = (0..grid.height as i32)
            .flat_map(|y| (0..grid.width as i32).map(move |x| GridPos::new(x, y)))
            .filter(|p| {
                grid.get(p.x, p.y) == Some(Tile::Free)
                    && !items.iter().any(|i| i.pos == *p)
                    && !agents.iter().any(|a| a.pos == *p)
            })
            .collect();

        candidates.shuffle(rng);
        for &pos in candidates.iter().take(n_spawn) {
            items.push(ItemState { pos, kind: budget.kind });
        }
    }
}
