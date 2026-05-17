// src/agent/strategy/mod.rs

pub mod bt;
pub mod fsm;
pub mod goap;
pub mod random;
mod rl;

pub use bt::BtStrategy;
pub use fsm::FsmStrategy;
pub use goap::GoapStrategy;
pub use random::RandomStrategy;

use serde::Deserialize;
use crate::agent::action::{Action, Dir};
use crate::agent::components::GridPos;
use crate::agent::debug::DebugDraw;
use crate::agent::observation::Observation;
use crate::agent::planner::PathPlanner;
use crate::algorithm::path_planning::graph_utils::dir_to;
use crate::world::tile::Tile;

// ── Trait ─────────────────────────────────────────────────────────────────────

pub trait DecisionStrategy: Send + Sync {
    fn name(&self) -> &'static str;
    fn decide(&mut self, obs: &Observation, planner: &mut impl PathPlanner) -> Action;
    /// Strategies that own their own planner (BtStrategy) override this
    /// to expose it. All others return None — Brain falls back to its
    /// outer planner's debug_draw().
    fn debug_draw(&self) -> Option<Box<dyn DebugDraw>> { None }
    fn reset(&mut self) {}
}

// ── Shared nav helper ─────────────────────────────────────────────────────────

pub fn try_move(
    planner:     &mut dyn PathPlanner,
    obs:         &Observation,
    stuck_ticks: &mut u8,
) -> Result<Action, ()> {
    let Some(next) = planner.next_step() else { return Ok(Action::Wait); };
    if !obs.is_walkable(next) {
        *stuck_ticks += 1;
        if *stuck_ticks >= 3 { *stuck_ticks = 0; return Err(()); }
        return Ok(Action::Wait);
    }
    *stuck_ticks = 0;
    Ok(dir_to(obs.pos, next).map(Action::Move).unwrap_or(Action::Wait))
}

// ── Combat helpers ────────────────────────────────────────────────────────────

pub fn adjacent_enemy_dir(obs: &Observation) -> Option<Dir> {
    Dir::all().iter().copied().find(|&dir| {
        let (dx, dy) = dir.delta();
        let check    = GridPos::new(obs.pos.x + dx, obs.pos.y + dy);
        obs.visible_agents().iter()
            .any(|a| a.is_enemy(obs.team) && a.pos == check)
    })
}

pub fn ranged_enemy_dir(obs: &Observation) -> Option<Dir> {
    let mut best_dir:  Option<Dir> = None;
    let mut best_dist: i32         = i32::MAX;
    for &dir in Dir::all() {
        let (dx, dy) = dir.delta();
        for dist in 1..=crate::config::RANGED_RANGE {
            let check = GridPos::new(obs.pos.x + dx * dist, obs.pos.y + dy * dist);
            if obs.grid_tile(check) == Some(Tile::Obstacle) { break; }
            if obs.visible_agents().iter().any(|a| a.is_enemy(obs.team) && a.pos == check) {
                if dist < best_dist { best_dist = dist; best_dir = Some(dir); }
                break;
            }
        }
    }
    best_dir
}

pub fn roam(obs: &Observation) -> Action {
    let dirs: Vec<Dir> = Dir::all().iter().copied()
        .filter(|&d| {
            let (dx, dy) = d.delta();
            obs.is_walkable(GridPos::new(obs.pos.x + dx, obs.pos.y + dy))
        })
        .collect();
    if dirs.is_empty() { Action::Wait }
    else { Action::Move(dirs[rand::random_range(0..dirs.len())]) }
}

// ── StrategyKind ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub enum StrategyKind {
    Fsm,
    BehaviorTree,
    Random,
    Goap,
}