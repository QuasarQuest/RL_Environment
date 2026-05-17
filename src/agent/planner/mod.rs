// src/agent/planner/mod.rs

pub mod a_star;
pub mod d_star;
pub mod none;

pub use a_star::AStarPlanner;
pub use d_star::DStarPlanner;
pub use none::NoPlanner;

use serde::Deserialize;
use crate::agent::components::GridPos;
use crate::agent::debug::DebugDraw;

// ── Trait ─────────────────────────────────────────────────────────────────────

pub trait PathPlanner: Send + Sync {
    fn set_goal(&mut self, start: GridPos, goal: GridPos, is_walkable: &dyn Fn(GridPos) -> bool);
    fn update(&mut self, current_pos: GridPos, is_walkable: &dyn Fn(GridPos) -> bool);
    fn next_step(&self) -> Option<GridPos>;
    /// Returns a DebugDraw impl carrying path + open/closed set data.
    /// Default: None (NoPlanner, any planner with nothing to show).
    fn debug_draw(&self) -> Option<Box<dyn DebugDraw>> { None }
    fn reset(&mut self);
}

// ── PlannerKind ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub enum PlannerKind {
    AStar,
    DStarLite,
    None,
}