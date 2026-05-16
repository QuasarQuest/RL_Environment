// src/agent/planner/mod.rs

pub mod a_star;
pub mod d_star;
pub mod none;

pub use a_star::AStarPlanner;
pub use d_star::DStarPlanner;
pub use none::NoPlanner;

use serde::Deserialize;
use crate::agent::components::GridPos;

// ── Trait ─────────────────────────────────────────────────────────────────────
//
// &dyn Fn instead of impl Fn — makes the trait dyn-compatible so
// BtStrategy can own a Box<dyn PathPlanner>.

pub trait PathPlanner: Send + Sync {
    fn set_goal(&mut self, start: GridPos, goal: GridPos, is_walkable: &dyn Fn(GridPos) -> bool);
    fn update(&mut self, current_pos: GridPos, is_walkable: &dyn Fn(GridPos) -> bool);
    fn next_step(&self) -> Option<GridPos>;
    fn path_for_debug(&self) -> Vec<GridPos> { Vec::new() }
    fn debug_rects(&self) -> Vec<(GridPos, bevy::prelude::Color)> { Vec::new() }
    fn reset(&mut self);
}

// ── PlannerKind ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub enum PlannerKind {
    AStar,
    DStarLite,
    None,
}