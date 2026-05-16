// src/agent/planner/none.rs

use crate::agent::components::GridPos;
use super::PathPlanner;

pub struct NoPlanner;

impl PathPlanner for NoPlanner {
    fn set_goal(&mut self, _s: GridPos, _g: GridPos, _w: &dyn Fn(GridPos) -> bool) {}
    fn update(&mut self, _pos: GridPos, _w: &dyn Fn(GridPos) -> bool) {}
    fn next_step(&self) -> Option<GridPos> { None }
    fn reset(&mut self) {}
}