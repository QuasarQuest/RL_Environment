// src/agent/planner/a_star.rs

use std::collections::VecDeque;
use bevy::prelude::Color;
use crate::agent::components::GridPos;
use crate::algorithm::path_planning::a_star::compute_path;
use super::PathPlanner;

pub struct AStarPlanner {
    path:         VecDeque<GridPos>,
    debug_open:   Vec<GridPos>,
    debug_closed: Vec<GridPos>,
}

impl AStarPlanner {
    pub fn new() -> Self {
        Self {
            path:         VecDeque::new(),
            debug_open:   Vec::new(),
            debug_closed: Vec::new(),
        }
    }
}

impl PathPlanner for AStarPlanner {
    fn set_goal(&mut self, start: GridPos, goal: GridPos, is_walkable: &dyn Fn(GridPos) -> bool) {
        let result        = compute_path(start, goal, is_walkable);
        self.debug_closed = result.closed_set.into_iter().collect();
        self.debug_open   = result.open_set;
        self.path         = result.path.into();
    }

    fn update(&mut self, current_pos: GridPos, _is_walkable: &dyn Fn(GridPos) -> bool) {
        if self.path.front() == Some(&current_pos) {
            self.path.pop_front();
        }
    }

    fn next_step(&self) -> Option<GridPos> { self.path.front().copied() }

    fn path_for_debug(&self) -> Vec<GridPos> { self.path.iter().copied().collect() }

    fn debug_rects(&self) -> Vec<(GridPos, Color)> {
        let mut out = Vec::new();
        for &p in &self.debug_closed { out.push((p, Color::srgba(0.85, 0.20, 0.20, 0.18))); }
        for &p in &self.debug_open   { out.push((p, Color::srgba(0.20, 0.85, 0.20, 0.28))); }
        out
    }

    fn reset(&mut self) {
        self.path.clear();
        self.debug_open.clear();
        self.debug_closed.clear();
    }
}