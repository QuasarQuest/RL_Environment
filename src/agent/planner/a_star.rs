// src/agent/planner/a_star.rs

use std::collections::VecDeque;
use bevy::prelude::Color;
use crate::agent::components::GridPos;
use crate::agent::debug::{DebugDraw, DebugLine};
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

    fn debug_draw(&self) -> Option<Box<dyn DebugDraw>> {
        if self.path.is_empty() { return None; }
        Some(Box::new(AStarDebugDraw {
            path: self.path.iter().copied().collect(),
        }))
    }

    fn reset(&mut self) {
        self.path.clear();
        self.debug_open.clear();
        self.debug_closed.clear();
    }
}

struct AStarDebugDraw {
    path: Vec<GridPos>,
}

impl DebugDraw for AStarDebugDraw {
    fn draw_lines(&self, agent_pos: GridPos) -> Vec<DebugLine> {
        if self.path.is_empty() { return Vec::new(); }
        let mut lines = Vec::with_capacity(self.path.len());
        let mut cur   = agent_pos;
        for &next in &self.path {
            lines.push(DebugLine { start: cur, end: next, color: Color::srgb(1.0, 0.90, 0.10) });
            cur = next;
        }
        lines
    }
}