// src/agent/planner/d_star.rs

use bevy::prelude::Color;
use crate::agent::action::Dir;
use crate::agent::components::GridPos;
use crate::agent::debug::{DebugDraw, DebugLine};
use crate::algorithm::path_planning::d_star_lite::DStarLite;
use super::PathPlanner;

pub struct DStarPlanner {
    inner:    Option<DStarLite>,
    last_pos: Option<GridPos>,
}

impl DStarPlanner {
    pub fn new() -> Self { Self { inner: None, last_pos: None } }
}

impl PathPlanner for DStarPlanner {
    fn set_goal(&mut self, start: GridPos, goal: GridPos, _is_walkable: &dyn Fn(GridPos) -> bool) {
        let mut p = DStarLite::new(start, goal);
        p.compute_shortest_path();
        self.inner    = Some(p);
        self.last_pos = Some(start);
    }

    fn update(&mut self, current_pos: GridPos, is_walkable: &dyn Fn(GridPos) -> bool) {
        let Some(ref mut p) = self.inner else { return };
        if self.last_pos.map(|l| l != current_pos).unwrap_or(false) {
            p.update_start(current_pos);
        }
        self.last_pos = Some(current_pos);
        let mut changed = false;
        for dir in Dir::all() {
            let (dx, dy) = dir.delta();
            let check    = GridPos::new(current_pos.x + dx, current_pos.y + dy);
            if !is_walkable(check) && !p.known_obstacles.contains(&check) {
                p.add_obstacle(check);
                changed = true;
            }
        }
        if changed { p.compute_shortest_path(); }
    }

    fn next_step(&self) -> Option<GridPos> { self.inner.as_ref()?.get_next_step() }

    fn debug_draw(&self) -> Option<Box<dyn DebugDraw>> {
        let p    = self.inner.as_ref()?;
        let path = p.generate_path();
        if path.is_empty() { return None; }
        Some(Box::new(DStarDebugDraw { path }))
    }

    fn reset(&mut self) { self.inner = None; self.last_pos = None; }
}

struct DStarDebugDraw {
    path: Vec<GridPos>,
}

impl DebugDraw for DStarDebugDraw {
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