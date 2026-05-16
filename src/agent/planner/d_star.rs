// src/agent/planner/d_star.rs

use bevy::prelude::Color;
use crate::agent::action::Dir;
use crate::agent::components::GridPos;
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

    fn path_for_debug(&self) -> Vec<GridPos> {
        self.inner.as_ref().map(|p| p.generate_path()).unwrap_or_default()
    }

    fn debug_rects(&self) -> Vec<(GridPos, Color)> {
        let Some(ref p) = self.inner else { return Vec::new() };
        let mut out = Vec::new();
        for &pos in &p.known_obstacles { out.push((pos, Color::srgba(1.0, 0.10, 0.10, 0.6))); }
        for pos in p.open_set()        { out.push((pos, Color::srgba(0.70, 0.20, 0.95, 0.15))); }
        out
    }

    fn reset(&mut self) { self.inner = None; self.last_pos = None; }
}