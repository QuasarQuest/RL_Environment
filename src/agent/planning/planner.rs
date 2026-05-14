// src/agent/planning/planner.rs

use std::collections::VecDeque;
use serde::Deserialize;
use crate::agent::action::Dir;
use crate::agent::components::GridPos;
use crate::algorithm::path_planning::a_star::compute_path;
use crate::algorithm::path_planning::d_star_lite::DStarLite;

// ── Trait ─────────────────────────────────────────────────────────────────────

pub trait PathPlanner: Send + Sync {
    fn name(&self) -> &'static str;
    fn set_goal(&mut self, start: GridPos, goal: GridPos, is_walkable: impl Fn(GridPos) -> bool);
    fn update(&mut self, current_pos: GridPos, is_walkable: impl Fn(GridPos) -> bool);
    fn next_step(&self) -> Option<GridPos>;
    fn path_for_debug(&self) -> Vec<GridPos> { Vec::new() }
    fn debug_rects(&self) -> Vec<(GridPos, bevy::prelude::Color)> { Vec::new() }
    fn reset(&mut self);
}

// ── NoPlanner ─────────────────────────────────────────────────────────────────

pub struct NoPlanner;

impl PathPlanner for NoPlanner {
    fn name(&self) -> &'static str { "None" }
    fn set_goal(&mut self, _s: GridPos, _g: GridPos, _w: impl Fn(GridPos) -> bool) {}
    fn update(&mut self, _pos: GridPos, _w: impl Fn(GridPos) -> bool) {}
    fn next_step(&self) -> Option<GridPos> { None }
    fn reset(&mut self) {}
}

// ── A* planner ────────────────────────────────────────────────────────────────

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
    fn name(&self) -> &'static str { "A*" }

    fn set_goal(&mut self, start: GridPos, goal: GridPos, is_walkable: impl Fn(GridPos) -> bool) {
        let result        = compute_path(start, goal, is_walkable);
        self.debug_closed = result.closed_set.into_iter().collect();
        self.debug_open   = result.open_set;
        self.path         = result.path.into();
    }

    fn update(&mut self, current_pos: GridPos, _is_walkable: impl Fn(GridPos) -> bool) {
        if self.path.front() == Some(&current_pos) {
            self.path.pop_front();
        }
    }

    fn next_step(&self) -> Option<GridPos> { self.path.front().copied() }
    fn path_for_debug(&self) -> Vec<GridPos> { self.path.iter().copied().collect() }

    fn debug_rects(&self) -> Vec<(GridPos, bevy::prelude::Color)> {
        use bevy::prelude::Color;
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

// ── D* Lite planner ───────────────────────────────────────────────────────────

pub struct DStarPlanner {
    inner:    Option<DStarLite>,
    last_pos: Option<GridPos>,
}

impl DStarPlanner {
    pub fn new() -> Self { Self { inner: None, last_pos: None } }
}

impl PathPlanner for DStarPlanner {
    fn name(&self) -> &'static str { "D* Lite" }

    fn set_goal(&mut self, start: GridPos, goal: GridPos, _is_walkable: impl Fn(GridPos) -> bool) {
        let mut p = DStarLite::new(start, goal);
        p.compute_shortest_path();
        self.inner    = Some(p);
        self.last_pos = Some(start);
    }

    fn update(&mut self, current_pos: GridPos, is_walkable: impl Fn(GridPos) -> bool) {
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

    fn debug_rects(&self) -> Vec<(GridPos, bevy::prelude::Color)> {
        use bevy::prelude::Color;
        let Some(ref p) = self.inner else { return Vec::new() };
        let mut out = Vec::new();
        for &pos in &p.known_obstacles { out.push((pos, Color::srgba(1.0, 0.10, 0.10, 0.6))); }
        for pos in p.open_set()        { out.push((pos, Color::srgba(0.70, 0.20, 0.95, 0.15))); }
        out
    }

    fn reset(&mut self) { self.inner = None; self.last_pos = None; }
}

// ── PlannerKind ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub enum PlannerKind {
    AStar,
    DStarLite,
    None,
}