// src/agent/planning/path_planning/astar_agent.rs

use std::collections::VecDeque;
use bevy::prelude::Color;
use crate::agent::action::Action;
use crate::agent::brain::Agent;
use crate::agent::components::GridPos;
use crate::agent::observation::Observation;
use crate::agent::debug::{DebugDraw, DebugLine, DebugRect};
use crate::item::ItemKind;
use crate::world::tile::Tile;
use crate::algorithm::path_planning::a_star::compute_path;
use crate::algorithm::path_planning::graph_utils::dir_to;

pub struct AStarAgent {
    path:         VecDeque<GridPos>,
    debug_open:   Vec<GridPos>,
    debug_closed: Vec<GridPos>,
}

impl AStarAgent {
    pub fn new() -> Self {
        Self { path: VecDeque::new(), debug_open: Vec::new(), debug_closed: Vec::new() }
    }
}

impl Agent for AStarAgent {
    fn name(&self) -> &str { "A* Search" }

    fn act(&mut self, obs: &Observation<'_>) -> Action {
        if self.path.front() == Some(&obs.pos) { self.path.pop_front(); }

        // Standing on own base with gold → drop.
        let on_own_base = matches!(obs.grid_tile(obs.pos), Some(Tile::Base(t)) if t == obs.team.0);
        if on_own_base && !obs.gold_carried.is_empty() {
            self.path.clear();
            return Action::Drop;
        }

        // Invalidate path only on hard terrain change.
        if let Some(&next) = self.path.front() {
            if !obs.grid_is_walkable_terrain(next) { self.path.clear(); }
        }

        // Replan if empty.
        if self.path.is_empty() {
            let target = if !obs.gold_carried.is_full() {
                obs.nearest_item(ItemKind::Gold)
            } else {
                obs.nearest_own_base()
            };

            if let Some(goal) = target {
                let result = compute_path(obs.pos, goal, |pos| {
                    obs.grid_is_walkable_terrain(pos)
                });
                self.path         = result.path.into();
                self.debug_closed = result.closed_set.into_iter().collect();
                self.debug_open   = result.open_set;
            }
        }

        if let Some(&next_pos) = self.path.front() {
            if obs.is_walkable(next_pos) {
                if let Some(dir) = dir_to(obs.pos, next_pos) {
                    return Action::Move(dir);
                }
            }
        }
        Action::Wait
    }

    fn debug_draw(&self) -> Option<Box<dyn DebugDraw>> {
        Some(Box::new(AStarDebugState {
            open:   self.debug_open.clone(),
            closed: self.debug_closed.clone(),
            path:   self.path.iter().copied().collect(),
        }))
    }

    fn reset(&mut self) {
        self.path.clear();
        self.debug_open.clear();
        self.debug_closed.clear();
    }
}

pub struct AStarDebugState {
    open:   Vec<GridPos>,
    closed: Vec<GridPos>,
    path:   Vec<GridPos>,
}

impl DebugDraw for AStarDebugState {
    fn draw_rects(&self) -> Vec<DebugRect> {
        let mut rects = Vec::new();
        for &p in &self.closed {
            rects.push(DebugRect { pos: p, color: Color::srgba(0.85, 0.20, 0.20, 0.18) });
        }
        for &p in &self.open {
            rects.push(DebugRect { pos: p, color: Color::srgba(0.20, 0.85, 0.20, 0.28) });
        }
        rects
    }

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