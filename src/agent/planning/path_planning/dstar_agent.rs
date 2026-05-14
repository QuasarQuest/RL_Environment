// src/agent/planning/path_planning/dstar_agent.rs

use bevy::prelude::Color;
use crate::agent::action::{Action, Dir};
use crate::agent::brain::Agent;
use crate::agent::components::GridPos;
use crate::agent::observation::Observation;
use crate::agent::debug::{DebugDraw, DebugLine, DebugRect};
use crate::item::ItemKind;
use crate::world::tile::Tile;
use crate::algorithm::path_planning::d_star_lite::DStarLite;
use crate::algorithm::path_planning::graph_utils::dir_to;

pub struct DStarAgent {
    planner:  Option<DStarLite>,
    last_pos: Option<GridPos>,
}

impl DStarAgent {
    pub fn new() -> Self {
        Self { planner: None, last_pos: None }
    }
}

impl Agent for DStarAgent {
    fn name(&self) -> &str { "D* Lite" }

    fn act(&mut self, obs: &Observation<'_>) -> Action {
        // Drop on own base if carrying gold.
        let on_own_base = matches!(
            obs.grid_tile(obs.pos),
            Some(Tile::Base(t)) if t == obs.team.0
        );
        if on_own_base && !obs.gold_carried.is_empty() {
            self.reset();
            return Action::Drop;
        }

        // Replan when no active planner.
        if self.planner.is_none() {
            let target = if !obs.gold_carried.is_full() {
                obs.nearest_item(ItemKind::Gold)
            } else {
                obs.nearest_own_base()
            };
            if let Some(goal) = target {
                let mut planner = DStarLite::new(obs.pos, goal);
                planner.compute_shortest_path();
                self.planner  = Some(planner);
                self.last_pos = Some(obs.pos);
            }
        }

        if let Some(planner) = &mut self.planner {
            if let Some(last) = self.last_pos {
                if last != obs.pos {
                    planner.update_start(obs.pos);
                    self.last_pos = Some(obs.pos);
                }
            }

            // Invalidate on target consumed — replanner picks next gold.
            if !obs.gold_carried.is_full() && obs.nearest_item(ItemKind::Gold).is_none() {
                self.reset();
                return Action::Wait;
            }

            let mut changed = false;
            for dir in Dir::all() {
                let (dx, dy) = dir.delta();
                let check    = GridPos::new(obs.pos.x + dx, obs.pos.y + dy);
                if !obs.grid_is_walkable_terrain(check)
                    && !planner.known_obstacles.contains(&check)
                {
                    planner.add_obstacle(check);
                    changed = true;
                }
            }
            if changed { planner.compute_shortest_path(); }

            if let Some(next) = planner.get_next_step() {
                if let Some(dir) = dir_to(obs.pos, next) {
                    return Action::Move(dir);
                }
            }
        }
        Action::Wait
    }

    fn debug_draw(&self) -> Option<Box<dyn DebugDraw>> {
        self.planner.as_ref().map(|p| -> Box<dyn DebugDraw> {
            Box::new(DStarDebugState {
                open:      p.open_set().iter().copied().collect(),
                obstacles: p.known_obstacles.iter().copied().collect(),
                path:      p.generate_path(),
            })
        })
    }

    fn reset(&mut self) {
        self.planner  = None;
        self.last_pos = None;
    }
}

pub struct DStarDebugState {
    open:      Vec<GridPos>,
    obstacles: Vec<GridPos>,
    path:      Vec<GridPos>,
}

impl DebugDraw for DStarDebugState {
    fn draw_rects(&self) -> Vec<DebugRect> {
        let mut rects = Vec::new();
        for &p in &self.obstacles {
            rects.push(DebugRect { pos: p, color: Color::srgba(1.0, 0.10, 0.10, 0.6) });
        }
        for &p in &self.open {
            rects.push(DebugRect { pos: p, color: Color::srgba(0.70, 0.20, 0.95, 0.15) });
        }
        rects
    }

    fn draw_lines(&self, agent_pos: GridPos) -> Vec<DebugLine> {
        if self.path.is_empty() { return Vec::new(); }
        let mut lines   = Vec::with_capacity(self.path.len());
        let mut current = agent_pos;
        for &next in &self.path {
            lines.push(DebugLine { start: current, end: next, color: Color::srgb(0.85, 0.30, 1.0) });
            current = next;
        }
        lines
    }
}