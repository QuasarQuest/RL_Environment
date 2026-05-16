// src/agent/composition.rs

use bevy::prelude::Color;
use crate::agent::action::Action;
use crate::agent::brain::AgentBehavior;
use crate::agent::components::GridPos;
use crate::agent::debug::{DebugDraw, DebugLine, DebugRect};
use crate::agent::observation::Observation;
use crate::agent::planning::planner::PathPlanner;
use crate::agent::planning::strategy::DecisionStrategy;

pub struct Brain<S: DecisionStrategy, P: PathPlanner> {
    pub strategy: S,
    pub planner:  P,
}

impl<S: DecisionStrategy, P: PathPlanner> Brain<S, P> {
    pub fn new(strategy: S, planner: P) -> Self {
        Self { strategy, planner }
    }
}

impl<S: DecisionStrategy, P: PathPlanner> AgentBehavior for Brain<S, P> {
    fn name(&self) -> &str { self.strategy.name() }

    fn act(&mut self, obs: &Observation) -> Action {
        self.strategy.decide(obs, &mut self.planner)
    }

    fn debug_draw(&self) -> Option<Box<dyn DebugDraw>> {
        let path  = self.planner.path_for_debug();
        let rects = self.planner.debug_rects();
        if path.is_empty() && rects.is_empty() { return None; }
        Some(Box::new(BrainDebugDraw { path, rects }))
    }

    fn reset(&mut self) {
        self.strategy.reset();
        self.planner.reset();
    }
}

struct BrainDebugDraw {
    path:  Vec<GridPos>,
    rects: Vec<(GridPos, Color)>,
}

impl DebugDraw for BrainDebugDraw {
    fn draw_rects(&self) -> Vec<DebugRect> {
        self.rects.iter().map(|&(pos, color)| DebugRect { pos, color }).collect()
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