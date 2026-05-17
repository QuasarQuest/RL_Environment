// src/agent/composition.rs

use crate::agent::action::Action;
use crate::agent::brain::AgentBehavior;
use crate::agent::debug::DebugDraw;
use crate::agent::observation::Observation;
use crate::agent::planner::PathPlanner;
use crate::agent::strategy::DecisionStrategy;

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
        // Strategy-owned planner (BtStrategy) takes priority.
        // Falls back to Brain's outer planner (FsmStrategy, GoapStrategy).
        self.strategy.debug_draw()
            .or_else(|| self.planner.debug_draw())
    }

    fn reset(&mut self) {
        self.strategy.reset();
        self.planner.reset();
    }
}