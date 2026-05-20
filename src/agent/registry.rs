// src/agent/registry.rs
//
// Pure simulation factory: constructs DecisionStrategy implementations
// from config. No display concerns — labels, colors, and info live in
// src/factory/.

use crate::world::config::AgentConfig;
use super::strategy::{
    BtStrategy, DecisionStrategy, FsmStrategy, GoapStrategy, RandomStrategy, StrategyKind,
};
use super::planner::PlannerKind;

pub fn make_agent(cfg: &AgentConfig) -> Box<dyn DecisionStrategy> {
    match (cfg.strategy, cfg.planner) {
        (StrategyKind::Fsm, PlannerKind::AStar)     => Box::new(FsmStrategy::new_astar()),
        (StrategyKind::Fsm, PlannerKind::DStarLite) => Box::new(FsmStrategy::new_dstar()),
        (StrategyKind::Fsm, PlannerKind::None)      => Box::new(FsmStrategy::new_astar()),

        (StrategyKind::BehaviorTree, PlannerKind::DStarLite) => Box::new(BtStrategy::new_dstar()),
        (StrategyKind::BehaviorTree, _)                      => Box::new(BtStrategy::new_astar()),

        (StrategyKind::Goap, PlannerKind::DStarLite) => Box::new(GoapStrategy::new_dstar()),
        (StrategyKind::Goap, _)                      => Box::new(GoapStrategy::new_astar()),

        (StrategyKind::Random, _) => Box::new(RandomStrategy),
    }
}