// src/agent/registry.rs
//
// Pure simulation factory: constructs AgentBehavior implementations from config.
// No display concerns — labels, colors, and info live in src/factory/.

use crate::world::config::AgentConfig;
use super::composition::Brain;
use super::brain::AgentBehavior;
use super::planner::{AStarPlanner, DStarPlanner, NoPlanner, PlannerKind};
use super::strategy::{BtStrategy, FsmStrategy, GoapStrategy, RandomStrategy, StrategyKind};

pub fn make_agent(cfg: &AgentConfig) -> Box<dyn AgentBehavior> {
    match (cfg.strategy, cfg.planner) {
        (StrategyKind::Fsm, PlannerKind::AStar)     => Box::new(Brain::new(FsmStrategy::new(), AStarPlanner::new())),
        (StrategyKind::Fsm, PlannerKind::DStarLite) => Box::new(Brain::new(FsmStrategy::new(), DStarPlanner::new())),
        (StrategyKind::Fsm, PlannerKind::None)      => Box::new(Brain::new(FsmStrategy::new(), NoPlanner)),

        (StrategyKind::BehaviorTree, PlannerKind::DStarLite) => Box::new(Brain::new(BtStrategy::new_dstar(), NoPlanner)),
        (StrategyKind::BehaviorTree, _)                      => Box::new(Brain::new(BtStrategy::new_astar(), NoPlanner)),

        (StrategyKind::Goap, PlannerKind::DStarLite) => Box::new(Brain::new(GoapStrategy::new(), DStarPlanner::new())),
        (StrategyKind::Goap, _)                      => Box::new(Brain::new(GoapStrategy::new(), AStarPlanner::new())),

        (StrategyKind::Random, _) => Box::new(Brain::new(RandomStrategy, NoPlanner)),
    }
}