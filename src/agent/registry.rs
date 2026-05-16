// src/agent/registry.rs

use bevy::prelude::Color;
use crate::team::Team;
use crate::world::map_config::AgentConfig;
use super::composition::Brain;
use super::brain::AgentBehavior;
use super::planning::planner::{AStarPlanner, DStarPlanner, NoPlanner, PlannerKind};
use super::planning::strategy::{
    BtAggressiveStrategy, BtCautiousStrategy,
    FsmStrategy, GoapStrategy, RandomStrategy, StrategyKind,
};

pub fn make_agent(cfg: &AgentConfig) -> Box<dyn AgentBehavior> {
    match (cfg.strategy, cfg.planner) {
        // FSM
        (StrategyKind::Fsm, PlannerKind::AStar)     => Box::new(Brain::new(FsmStrategy::new(),  AStarPlanner::new())),
        (StrategyKind::Fsm, PlannerKind::DStarLite) => Box::new(Brain::new(FsmStrategy::new(),  DStarPlanner::new())),
        (StrategyKind::Fsm, PlannerKind::None)      => Box::new(Brain::new(FsmStrategy::new(),  NoPlanner)),

        // BT Cautious
        (StrategyKind::BehaviorTree, PlannerKind::AStar)     => Box::new(Brain::new(BtCautiousStrategy::<AStarPlanner>::new(), AStarPlanner::new())),
        (StrategyKind::BehaviorTree, PlannerKind::DStarLite) => Box::new(Brain::new(BtCautiousStrategy::<DStarPlanner>::new(), DStarPlanner::new())),
        (StrategyKind::BehaviorTree, PlannerKind::None)      => Box::new(Brain::new(BtCautiousStrategy::<NoPlanner>::new(),    NoPlanner)),

        // BT Aggressive
        (StrategyKind::BehaviorTreeAggressive, PlannerKind::AStar)     => Box::new(Brain::new(BtAggressiveStrategy::<AStarPlanner>::new(), AStarPlanner::new())),
        (StrategyKind::BehaviorTreeAggressive, PlannerKind::DStarLite) => Box::new(Brain::new(BtAggressiveStrategy::<DStarPlanner>::new(), DStarPlanner::new())),
        (StrategyKind::BehaviorTreeAggressive, PlannerKind::None)      => Box::new(Brain::new(BtAggressiveStrategy::<NoPlanner>::new(),    NoPlanner)),

        // GOAP
        (StrategyKind::Goap, PlannerKind::DStarLite) => Box::new(Brain::new(GoapStrategy::new(), DStarPlanner::new())),
        (StrategyKind::Goap, _)                      => Box::new(Brain::new(GoapStrategy::new(), AStarPlanner::new())),

        // Random
        (StrategyKind::Random, _) => Box::new(Brain::new(RandomStrategy, NoPlanner)),

        // Fallback
        (_, PlannerKind::None) => Box::new(Brain::new(FsmStrategy::new(), NoPlanner)),
    }
}

pub fn agent_color(_cfg: &AgentConfig, team: Team) -> Color { team.color() }

pub fn agent_label(cfg: &AgentConfig, id: usize, team: Team) -> String {
    format!("{} {:?}+{:?} #{id}", team.name(), cfg.strategy, cfg.planner)
}