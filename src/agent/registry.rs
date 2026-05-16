// src/agent/registry.rs

use bevy::prelude::Color;
use crate::team::Team;
use crate::world::map_config::AgentConfig;
use super::composition::Brain;
use super::brain::AgentBehavior;
use super::planning::planner::{AStarPlanner, DStarPlanner, NoPlanner, PlannerKind};
use super::planning::strategy::{BtStrategy, FsmStrategy, GoapStrategy, RandomStrategy, StrategyKind};

pub fn make_agent(cfg: &AgentConfig) -> Box<dyn AgentBehavior> {
    match (cfg.strategy, cfg.planner) {
        // FSM — Brain planner used externally
        (StrategyKind::Fsm, PlannerKind::AStar)     => Box::new(Brain::new(FsmStrategy::new(), AStarPlanner::new())),
        (StrategyKind::Fsm, PlannerKind::DStarLite) => Box::new(Brain::new(FsmStrategy::new(), DStarPlanner::new())),
        (StrategyKind::Fsm, PlannerKind::None)      => Box::new(Brain::new(FsmStrategy::new(), NoPlanner)),

        // BT — owns its planner internally; Brain planner is NoPlanner
        (StrategyKind::BehaviorTree, PlannerKind::DStarLite) => Box::new(Brain::new(BtStrategy::new_dstar(), NoPlanner)),
        (StrategyKind::BehaviorTree, _)                      => Box::new(Brain::new(BtStrategy::new_astar(), NoPlanner)),

        // GOAP — Brain planner used externally
        (StrategyKind::Goap, PlannerKind::DStarLite) => Box::new(Brain::new(GoapStrategy::new(), DStarPlanner::new())),
        (StrategyKind::Goap, _)                      => Box::new(Brain::new(GoapStrategy::new(), AStarPlanner::new())),

        // Random
        (StrategyKind::Random, _) => Box::new(Brain::new(RandomStrategy, NoPlanner)),
    }
}

pub fn agent_color(_cfg: &AgentConfig, team: Team) -> Color { team.color() }

pub fn agent_label(cfg: &AgentConfig, id: usize, team: Team) -> String {
    format!("{} {:?}+{:?} #{id}", team.name(), cfg.strategy, cfg.planner)
}