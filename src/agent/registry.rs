// src/agent/registry.rs

use bevy::prelude::Color;
use crate::team::Team;
use crate::world::map_config::AgentConfig;
use super::components::AgentInfo;
use super::composition::Brain;
use super::brain::AgentBehavior;
use super::planning::planner::{AStarPlanner, DStarPlanner, NoPlanner, PlannerKind};
use super::planning::strategy::{BtStrategy, FsmStrategy, GoapStrategy, RandomStrategy, StrategyKind};

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

/// Per-team consecutive label: "Red #1", "Blue #1", etc.
pub fn agent_label(team: Team, team_index: usize) -> String {
    format!("{} #{}", team.name(), team_index)
}

/// Strategy and planner names for scoreboard subtext.
pub fn agent_info(cfg: &AgentConfig) -> AgentInfo {
    let strategy = match cfg.strategy {
        StrategyKind::Fsm          => "FSM",
        StrategyKind::BehaviorTree => "Behavior Tree",
        StrategyKind::Goap         => "GOAP",
        StrategyKind::Random       => "Random",
    };
    let planner = match cfg.planner {
        PlannerKind::AStar     => "A*",
        PlannerKind::DStarLite => "D* Lite",
        PlannerKind::None      => "None",
    };
    AgentInfo { strategy, planner }
}

pub fn agent_color(_cfg: &AgentConfig, team: Team) -> Color { team.color() }