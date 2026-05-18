// src/factory/display.rs
//
// Bridges simulation and visualization.
// Not compiled in headless mode — viz components do not exist there.

#![cfg(not(feature = "headless"))]

use bevy::prelude::*;
use std::collections::HashMap;
use crate::agent::brain::AgentBrain;
use crate::agent::strategy::StrategyKind;
use crate::agent::planner::PlannerKind;
use crate::team::Team;
use crate::world::config::{AgentConfig, WorldConfig};
use crate::viz::components::{AgentInfo, AgentLabel, HidePathViz, HideRangeViz, HideViz};
use super::AgentConfigIndex;

// ── Display assignment ────────────────────────────────────────────────────────

pub fn assign_display_components(
    mut commands: Commands,
    map:          Res<WorldConfig>,
    agents:       Query<(Entity, &AgentConfigIndex, &AgentBrain), Without<AgentLabel>>,
) {
    let mut team_counters: HashMap<u8, usize> = HashMap::new();
    let team_rank: Vec<usize> = map.agents.iter().map(|cfg| {
        let team_id = cfg.team.unwrap_or(0) as u8;
        let rank    = team_counters.entry(team_id).or_insert(0);
        *rank      += 1;
        *rank
    }).collect();

    for (entity, AgentConfigIndex(idx), brain) in &agents {
        let cfg   = &map.agents[*idx];
        let team  = Team(cfg.team.unwrap_or(0) as u8);
        let rank  = team_rank[*idx];
        let label = agent_label(team, rank);
        let info  = agent_info(cfg);

        info!("Assigned display: {} [{}]", label, brain.0.name());

        commands.entity(entity)
            .insert(AgentLabel::new(label))
            .insert(info)
            .insert(HideRangeViz)
            .insert(HidePathViz)
            .insert(HideViz);
    }
}

// ── Display helpers ───────────────────────────────────────────────────────────

fn agent_label(team: Team, rank: usize) -> String {
    format!("{} #{}", team.name(), rank)
}

fn agent_info(cfg: &AgentConfig) -> AgentInfo {
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