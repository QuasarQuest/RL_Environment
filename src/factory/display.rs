// src/factory/display.rs
//
// Bridges simulation and visualization — assigns display components to agents.
// Not compiled in headless mode.

#![cfg(not(feature = "headless"))]

use bevy::prelude::*;
use std::collections::HashMap;
use crate::agent::brain::AgentBrain;
use crate::agent::strategy::StrategyKind;
use crate::agent::planner::PlannerKind;
use crate::team::Team;
use crate::world::layout::{ResolvedAgent, ResolvedLayout};
use crate::viz::components::{AgentInfo, AgentLabel, HidePathViz, HideRangeViz, HideViz};
use super::AgentConfigIndex;

// ── Display assignment ────────────────────────────────────────────────────────

pub fn assign_display_components(
    mut commands: Commands,
    layout:       Res<ResolvedLayout>,
    agents:       Query<(Entity, &AgentConfigIndex, &AgentBrain), Without<AgentLabel>>,
) {
    // Count how many agents per team so we can number them (Red #1, Red #2…)
    let mut team_counters: HashMap<u8, usize> = HashMap::new();

    // Pre-compute ranks in layout order so indexing stays stable
    let team_ranks: Vec<usize> = layout.agents.iter().map(|a| {
        let rank = team_counters.entry(a.team).or_insert(0);
        *rank += 1;
        *rank
    }).collect();

    for (entity, AgentConfigIndex(idx), brain) in &agents {
        let Some(agent) = layout.agents.get(*idx) else { continue };
        let team  = Team(agent.team);
        let rank  = team_ranks[*idx];
        let label = format!("{} #{}", team.name(), rank);
        let info  = agent_info(agent);

        info!("Assigned display: {} [{}]", label, brain.0.name());

        commands.entity(entity)
            .insert(AgentLabel::new(label))
            .insert(info)
            .insert(HideRangeViz)
            .insert(HidePathViz)
            .insert(HideViz);
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn agent_info(agent: &ResolvedAgent) -> AgentInfo {
    let strategy = match agent.strategy {
        StrategyKind::Fsm          => "FSM",
        StrategyKind::BehaviorTree => "Behavior Tree",
        StrategyKind::Goap         => "GOAP",
        StrategyKind::Random       => "Random",
    };
    let planner = match agent.planner {
        PlannerKind::AStar     => "A*",
        PlannerKind::DStarLite => "D* Lite",
        PlannerKind::None      => "None",
    };
    AgentInfo { strategy, planner }
}