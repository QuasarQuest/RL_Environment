// src/factory/mod.rs
//
// Factory layer: bridges simulation and visualization.
//
// Runs after AgentPlugin::spawn_agents (Startup). Queries all spawned agent
// entities, assigns display components (AgentLabel, AgentInfo, HideViz) that
// the viz layer needs but the sim layer must not know about.
//
// Dependency arrows:
//   factory → agent (reads Team, AgentBrain for name)
//   factory → world (reads MapConfig for strategy/planner metadata)
//   factory → viz   (inserts AgentLabel, AgentInfo, HideViz)
//   agent   → nothing in factory or viz  ✓

use bevy::prelude::*;
use std::collections::HashMap;
use crate::agent::brain::AgentBrain;
use crate::agent::components::GridPos;
use crate::team::Team;
use crate::world::map_config::{AgentConfig, MapConfig};
use crate::agent::strategy::StrategyKind;
use crate::agent::planner::PlannerKind;
use crate::viz::components::{AgentInfo, AgentLabel, HideViz};

// ── Display assignment ────────────────────────────────────────────────────────
//
// Runs once at Startup after spawn_agents. Pairs spawned agent entities
// with their MapConfig entry by spawn order — both iterate in the same
// order (MapConfig::agents index == spawn order).

pub fn assign_display_components(
    mut commands: Commands,
    map:          Res<MapConfig>,
    agents:       Query<(Entity, &Team, &AgentBrain), Without<AgentLabel>>,
) {
    // Collect agents sorted by entity insertion order isn't guaranteed,
    // so we match by Team + per-team index using the same counter logic
    // as MapConfig.agents iteration order.
    let mut team_counters: HashMap<u8, usize> = HashMap::new();

    // Build a lookup: (team_id, team_index) → AgentConfig
    let mut config_map: HashMap<(u8, usize), &AgentConfig> = HashMap::new();
    let mut cfg_counters: HashMap<u8, usize> = HashMap::new();
    for cfg in &map.agents {
        let team_id = cfg.team.unwrap_or(0) as u8;
        let counter = cfg_counters.entry(team_id).or_insert(0);
        *counter += 1;
        config_map.insert((team_id, *counter), cfg);
    }

    // Agents are spawned in MapConfig order. Query Without<AgentLabel>
    // ensures this is idempotent if called more than once.
    // We sort by entity to get a stable order.
    let mut agent_list: Vec<(Entity, u8, &AgentBrain)> = agents
        .iter()
        .map(|(e, team, brain)| (e, team.0, brain))
        .collect();
    agent_list.sort_by_key(|(e, _, _)| *e);

    for (entity, team_id, brain) in agent_list {
        let counter  = team_counters.entry(team_id).or_insert(0);
        *counter    += 1;
        let team_idx = *counter;
        let team     = Team(team_id);

        let label = agent_label(team, team_idx);
        let info  = config_map
            .get(&(team_id, team_idx))
            .map(|cfg| agent_info(cfg))
            .unwrap_or(AgentInfo { strategy: "?", planner: "?" });

        info!("Assigned display: {} [{}]", label, brain.0.name());

        commands.entity(entity)
            .insert(AgentLabel::new(label))
            .insert(info)
            .insert(HideViz); // debug overlay hidden by default
    }
}

// ── Display helpers ───────────────────────────────────────────────────────────

fn agent_label(team: Team, team_index: usize) -> String {
    format!("{} #{}", team.name(), team_index)
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