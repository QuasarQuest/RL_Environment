// src/agent/registry.rs
//
// Maps AgentKind → Box<dyn Agent>.
// Adding a new agent = add one arm here. Nothing else changes.

use crate::world::map_config::AgentKind;
use crate::team::Team;
use super::brain::Agent;
use super::planning::random_agent::RandomAgent;
use super::planning::path_planning::astar_agent::AStarAgent;
use super::planning::path_planning::dstar_agent::DStarAgent;

pub fn agent_color(kind: AgentKind, team: Team) -> bevy::prelude::Color {
    // Team colour takes priority when teams are active.
    // Falls back to algorithm colour for solo/no-team scenarios.
    team.color()
}

pub fn make_brain(kind: AgentKind) -> Box<dyn Agent> {
    match kind {
        AgentKind::Random    => Box::new(RandomAgent),
        AgentKind::AStar     => Box::new(AStarAgent::new()),
        AgentKind::DStarLite => Box::new(DStarAgent::new()),
    }
}

pub fn agent_name_prefix(kind: AgentKind) -> &'static str {
    match kind {
        AgentKind::Random    => "Random",
        AgentKind::AStar     => "A*",
        AgentKind::DStarLite => "D* Lite",
    }
}