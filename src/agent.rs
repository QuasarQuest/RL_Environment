// src/agent.rs

pub mod action;
pub mod brain;
pub mod combat;
pub mod components;
pub mod debug;
pub mod observation;
pub mod planning;
pub mod registry;
pub mod spawn;
pub mod systems;

use bevy::prelude::*;
use crate::sim::OnSimTick;
use crate::team::TeamScore;
use systems::{tick_agents, apply_actions};
use spawn::spawn_agents;
use combat::{resolve_combat, despawn_dead};

pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<TeamScore>()
            .add_systems(Startup, spawn_agents)
            .add_systems(OnSimTick, (
                tick_agents,
                apply_actions,
                resolve_combat,
                despawn_dead,
            ).chain());
    }
}