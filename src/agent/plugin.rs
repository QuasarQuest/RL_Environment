// src/agent/plugin.rs

use bevy::prelude::*;
use crate::sim::OnSimTick;
use crate::team::TeamScore;
use super::systems::{tick_agents, apply_actions, tick_speed_buff};
use super::spawn::spawn_agents;
use super::combat::{tick_attack_cooldown, resolve_combat, tick_respawn};

pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<TeamScore>()
            .add_systems(Startup, spawn_agents)
            .add_systems(OnSimTick, (
                tick_speed_buff,
                tick_attack_cooldown,
                tick_agents,
                apply_actions,
                resolve_combat,
                tick_respawn,
            ).chain());
    }
}