// src/agent/spawn.rs

use bevy::prelude::*;
use std::collections::HashMap;
use crate::config;
use crate::team::Team;
use crate::world::map_config::MapConfig;
use crate::viz::hud::HideViz;
use super::brain::AgentBrain;
use super::components::{AgentInfo, AgentLabel, Ammo, GoldCarried, GridPos, Hearts, Score, SpawnPoint};
use super::systems::PendingAction;
use super::registry::{agent_color, agent_info, agent_label, make_agent};

#[derive(Bundle)]
pub struct AgentBundle {
    pub pos:         GridPos,
    pub spawn_point: SpawnPoint,
    pub hearts:      Hearts,
    pub ammo:        Ammo,
    pub gold:        GoldCarried,
    pub score:       Score,
    pub label:       AgentLabel,
    pub info:        AgentInfo,
    pub brain:       AgentBrain,
    pub team:        Team,
    pub pending:     PendingAction,
    pub sprite:      Sprite,
    pub transform:   Transform,
    pub visibility:  Visibility,
}

impl AgentBundle {
    pub fn new(
        x: i32, y: i32,
        label: AgentLabel,
        info:  AgentInfo,
        brain: AgentBrain,
        team:  Team,
        color: Color,
    ) -> Self {
        let pos = GridPos::new(x, y);
        Self {
            pos,
            spawn_point: SpawnPoint(pos),
            hearts:  Hearts::default(),
            ammo:    Ammo::default(),
            gold:    GoldCarried::default(),
            score:   Score::default(),
            label,
            info,
            brain,
            team,
            pending:   PendingAction::default(),
            sprite:    Sprite {
                color,
                custom_size: Some(Vec2::splat(config::TILE_SIZE * 0.8)),
                ..default()
            },
            transform:  Transform::from_xyz(0.0, 0.0, 1.0),
            visibility: Visibility::default(),
        }
    }
}

pub fn spawn_agents(mut commands: Commands, map: Res<MapConfig>) {
    // Count agents per team to assign consecutive per-team numbers.
    let mut team_counters: HashMap<u8, usize> = HashMap::new();

    for cfg in map.agents.iter() {
        let team_id  = cfg.team.unwrap_or(0) as u8;
        let team     = Team(team_id);
        let counter  = team_counters.entry(team_id).or_insert(0);
        *counter    += 1;
        let team_idx = *counter;

        let label = AgentLabel::new(agent_label(team, team_idx));
        let info  = agent_info(cfg);
        let brain = AgentBrain(make_agent(cfg));
        let color = agent_color(cfg, team);

        info!("Spawning {} [{}] for team {}",
            label.0, brain.0.name(), team.name());

        commands.spawn(AgentBundle::new(cfg.x, cfg.y, label, info, brain, team, color))
            .insert(HideViz);
    }
}