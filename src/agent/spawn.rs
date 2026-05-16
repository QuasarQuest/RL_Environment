// src/agent/spawn.rs

use bevy::prelude::*;
use crate::config;
use crate::team::Team;
use crate::world::map_config::MapConfig;
use crate::viz::hud::HideViz;
use super::brain::AgentBrain;
use super::components::{AgentLabel, Ammo, GoldCarried, GridPos, Hearts, Score, SpawnPoint};
use super::systems::PendingAction;
use super::registry::{agent_color, agent_label, make_agent};

#[derive(Bundle)]
pub struct AgentBundle {
    pub pos:         GridPos,
    pub spawn_point: SpawnPoint,
    pub hearts:      Hearts,
    pub ammo:        Ammo,
    pub gold:        GoldCarried,
    pub score:       Score,
    pub label:       AgentLabel,
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
        brain: AgentBrain,
        team:  Team,
        color: Color,
    ) -> Self {
        let pos = GridPos::new(x, y);
        Self {
            pos,
            spawn_point: SpawnPoint(pos),
            hearts:     Hearts::default(),
            ammo:       Ammo::default(),
            gold:       GoldCarried::default(),
            score:      Score::default(),
            label,
            brain,
            team,
            pending:    PendingAction::default(),
            sprite:     Sprite {
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
    for (i, cfg) in map.agents.iter().enumerate() {
        let id    = i + 1;
        let team  = Team(cfg.team.unwrap_or(0) as u8);
        let label = AgentLabel::new(agent_label(cfg, id, team));
        let brain = AgentBrain(make_agent(cfg));

        // Log the brain name at spawn — this is also what makes
        // AgentBehavior::name() reachable at the ECS level.
        info!("Spawning agent #{id} [{}] for team {}", brain.0.name(), team.name());

        let color = agent_color(cfg, team);
        commands.spawn(AgentBundle::new(cfg.x, cfg.y, label, brain, team, color))
            .insert(HideViz);
    }
}