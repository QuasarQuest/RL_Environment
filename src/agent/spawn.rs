// src/agent/spawn.rs

use bevy::prelude::*;
use crate::config;
use crate::factory::AgentConfigIndex;
use crate::team::Team;
use crate::world::config::WorldConfig;
use super::brain::AgentBrain;
use super::components::{
    Ammo, DeathCount, GoldCarried, GridPos, Hearts, KillCount, Score, SpawnPoint,
};
use super::systems::PendingAction;
use super::registry::make_agent;

#[derive(Bundle)]
pub struct AgentBundle {
    pub pos:         GridPos,
    pub spawn_point: SpawnPoint,
    pub hearts:      Hearts,
    pub ammo:        Ammo,
    pub gold:        GoldCarried,
    pub score:       Score,
    pub kills:       KillCount,
    pub deaths:      DeathCount,
    pub brain:       AgentBrain,
    pub team:        Team,
    pub pending:     PendingAction,
    pub sprite:      Sprite,
    pub transform:   Transform,
    pub visibility:  Visibility,
}

impl AgentBundle {
    pub fn new(
        x:     i32,
        y:     i32,
        brain: AgentBrain,
        team:  Team,
        color: bevy::prelude::Color,
    ) -> Self {
        let pos = GridPos::new(x, y);
        Self {
            pos,
            spawn_point: SpawnPoint(pos),
            hearts:  Hearts::default(),
            ammo:    Ammo::default(),
            gold:    GoldCarried::default(),
            score:   Score::default(),
            kills:   KillCount::default(),
            deaths:  DeathCount::default(),
            brain,
            team,
            pending:    PendingAction::default(),
            sprite: Sprite {
                color,
                custom_size: Some(Vec2::splat(config::TILE_SIZE * 0.8)),
                ..default()
            },
            transform:  Transform::from_xyz(0.0, 0.0, 1.0),
            visibility: Visibility::default(),
        }
    }
}

pub fn spawn_agents(mut commands: Commands, map: Res<WorldConfig>) {
    for (idx, cfg) in map.agents.iter().enumerate() {
        let team  = Team(cfg.team.unwrap_or(0) as u8);
        let brain = AgentBrain(make_agent(cfg));
        let color = team.color();
        commands.spawn((
            AgentBundle::new(cfg.x, cfg.y, brain, team, color),
            AgentConfigIndex(idx),
        ));
    }
}