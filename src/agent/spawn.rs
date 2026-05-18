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

#[cfg(feature = "python")]
use crate::rl::marker::RlAgent;

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
        start_pos:   GridPos,
        spawn_point: GridPos,
        brain:       AgentBrain,
        team:        Team,
        color:       bevy::prelude::Color,
    ) -> Self {
        Self {
            pos:         start_pos,
            spawn_point: SpawnPoint(spawn_point),
            hearts:      Hearts::default(),
            ammo:        Ammo::default(),
            gold:        GoldCarried::default(),
            score:       Score::default(),
            kills:       KillCount::default(),
            deaths:      DeathCount::default(),
            brain,
            team,
            pending:     PendingAction::default(),
            sprite: Sprite {
                color,
                custom_size: Some(Vec2::splat(config::TILE_SIZE * 0.8)),
                ..default()
            },
            // Transform is corrected every frame by sync_agent_transforms
            // in viz/agent_renderer.rs — no world pos needed here.
            transform:  Transform::from_xyz(0.0, 0.0, 1.0),
            visibility: Visibility::default(),
        }
    }
}

/// Find the base tile position for a given team from WorldConfig::fixed.
/// Falls back to the agent start position if no base tile is defined.
fn find_base_tile(map: &WorldConfig, team_id: u8) -> Option<GridPos> {
    use crate::world::config::TileKind;
    map.fixed.iter().find_map(|f| {
        let matches = match f.tile {
            TileKind::BaseRed  | TileKind::Base => team_id == 0,
            TileKind::BaseBlue                  => team_id == 1,
            _                                   => false,
        };
        if matches { Some(GridPos::new(f.x as i32, f.y as i32)) } else { None }
    })
}

pub fn spawn_agents(mut commands: Commands, map: Res<WorldConfig>) {
    for (idx, cfg) in map.agents.iter().enumerate() {
        let team_id   = cfg.team.unwrap_or(0) as u8;
        let team      = Team(team_id);
        let brain     = AgentBrain(make_agent(cfg));
        let color     = team.color();
        let start_pos = GridPos::new(cfg.x, cfg.y);

        // SpawnPoint = base tile so respawn lands on the base.
        // Falls back to start pos if no matching base is defined in config.ron.
        let spawn_point = find_base_tile(&map, team_id).unwrap_or(start_pos);

        #[cfg(not(feature = "python"))]
        commands.spawn((
            AgentBundle::new(start_pos, spawn_point, brain, team, color),
            AgentConfigIndex(idx),
        ));

        #[cfg(feature = "python")]
        {
            let mut entity = commands.spawn((
                AgentBundle::new(start_pos, spawn_point, brain, team, color),
                AgentConfigIndex(idx),
            ));
            // Tag the first Blue (team=1) agent as the RL-controlled agent.
            // To support multiple Blue agents: add `rl_controlled: bool`
            // to AgentConfig in config.ron and check that flag here.
            if cfg.team == Some(1) {
                entity.insert(RlAgent);
            }
        }
    }
}