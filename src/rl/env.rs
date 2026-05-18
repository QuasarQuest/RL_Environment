// src/rl/env.rs

use bevy::prelude::*;

use crate::agent::brain::AgentBrain;
use crate::agent::components::{
    Ammo, DeathCount, GoldCarried, GridPos, Hearts, KillCount, Score, SpawnPoint,
};
use crate::agent::registry::make_agent;
use crate::agent::systems::PendingAction;
use crate::factory::AgentConfigIndex;
use crate::item::Item;
use crate::item::spawner::FreeTilePool;
use crate::sim::config::SimConfig;
use crate::sim::schedule::OnSimTick;
use crate::team::{Team, TeamScore};
use crate::world::config::WorldConfig;
use crate::world::Grid;
use crate::world::plugin::{apply_fixed_tiles, regenerate_obstacles};
use crate::config;
use super::marker::RlAgent;
use super::obs::{build_obs, OBS_SIZE};
use super::action::int_to_action;
use super::reward::{compute_reward, PrevAgentState};

pub use super::action::ACTION_SIZE;
pub const OBS_DIM: usize = OBS_SIZE;

// ── RlEnv ─────────────────────────────────────────────────────────────────────

pub struct RlEnv {
    app:        App,
    prev_state: PrevAgentState,
}

impl RlEnv {
    pub fn new() -> Self {
        let mut app = build_headless_app();
        app.update();
        let prev_state = PrevAgentState::read(app.world_mut());
        Self { app, prev_state }
    }

    pub fn reset(&mut self) -> Vec<f32> {
        headless_restart(self.app.world_mut());
        self.prev_state = PrevAgentState::read(self.app.world_mut());
        build_obs(self.app.world_mut())
    }

    pub fn step(&mut self, action: u32) -> (Vec<f32>, f32, bool) {
        self.inject_action(action);
        let done   = self.advance_tick();
        self.app.world_mut().run_schedule(OnSimTick);
        let reward = compute_reward(self.app.world_mut(), &self.prev_state);
        let obs    = build_obs(self.app.world_mut());
        self.prev_state = PrevAgentState::read(self.app.world_mut());
        (obs, reward, done)
    }

    fn inject_action(&mut self, action: u32) {
        let sim_action = int_to_action(action);
        let world = self.app.world_mut();
        let mut q = world.query_filtered::<&mut PendingAction, With<RlAgent>>();
        if let Ok(mut pending) = q.single_mut(world) {
            *pending = PendingAction(Some(sim_action));
        }
    }

    fn advance_tick(&mut self) -> bool {
        let mut cfg = self.app.world_mut().resource_mut::<SimConfig>();
        cfg.tick += 1;
        if cfg.tick >= cfg.match_duration_ticks {
            cfg.game_over = true;
            info!("RL episode complete — tick {}", cfg.tick);
        }
        cfg.game_over
    }
}

// ── Headless restart ──────────────────────────────────────────────────────────

fn headless_restart(world: &mut World) {
    info!("=== Headless episode restart ===");

    {
        let mut cfg = world.resource_mut::<SimConfig>();
        cfg.tick      = 0;
        cfg.game_over = false;
        cfg.paused    = false;
    }

    *world.resource_mut::<TeamScore>() = TeamScore::default();

    // Despawn agents
    let agents: Vec<Entity> = world
        .query_filtered::<Entity, With<AgentBrain>>()
        .iter(world)
        .collect();
    for e in agents { world.despawn(e); }

    // Despawn items
    let items: Vec<Entity> = world
        .query_filtered::<Entity, With<Item>>()
        .iter(world)
        .collect();
    for e in items { world.despawn(e); }

    // Reset grid
    {
        let map = world.resource::<WorldConfig>().clone();
        let mut grid = world.resource_mut::<Grid>();
        apply_fixed_tiles(&map, &mut grid);
        regenerate_obstacles(&map, &mut grid);
    }

    // Rebuild FreeTilePool
    {
        let pool = FreeTilePool::build(world.resource::<Grid>());
        world.insert_resource(pool);
    }

    // Respawn agents — tag Blue (team=1) with RlAgent
    {
        let map = world.resource::<WorldConfig>().clone();
        for (idx, cfg) in map.agents.iter().enumerate() {
            let team_id     = cfg.team.unwrap_or(0) as u8;
            let team        = Team(team_id);
            let brain       = AgentBrain(make_agent(cfg));
            let color       = team.color();
            let start_pos   = GridPos::new(cfg.x, cfg.y);
            let spawn_point = find_base_tile(&map, team_id).unwrap_or(start_pos);

            let mut cmd = world.spawn((
                start_pos,
                SpawnPoint(spawn_point),
                Hearts::default(),
                Ammo::default(),
                GoldCarried::default(),
                Score::default(),
                KillCount::default(),
                DeathCount::default(),
                brain,
                team,
                PendingAction::default(),
                Sprite {
                    color,
                    custom_size: Some(Vec2::splat(config::TILE_SIZE * 0.8)),
                    ..default()
                },
                Transform::from_xyz(0.0, 0.0, 1.0),
                Visibility::default(),
                AgentConfigIndex(idx),
            ));

            if cfg.team == Some(1) {
                cmd.insert(RlAgent);
            }
        }
    }

    info!("Headless restart complete — tick 0.");
}

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

// ── App builder ───────────────────────────────────────────────────────────────

fn build_headless_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins((
        crate::world::plugin::WorldPlugin,
        crate::sim::plugin::SimPlugin,
        crate::item::ItemPlugin,
        crate::agent::plugin::AgentPlugin,
    ));
    app
}