// src/rl/env.rs
//
// RlEnv wraps a headless Bevy App and exposes a gym-style step/reset
// interface for use from Python via PyO3.
//
// Tick control — why we bypass fire_sim_tick:
//   fire_sim_tick is timer-driven (TickTimer). Calling app.update() would
//   require real wall-clock time to elapse before any ticks fire, making
//   training speed nondeterministic. Instead, RlEnv owns the tick loop:
//
//     step() {
//       inject PendingAction onto RlAgent
//       increment SimConfig::tick, set game_over if needed
//       world.run_schedule(OnSimTick)      ← exactly one sim tick
//       read reward / obs / done
//     }
//
//   app.update() is called once during new() to run Startup systems
//   (spawn_agents, init_sim_config, etc.). It is never called inside step().
//
// Threading: RlEnv is not Send. Python must run training on one thread,
// or spawn separate RlEnv instances per process (recommended for parallel
// envs — each process owns one App).

use bevy::prelude::*;

use crate::agent::brain::AgentBrain;
use crate::agent::components::{
    Ammo, DeathCount, GoldCarried, GridPos, Hearts, KillCount, Score, SpawnPoint,
};
use crate::agent::registry::make_agent;
use crate::agent::systems::PendingAction;
use crate::factory::AgentConfigIndex;
use crate::item::Item;
use crate::item::spawner::{FreeTilePool, ItemSpawner};
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

// ── Public constants ──────────────────────────────────────────────────────────

pub use super::action::ACTION_SIZE;
pub const OBS_DIM: usize = OBS_SIZE;

// ── RlEnv ─────────────────────────────────────────────────────────────────────

pub struct RlEnv {
    app:        App,
    prev_state: PrevAgentState,
}

impl RlEnv {
    /// Build a headless Bevy App and run Startup to completion.
    pub fn new() -> Self {
        let mut app = build_headless_app();
        app.update();
        let prev_state = PrevAgentState::read(app.world_mut());
        Self { app, prev_state }
    }

    /// Reset to a fresh episode. Returns the initial observation.
    ///
    /// Performs a full in-process restart:
    ///   - despawns agents and items
    ///   - regenerates grid obstacles
    ///   - respawns agents (including RlAgent marker on Blue)
    ///   - resets SimConfig and TeamScore
    pub fn reset(&mut self) -> Vec<f32> {
        headless_restart(self.app.world_mut());

        self.prev_state = PrevAgentState::read(self.app.world_mut());
        build_obs(self.app.world_mut())
    }

    /// Advance the simulation by exactly one tick.
    ///
    /// Returns (obs, reward, done).
    pub fn step(&mut self, action: u32) -> (Vec<f32>, f32, bool) {
        self.inject_action(action);
        let done = self.advance_tick();
        self.app.world_mut().run_schedule(OnSimTick);
        let reward = compute_reward(self.app.world_mut(), &self.prev_state);
        let obs    = build_obs(self.app.world_mut());
        self.prev_state = PrevAgentState::read(self.app.world_mut());
        (obs, reward, done)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

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
//
// Mirrors viz::restart::restart_episode but without any UI steps
// (no EndScreen hiding, no ScoreboardRow despawn — those don't exist
// in a headless build).

fn headless_restart(world: &mut World) {
    info!("=== Headless episode restart ===");

    // 1. Reset SimConfig
    {
        let mut cfg = world.resource_mut::<SimConfig>();
        cfg.tick      = 0;
        cfg.game_over = false;
        cfg.paused    = false;
    }

    // 2. Reset TeamScore
    *world.resource_mut::<TeamScore>() = TeamScore::default();

    // 3. Despawn all agents
    let agents: Vec<Entity> = world
        .query_filtered::<Entity, With<AgentBrain>>()
        .iter(world)
        .collect();
    for e in agents { world.despawn(e); }

    // 4. Despawn all items
    let items: Vec<Entity> = world
        .query_filtered::<Entity, With<Item>>()
        .iter(world)
        .collect();
    for e in items { world.despawn(e); }

    // 5. Reset grid
    {
        let map = world.resource::<WorldConfig>().clone();
        let mut grid = world.resource_mut::<Grid>();
        apply_fixed_tiles(&map, &mut grid);
        regenerate_obstacles(&map, &mut grid);
    }

    // 6. Rebuild FreeTilePool
    {
        let pool = FreeTilePool::build(world.resource::<Grid>());
        world.insert_resource(pool);
    }

    // 7. Reset ItemSpawner
    {
        let map = world.resource::<WorldConfig>().clone();
        world.insert_resource(ItemSpawner::from_map_config(&map));
    }

    // 8. Respawn agents — tag Blue (team=1) with RlAgent
    {
        let map = world.resource::<WorldConfig>().clone();
        for (idx, cfg) in map.agents.iter().enumerate() {
            let team  = Team(cfg.team.unwrap_or(0) as u8);
            let brain = AgentBrain(make_agent(cfg));
            let color = team.color();
            let pos   = GridPos::new(cfg.x, cfg.y);

            let mut cmd = world.spawn((
                pos,
                SpawnPoint(pos),
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