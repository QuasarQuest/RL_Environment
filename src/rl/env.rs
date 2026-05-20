// src/rl/env.rs
//
// Headless Bevy environment for RL training.
//
// Config path flow:
//   Python: atb.PyRlEnv("assets/world/config_stage1.ron")
//     → pyo3.rs: RlEnv::new(path)
//       → env.rs: build_headless_app(path) inserts ConfigPath resource
//         → world/plugin.rs: load_map reads ConfigPath → WorldConfig::load()
//
// This means every stage config is fully supported with no code changes.

use bevy::prelude::*;

use crate::agent::brain::AgentBrain;
use crate::agent::systems::PendingAction;
use crate::item::Item;
use crate::item::spawner::FreeTilePool;
use crate::sim::config::SimConfig;
use crate::sim::schedule::OnSimTick;
use crate::team::{Team, TeamScore};
use crate::world::config::WorldConfig;
use crate::world::Grid;
use crate::world::plugin::{rebuild_grid, ConfigPath};
use crate::config as global;
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
    /// Create a new headless environment loading the given config file.
    pub fn new(config_path: String) -> Self {
        let mut app = build_headless_app(config_path);
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

    // ── Private ───────────────────────────────────────────────────────────────

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

// ── Headless episode restart ──────────────────────────────────────────────────

fn headless_restart(world: &mut World) {
    info!("=== Headless episode restart ===");

    // Reset sim state
    {
        let mut sim = world.resource_mut::<SimConfig>();
        sim.tick      = 0;
        sim.game_over = false;
        sim.paused    = false;
    }
    *world.resource_mut::<TeamScore>() = TeamScore::default();

    // Despawn all agents
    let agents: Vec<Entity> = world
        .query_filtered::<Entity, With<AgentBrain>>()
        .iter(world)
        .collect();
    for e in agents { world.despawn(e); }

    // Despawn all items
    let items: Vec<Entity> = world
        .query_filtered::<Entity, With<Item>>()
        .iter(world)
        .collect();
    for e in items { world.despawn(e); }

    // Rebuild grid (bases, safe zones, fresh obstacles)
    let layout = {
        let cfg = world.resource::<WorldConfig>().clone();
        let mut grid = world.resource_mut::<Grid>();
        rebuild_grid(&cfg, &mut grid)
    };

    // Rebuild FreeTilePool
    {
        let pool = FreeTilePool::build(world.resource::<Grid>());
        world.insert_resource(pool);
    }

    // Respawn items
    {
        let cfg = world.resource::<WorldConfig>().clone();
        let grid = world.resource::<Grid>().clone();
        // We need Commands — use world.spawn() directly for items
        let free_count = crate::world::layout::count_free_tiles(
            &(0..grid.height).flat_map(|y| {
                (0..grid.width).map(move |x| {
                    grid.get(x as i32, y as i32).unwrap_or(crate::world::tile::Tile::Free)
                })
            }).collect::<Vec<_>>()
        );
        let item_cfgs = crate::world::layout::item_configs_for_free_count(&cfg, free_count);
        for item_cfg in &item_cfgs {
            let initial = (item_cfg.max_on_map / 2).max(1);
            let mut placed   = 0usize;
            let mut attempts = 0usize;
            while placed < initial && attempts < initial * 200 {
                attempts += 1;
                let x = rand::random_range(0..cfg.width  as i32);
                let y = rand::random_range(0..cfg.height as i32);
                if grid.get(x, y) == Some(crate::world::tile::Tile::Free) {
                    world.spawn((
                        Item { kind: item_cfg.kind },
                        GridPos::new(x, y),
                        Sprite {
                            color:       item_cfg.kind.color(),
                            custom_size: Some(Vec2::splat(global::TILE_SIZE * 0.6)),
                            ..default()
                        },
                        Transform::from_xyz(0.0, 0.0, item_cfg.kind.z_layer()),
                        Visibility::default(),
                    ));
                    placed += 1;
                }
            }
        }
    }

    // Respawn agents — spawn_agent_world is the single source of truth
    for (idx, resolved) in layout.agents.iter().enumerate() {
        crate::agent::spawn::spawn_agent_world(world, resolved, &layout, idx);
    }

    info!("Headless restart complete — tick 0");
}

// ── App builder ───────────────────────────────────────────────────────────────

fn build_headless_app(config_path: String) -> App {
    let mut app = App::new();

    // Inject config path BEFORE WorldPlugin runs
    app.insert_resource(ConfigPath(config_path));

    app.add_plugins(MinimalPlugins);
    app.add_plugins((
        crate::world::plugin::WorldPlugin,
        crate::sim::plugin::SimPlugin,
        crate::item::ItemPlugin,
        crate::agent::plugin::AgentPlugin,
    ));
    app
}