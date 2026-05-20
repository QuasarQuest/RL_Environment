// src/rl/env.rs
//
// Headless Bevy environment for RL training.
//
// Config path flow:
//   Python: atb.PyRlEnv("assets/world/config.ron")
//     → pyo3.rs: RlEnv::new(path)
//       → env.rs: build_headless_app(path) inserts ConfigPath resource
//         → world/plugin.rs: load_map reads ConfigPath → WorldConfig::load()
//
// Observation contract:
//   The env returns a flat Vec<f32> of length OBS_TOTAL. The Python side
//   reshapes it to (C, H, W) using OBS_SHAPE. We deliberately do the
//   reshape in Python (zero-copy view) rather than constructing a 3D array
//   on the Rust side — keeps the PyO3 boundary boring.

use bevy::prelude::*;

use crate::agent::brain::AgentBrain;
use crate::agent::systems::PendingAction;
use crate::item::Item;
use crate::item::spawner::FreeTilePool;
use crate::sim::config::SimConfig;
use crate::sim::schedule::OnSimTick;
use crate::team::TeamScore;
use crate::world::config::WorldConfig;
use crate::world::Grid;
use crate::world::plugin::{rebuild_grid, ConfigPath};
use crate::config as global;
use crate::world::coords::GridPos;
use super::marker::RlAgent;
use super::obs::{build_obs_grid, OBS_TOTAL};
use super::action::int_to_action;
use super::reward::{compute_reward, PrevState};

pub use super::action::ACTION_SIZE;
pub use super::obs::{OBS_SHAPE, OBS_CHANNELS, OBS_CROP_SIZE};

/// Total floats in one observation. Python reshapes this to OBS_SHAPE.
pub const OBS_DIM: usize = OBS_TOTAL;

// ── RlEnv ─────────────────────────────────────────────────────────────────────

pub struct RlEnv {
    app:        App,
    prev_state: PrevState,
}

impl RlEnv {
    /// Create a new headless environment loading the given config file.
    pub fn new(config_path: String) -> Self {
        let mut app = build_headless_app(config_path);
        app.update();
        let prev_state = PrevState::read(app.world_mut());
        Self { app, prev_state }
    }

    pub fn reset(&mut self) -> Vec<f32> {
        headless_restart(self.app.world_mut());
        self.prev_state = PrevState::read(self.app.world_mut());
        build_obs_grid(self.app.world_mut())
    }

    pub fn step(&mut self, action: u32) -> (Vec<f32>, f32, bool) {
        self.inject_action(action);
        let done   = self.advance_tick();
        self.app.world_mut().run_schedule(OnSimTick);
        let reward = compute_reward(self.app.world_mut(), &self.prev_state);
        let obs    = build_obs_grid(self.app.world_mut());
        self.prev_state = PrevState::read(self.app.world_mut());
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
        let cfg  = world.resource::<WorldConfig>().clone();
        let grid = world.resource::<Grid>().clone();

        // Count free tiles via Grid::iter (already exists on the Grid type).
        // The previous nested-closure version moved `grid` into the inner
        // FnMut, which the borrow checker rejects because the outer flat_map
        // closure may run multiple times.
        let free_count = grid.iter()
            .filter(|(_, _, tile)| *tile == crate::world::tile::Tile::Free)
            .count();

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

    // Tag the first (team-0) agent with the RlAgent marker. The marker is a
    // zero-size component that obs / reward / action-injection filter on.
    // We tag here rather than relying on `strategy: Rl` in config.ron because
    // the Python entry point owns RL-ness — the world config shouldn't need
    // to know whether it's being driven by a strategy or by an external policy.
    {
        let agent_e: Option<Entity> = world
            .query_filtered::<(Entity, &crate::team::Team), With<AgentBrain>>()
            .iter(world)
            .find(|(_, team)| team.0 == 0)
            .map(|(e, _)| e);
        match agent_e {
            Some(e) => {
                world.entity_mut(e).insert(RlAgent);
                info!("Tagged entity {:?} with RlAgent marker", e);
            }
            None => warn!("No team-0 agent found to tag as RlAgent"),
        }
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