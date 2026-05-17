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
//   (spawn_agents, init_sim_config, etc.), and once during reset() after
//   injecting a RestartMessage. It is never called inside step().
//
// Threading: RlEnv is not Send. Python must run training on one thread,
// or spawn separate RlEnv instances per process (recommended for parallel
// envs — each process owns one App).

use bevy::prelude::*;

use crate::agent::systems::PendingAction;
use crate::sim::config::SimConfig;
use crate::sim::schedule::OnSimTick;
use crate::viz::restart::RestartMessage;
use super::marker::RlAgent;
use super::obs::{build_obs, OBS_SIZE};
use super::action::int_to_action;
use super::reward::{compute_reward, PrevAgentState};

// ── Public constants ──────────────────────────────────────────────────────────

pub use super::action::ACTION_SIZE;

// ── RlEnv ─────────────────────────────────────────────────────────────────────

pub struct RlEnv {
    app:        App,
    prev_state: PrevAgentState,
}

impl RlEnv {
    /// Build a headless Bevy App and run Startup to completion.
    /// This spawns agents, loads config.ron, initialises the grid, etc.
    pub fn new() -> Self {
        let mut app = build_headless_app();

        // Run Startup + first Update so all Startup systems complete.
        // fire_sim_tick will find ticks_due == 0 on this first update
        // (delta ≈ 0) — no sim ticks fire.
        app.update();

        let prev_state = PrevAgentState::read(app.world_mut());

        Self { app, prev_state }
    }

    /// Reset to a fresh episode. Returns the initial observation.
    ///
    /// Internally sends RestartMessage and calls app.update() once so
    /// restart_episode (exclusive system) runs and agent/item entities
    /// are respawned. A second update lets Added<AgentBrain> systems
    /// (label assignment, etc.) fire.
    pub fn reset(&mut self) -> Vec<f32> {
        {
            use crate::viz::restart::restart_episode;
            self.app.world_mut().write_message(RestartMessage);
            restart_episode(self.app.world_mut());
        }

        // Reset SimConfig cleanly (restart_episode already zeroed tick,
        // but be defensive here in case reset() is called mid-episode).
        {
            let mut cfg = self.app.world_mut().resource_mut::<SimConfig>();
            cfg.tick      = 0;
            cfg.game_over = false;
            cfg.paused    = false;
        }

        self.prev_state = PrevAgentState::read(self.app.world_mut());

        build_obs(self.app.world_mut())
    }

    /// Advance the simulation by exactly one tick.
    ///
    /// Returns (obs, reward, done):
    ///   obs    — Vec<f32> of length OBS_SIZE (53)
    ///   reward — scalar reward for this tick
    ///   done   — true when SimConfig::game_over is set
    pub fn step(&mut self, action: u32) -> (Vec<f32>, f32, bool) {
        // 1. Inject action onto the RlAgent entity.
        self.inject_action(action);

        // 2. Advance tick counter and check termination.
        let done = self.advance_tick();

        // 3. Fire exactly one sim tick.
        self.app.world_mut().run_schedule(OnSimTick);

        // 4. Compute reward from world state after the tick.
        let reward = compute_reward(self.app.world_mut(), &self.prev_state);

        // 5. Build observation.
        let obs = build_obs(self.app.world_mut());

        // 6. Snapshot state for next tick's reward delta.
        self.prev_state = PrevAgentState::read(self.app.world_mut());

        (obs, reward, done)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Write `action` as a PendingAction on the RlAgent entity.
    /// No-ops if RlAgent is dead (PendingAction is ignored by apply_actions
    /// when RespawnIn is present).
    fn inject_action(&mut self, action: u32) {
        let sim_action = int_to_action(action);

        let world = self.app.world_mut();

        let mut q = world.query_filtered::<&mut PendingAction, With<RlAgent>>();
        if let Ok(mut pending) = q.single_mut(world) {
            *pending = PendingAction(Some(sim_action));
        }
        // If RlAgent not found (e.g. between reset() calls), silently skip.
    }

    /// Increment SimConfig::tick and set game_over if the match ends.
    /// Returns the updated game_over flag.
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

// ── App builder ───────────────────────────────────────────────────────────────

/// Construct a minimal headless Bevy App with all required plugins.
/// Mirrors what main.rs does under `--features headless`, but without
/// any rendering or window plugins.
fn build_headless_app() -> App {
    let mut app = App::new();

    app.add_plugins(MinimalPlugins);

    // Core sim plugins — order must match main.rs headless branch.
    app.add_plugins((
        crate::world::plugin::WorldPlugin,
        crate::sim::plugin::SimPlugin,
        crate::agent::plugin::AgentPlugin,
    ));

    // RL marker registration — RlAgent must be added to the Blue agent
    // entity in spawn.rs. If it isn't yet, this is a no-op here; the
    // marker is read by obs/reward systems which return zeros when missing.

    app
}

// ── Observation size re-export for Python binding ─────────────────────────────

pub const OBS_DIM: usize = OBS_SIZE;