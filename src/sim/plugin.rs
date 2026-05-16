// src/sim/plugin.rs

use bevy::prelude::*;
use super::config::{SimConfig, AVAILABLE_SPEEDS};
use super::schedule::OnSimTick;
use super::timer::TickTimer;
use crate::world::map_config::MapConfig;

/// Public system-set label so other plugins can order after sim writes.
/// NOTE: exclusive systems (fn(&mut World)) cannot belong to a SystemSet;
/// fire_sim_tick is scheduled separately with .after(SimSystems).
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SimSystems;

// ── Systems ───────────────────────────────────────────────────────────────────

/// Reads match_duration_ticks from the map asset into SimConfig.
/// Runs at Startup — MapConfig is inserted during PreStartup by WorldPlugin,
/// so it is guaranteed to exist by the time Startup runs.
fn init_sim_config(map: Res<MapConfig>, mut cfg: ResMut<SimConfig>) {
    cfg.match_duration_ticks = map.match_duration_ticks;
}

fn handle_input(
    keys:      Res<ButtonInput<KeyCode>>,
    mut cfg:   ResMut<SimConfig>,
    mut timer: ResMut<TickTimer>,
) {
    if keys.just_pressed(KeyCode::Space) {
        cfg.paused = !cfg.paused;
    }

    let mut speed_changed = false;

    if keys.just_pressed(KeyCode::KeyF) {
        let idx = AVAILABLE_SPEEDS.iter()
            .position(|&s| (s - cfg.ticks_per_second).abs() < f32::EPSILON)
            .unwrap_or(0);
        cfg.ticks_per_second = AVAILABLE_SPEEDS[(idx + 1).min(AVAILABLE_SPEEDS.len() - 1)];
        speed_changed = true;
    }
    if keys.just_pressed(KeyCode::KeyS) {
        let idx = AVAILABLE_SPEEDS.iter()
            .position(|&s| (s - cfg.ticks_per_second).abs() < f32::EPSILON)
            .unwrap_or(0);
        cfg.ticks_per_second = AVAILABLE_SPEEDS[idx.saturating_sub(1)];
        speed_changed = true;
    }

    if speed_changed {
        // Reset accumulator when speed changes to avoid a burst of catch-up ticks.
        timer.0 = Timer::from_seconds(1.0 / cfg.ticks_per_second, TimerMode::Repeating);
    }
}

/// Exclusive system — fires as many ticks as the accumulated wall-clock delta
/// covers at the current ticks_per_second.
///
/// Episode length is match_duration_ticks — speed-independent.
/// At 10 tps a 10 000-tick episode takes ~16 min real time.
/// At 200 tps the same episode takes ~50 s — ideal for RL throughput.
pub fn fire_sim_tick(world: &mut World) {
    let (paused, game_over) = {
        let cfg = world.resource::<SimConfig>();
        (cfg.paused, cfg.game_over)
    };

    if paused || game_over {
        // Still tick the timer so it doesn't accumulate while paused.
        let delta = world.resource::<Time>().delta();
        world.resource_mut::<TickTimer>().0.tick(delta);
        return;
    }

    // Accumulate wall time into the timer and count how many ticks are due.
    let ticks_due: u32 = {
        let delta = world.resource::<Time>().delta();
        let mut timer = world.resource_mut::<TickTimer>();
        timer.0.tick(delta);
        timer.0.times_finished_this_tick()
    };

    // Cap ticks per frame to avoid a death-spiral when the sim is too slow
    // to process all ticks within a frame (e.g. very heavy agent logic).
    // 200 is enough headroom for 100 tps at 2 fps — adjust if needed.
    let ticks_to_fire = ticks_due.min(200);

    for _ in 0..ticks_to_fire {
        let game_over = {
            let mut cfg = world.resource_mut::<SimConfig>();
            cfg.tick += 1;
            if cfg.tick >= cfg.match_duration_ticks {
                cfg.game_over = true;
                info!("Match over! Total ticks: {}", cfg.tick);
            }
            cfg.game_over
        };

        world.run_schedule(OnSimTick);

        if game_over { break; }
    }
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub struct SimPlugin;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<SimConfig>()
            .init_resource::<TickTimer>()
            .init_schedule(OnSimTick)
            .add_systems(Startup, init_sim_config)
            .add_systems(Update, handle_input.in_set(SimSystems))
            .add_systems(Update, fire_sim_tick.after(SimSystems));
    }
}