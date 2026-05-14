// src/sim/plugin.rs

use bevy::prelude::*;
use super::config::{SimConfig, AVAILABLE_SPEEDS};
use super::schedule::OnSimTick;
use super::timer::TickTimer;

fn advance_match_timer(
    time:    Res<Time>,
    mut cfg: ResMut<SimConfig>,
) {
    if cfg.game_over || cfg.paused { return; }
    cfg.elapsed_secs += time.delta_secs();
    if cfg.elapsed_secs >= cfg.match_duration_secs {
        cfg.elapsed_secs = cfg.match_duration_secs;
        cfg.game_over    = true;
        info!("Match over!");
    }
}

fn fire_sim_tick(world: &mut World) {
    let delta = world.resource::<Time>().delta();

    let should_tick = {
        let cfg = world.resource::<SimConfig>();
        if cfg.paused || cfg.game_over {
            false
        } else {
            world.resource_mut::<TickTimer>().0.tick(delta).just_finished()
        }
    };

    if should_tick {
        world.resource_mut::<SimConfig>().tick += 1;
        world.run_schedule(OnSimTick);
    }
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
        timer.0 = Timer::from_seconds(1.0 / cfg.ticks_per_second, TimerMode::Repeating);
    }
}

pub struct SimPlugin;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimConfig>()
            .init_resource::<TickTimer>()
            .init_schedule(OnSimTick)
            .add_systems(Update, handle_input)
            .add_systems(Update, advance_match_timer)
            .add_systems(Update, fire_sim_tick);
    }
}