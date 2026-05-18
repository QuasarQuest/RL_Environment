// src/sim/plugin.rs

use bevy::prelude::*;
use super::config::SimConfig;
use super::schedule::OnSimTick;
use super::timer::TickTimer;
use crate::world::config::WorldConfig;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SimSystems;

fn init_sim_config(
    map:       Res<WorldConfig>,
    mut cfg:   ResMut<SimConfig>,
    mut timer: ResMut<TickTimer>,
) {
    cfg.match_duration_ticks = map.match_duration_ticks;
    cfg.ticks_per_second     = map.sim_speed;
    cfg.available_speeds     = map.sim_speeds.clone();
    timer.0 = Timer::from_seconds(1.0 / map.sim_speed, TimerMode::Repeating);
}

#[cfg(not(feature = "headless"))]
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
        let idx = cfg.speed_index();
        cfg.ticks_per_second = cfg.available_speeds[(idx + 1).min(cfg.available_speeds.len() - 1)];
        speed_changed = true;
    }
    if keys.just_pressed(KeyCode::KeyS) {
        let idx = cfg.speed_index();
        cfg.ticks_per_second = cfg.available_speeds[idx.saturating_sub(1)];
        speed_changed = true;
    }

    if speed_changed {
        timer.0 = Timer::from_seconds(1.0 / cfg.ticks_per_second, TimerMode::Repeating);
    }
}

// In headless mode RlEnv drives ticks directly via world.run_schedule(OnSimTick).
// fire_sim_tick and handle_input are timer/input driven and not needed.
#[cfg(not(feature = "headless"))]
pub fn fire_sim_tick(world: &mut World) {
    let (paused, game_over) = {
        let cfg = world.resource::<SimConfig>();
        (cfg.paused, cfg.game_over)
    };

    if paused || game_over {
        let delta = world.resource::<Time>().delta();
        world.resource_mut::<TickTimer>().0.tick(delta);
        return;
    }

    let ticks_due: u32 = {
        let delta = world.resource::<Time>().delta();
        let mut timer = world.resource_mut::<TickTimer>();
        timer.0.tick(delta);
        timer.0.times_finished_this_tick()
    };

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

pub struct SimPlugin;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<SimConfig>()
            .init_resource::<TickTimer>()
            .init_schedule(OnSimTick)
            .add_systems(Startup, init_sim_config);

        #[cfg(not(feature = "headless"))]
        app
            .add_systems(Update, handle_input.in_set(SimSystems))
            .add_systems(Update, fire_sim_tick.after(SimSystems));
    }
}