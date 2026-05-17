// src/sim/config.rs

use bevy::prelude::*;
use crate::config;

pub const AVAILABLE_SPEEDS: &[f32] = &[1.0, 2.0, 5.0, 10.0, 25.0, 50.0,
    100.0, 500.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0];

#[derive(Resource)]
pub struct SimConfig {
    pub ticks_per_second:     f32,
    pub paused:               bool,
    pub tick:                 u64,
    /// Episode length in sim-ticks — set from MapConfig at startup.
    /// Speed-independent: RL always gets exactly this many steps per episode.
    pub match_duration_ticks: u64,
    /// Set to true when tick >= match_duration_ticks.
    /// When true, fire_sim_tick stops — terminal state for RL.
    pub game_over:            bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            ticks_per_second:     config::DEFAULT_TICKS_PER_SECOND,
            paused:               false,
            tick:                 0,
            match_duration_ticks: 10_000, // overwritten by init_sim_config at PreStartup
            game_over:            false,
        }
    }
}

impl SimConfig {
    /// Progress displayed in the HUD as "tick / total".
    pub fn remaining_display(&self) -> String {
        format!("{} / {}", self.tick, self.match_duration_ticks)
    }
}