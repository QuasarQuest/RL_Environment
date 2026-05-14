// src/sim/config.rs

use bevy::prelude::*;
use crate::config;

pub const AVAILABLE_SPEEDS: &[f32] = &[1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0];

#[derive(Resource)]
pub struct SimConfig {
    pub ticks_per_second: f32,
    pub paused:           bool,
    pub tick:             u64,

    // ── Match timer ───────────────────────────────────────────────────────────
    /// Total match duration in real seconds (e.g. 90.0 for 1:30).
    pub match_duration_secs: f32,
    /// Real seconds elapsed since match start (unaffected by sim speed).
    pub elapsed_secs:        f32,
    /// Set to true when elapsed_secs >= match_duration_secs.
    /// When true, fire_sim_tick stops running — terminal state for RL.
    pub game_over:           bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            ticks_per_second:    config::DEFAULT_TICKS_PER_SECOND,
            paused:              false,
            tick:                0,
            match_duration_secs: config::MATCH_DURATION_SECS,
            elapsed_secs:        0.0,
            game_over:           false,
        }
    }
}

impl SimConfig {
    /// Remaining time in seconds, clamped to 0.
    pub fn remaining_secs(&self) -> f32 {
        (self.match_duration_secs - self.elapsed_secs).max(0.0)
    }

    /// Remaining time formatted as "M:SS".
    pub fn remaining_display(&self) -> String {
        let secs = self.remaining_secs() as u32;
        format!("{}:{:02}", secs / 60, secs % 60)
    }
}