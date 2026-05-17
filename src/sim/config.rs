// src/sim/config.rs

use bevy::prelude::*;
use crate::config;

#[derive(Resource)]
pub struct SimConfig {
    pub ticks_per_second:     f32,
    pub paused:               bool,
    pub tick:                 u64,
    pub match_duration_ticks: u64,
    pub game_over:            bool,
    /// Available speed steps — loaded from WorldConfig::sim_speeds at startup.
    pub available_speeds:     Vec<f32>,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            ticks_per_second:     config::DEFAULT_TICKS_PER_SECOND,
            paused:               false,
            tick:                 0,
            match_duration_ticks: 10_000,
            game_over:            false,
            available_speeds:     vec![1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0],
        }
    }
}

impl SimConfig {
    pub fn remaining_display(&self) -> String {
        format!("{} / {}", self.tick, self.match_duration_ticks)
    }

    /// Index of the nearest speed step to the current ticks_per_second.
    pub fn speed_index(&self) -> usize {
        self.available_speeds
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (**a - self.ticks_per_second).abs()
                    .partial_cmp(&(**b - self.ticks_per_second).abs())
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}