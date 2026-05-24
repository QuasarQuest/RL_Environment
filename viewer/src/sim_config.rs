use bevy::prelude::*;
use atb::config;

#[derive(Resource)]
pub struct SimConfig {
    pub paused:           bool,
    pub ticks_per_second: f32,
    pub available_speeds: Vec<f32>,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            paused:           false,
            ticks_per_second: config::DEFAULT_TICKS_PER_SECOND,
            available_speeds: vec![1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0],
        }
    }
}

impl SimConfig {
    pub fn speed_index(&self) -> usize {
        self.available_speeds
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (**a - self.ticks_per_second)
                    .abs()
                    .partial_cmp(&(**b - self.ticks_per_second).abs())
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

#[derive(Resource)]
pub struct TickTimer(pub Timer);

impl Default for TickTimer {
    fn default() -> Self {
        Self(Timer::from_seconds(
            1.0 / config::DEFAULT_TICKS_PER_SECOND,
            TimerMode::Repeating,
        ))
    }
}
