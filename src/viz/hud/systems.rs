// src/viz/hud/systems.rs

use bevy::prelude::*;
use crate::sim::config::{SimConfig, AVAILABLE_SPEEDS};
use crate::sim::timer::TickTimer;
use crate::team::{Team, TeamScore};
use super::components::{
    TickLabelMarker, TimeLabelMarker, TeamScoreMarker,
    SpeedDecreaseButton, SpeedIncreaseButton, SpeedResetButton,
    CurrentSpeedLabel, PauseButtonMarker, PauseButtonText,
};

// ── Label updates ─────────────────────────────────────────────────────────────

pub fn update_tick_label(
    cfg:       Res<SimConfig>,
    mut query: Query<&mut Text, With<TickLabelMarker>>,
) {
    if !cfg.is_changed() { return; }
    for mut text in query.iter_mut() {
        *text = Text::new(format!("{}", cfg.tick));
    }
}

pub fn update_time_label(
    cfg:       Res<SimConfig>,
    mut query: Query<&mut Text, With<TimeLabelMarker>>,
) {
    if !cfg.is_changed() { return; }
    for mut text in query.iter_mut() {
        *text = Text::new(cfg.remaining_display());
    }
}

pub fn update_team_scores(
    team_score: Res<TeamScore>,
    mut query:  Query<(&mut Text, &TeamScoreMarker)>,
) {
    if !team_score.is_changed() { return; }
    for (mut text, marker) in query.iter_mut() {
        *text = Text::new(team_score.get(Team(marker.0)).to_string());
    }
}

// ── Speed controls ────────────────────────────────────────────────────────────

pub fn handle_speed_buttons(
    mut cfg:         ResMut<SimConfig>,
    mut timer:       ResMut<TickTimer>,
    decrease_q:      Query<&Interaction, (Changed<Interaction>, With<SpeedDecreaseButton>)>,
    increase_q:      Query<&Interaction, (Changed<Interaction>, With<SpeedIncreaseButton>)>,
    reset_q:         Query<&Interaction, (Changed<Interaction>, With<SpeedResetButton>)>,
    mut speed_label: Query<&mut Text, With<CurrentSpeedLabel>>,
) {
    let mut changed = false;

    for interaction in decrease_q.iter() {
        if *interaction == Interaction::Pressed {
            let idx = AVAILABLE_SPEEDS.iter()
                .position(|&s| (s - cfg.ticks_per_second).abs() < f32::EPSILON)
                .unwrap_or(0);
            cfg.ticks_per_second = AVAILABLE_SPEEDS[idx.saturating_sub(1)];
            changed = true;
        }
    }
    for interaction in increase_q.iter() {
        if *interaction == Interaction::Pressed {
            let idx = AVAILABLE_SPEEDS.iter()
                .position(|&s| (s - cfg.ticks_per_second).abs() < f32::EPSILON)
                .unwrap_or(0);
            cfg.ticks_per_second = AVAILABLE_SPEEDS[(idx + 1).min(AVAILABLE_SPEEDS.len() - 1)];
            changed = true;
        }
    }
    for interaction in reset_q.iter() {
        if *interaction == Interaction::Pressed {
            cfg.ticks_per_second = crate::config::DEFAULT_TICKS_PER_SECOND;
            changed = true;
        }
    }

    if changed {
        timer.0 = Timer::from_seconds(1.0 / cfg.ticks_per_second, TimerMode::Repeating);
        let label = format!("{}x", cfg.ticks_per_second as u32);
        for mut text in speed_label.iter_mut() {
            *text = Text::new(&label);
        }
    }
}

// ── Pause button ──────────────────────────────────────────────────────────────
//
// Toggles cfg.paused on click. Visual sync (text + bg) is handled separately
// by sync_pause_visuals so Space key toggles also update the button correctly.

pub fn handle_pause_button(
    mut cfg: ResMut<SimConfig>,
    btn_q:   Query<&Interaction, (Changed<Interaction>, With<PauseButtonMarker>)>,
) {
    for interaction in btn_q.iter() {
        if *interaction == Interaction::Pressed {
            cfg.paused = !cfg.paused;
        }
    }
}

/// Syncs pause button visuals to SimConfig::paused every frame it changes.
/// Reacts to both the button click and the Space key (handled in sim/plugin.rs).
pub fn sync_pause_visuals(
    cfg:        Res<SimConfig>,
    mut text_q: Query<(&mut Text, &mut TextColor), With<PauseButtonText>>,
    mut bg_q:   Query<&mut BackgroundColor, With<PauseButtonMarker>>,
) {
    if !cfg.is_changed() { return; }

    let (label, text_color, bg_color) = if cfg.paused {
        ("|>", Color::srgb(0.95, 0.78, 0.20), Color::srgb(0.50, 0.35, 0.05))
    } else {
        ("||", Color::srgb(0.40, 0.90, 0.55), Color::srgb(0.12, 0.42, 0.24))
    };

    for (mut text, mut color) in text_q.iter_mut() {
        *text  = Text::new(label);
        *color = TextColor(text_color);
    }
    for mut bg in bg_q.iter_mut() {
        bg.0 = bg_color;
    }
}