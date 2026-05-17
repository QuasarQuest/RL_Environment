// src/viz/hud/systems.rs

use bevy::prelude::*;
use crate::sim::config::SimConfig;
use crate::sim::timer::TickTimer;
use crate::team::{Team, TeamScore};
use crate::style::color::{GOLD_500, GOLD_800, GREEN_400, GREEN_500};
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

/// Format speed for the HUD label — no decimal for whole numbers.
fn format_speed(tps: f32) -> String {
    if tps >= 1000.0 {
        format!("{}k", (tps as u32) / 1000)
    } else {
        format!("{}x", tps as u32)
    }
}
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
            let idx = cfg.speed_index();
            cfg.ticks_per_second = cfg.available_speeds[idx.saturating_sub(1)];
            changed = true;
        }
    }
    for interaction in increase_q.iter() {
        if *interaction == Interaction::Pressed {
            let idx = cfg.speed_index();
            cfg.ticks_per_second = cfg.available_speeds[(idx + 1).min(cfg.available_speeds.len() - 1)];
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
        let label = format_speed(cfg.ticks_per_second);
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
        ("▶", GREEN_400, GREEN_500)  // paused → show play  (green = go)
    } else {
        ("⏸", GOLD_500,  GOLD_800)   // running → show pause (gold = caution)
    };

    for (mut text, mut color) in text_q.iter_mut() {
        *text  = Text::new(label);
        *color = TextColor(text_color);
    }
    for mut bg in bg_q.iter_mut() {
        bg.0 = bg_color;
    }
}