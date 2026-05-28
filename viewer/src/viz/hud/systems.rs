use bevy::prelude::*;
use crate::sim_bridge::SimBridge;
use crate::sim_config::{SimConfig, TickTimer};
use crate::style::color::{GOLD_500, GOLD_800, GREEN_400, GREEN_500};
use super::components::{
    TickLabelMarker, TimeLabelMarker, TeamScoreMarker,
    SpeedDecreaseButton, SpeedIncreaseButton,
    CurrentSpeedLabel, PauseButtonMarker, PauseButtonText,
    PolicyModeLabel,
};

pub fn update_tick_label(
    bridge:    Res<SimBridge>,
    mut query: Query<&mut Text, With<TickLabelMarker>>,
) {
    if !bridge.is_changed() { return; }
    for mut text in query.iter_mut() {
        *text = Text::new(format!("{}", bridge.tick()));
    }
}

pub fn update_time_label(
    bridge:    Res<SimBridge>,
    mut query: Query<&mut Text, With<TimeLabelMarker>>,
) {
    if !bridge.is_changed() { return; }
    for mut text in query.iter_mut() {
        *text = Text::new(bridge.remaining_display());
    }
}

pub fn update_team_scores(
    bridge:    Res<SimBridge>,
    mut query: Query<(&mut Text, &TeamScoreMarker)>,
) {
    if !bridge.is_changed() { return; }
    for (mut text, marker) in query.iter_mut() {
        *text = Text::new(bridge.team_score(marker.0).to_string());
    }
}

fn format_speed(tps: f32) -> String {
    if tps >= 1000.0 { format!("{}k", (tps as u32) / 1000) }
    else             { format!("{}x", tps as u32) }
}

pub fn handle_speed_buttons(
    mut cfg:         ResMut<SimConfig>,
    mut timer:       ResMut<TickTimer>,
    keys:            Res<ButtonInput<KeyCode>>,
    decrease_q:      Query<&Interaction, (Changed<Interaction>, With<SpeedDecreaseButton>)>,
    increase_q:      Query<&Interaction, (Changed<Interaction>, With<SpeedIncreaseButton>)>,
    mut speed_label: Query<&mut Text, With<CurrentSpeedLabel>>,
) {
    let mut changed = false;

    let want_decrease = keys.just_pressed(KeyCode::ArrowDown)
        || decrease_q.iter().any(|i| *i == Interaction::Pressed);
    let want_increase = keys.just_pressed(KeyCode::ArrowUp)
        || increase_q.iter().any(|i| *i == Interaction::Pressed);

    if want_decrease {
        let idx = cfg.speed_index();
        cfg.ticks_per_second = cfg.available_speeds[idx.saturating_sub(1)];
        changed = true;
    }
    if want_increase {
        let idx = cfg.speed_index();
        let max = cfg.available_speeds.len() - 1;
        cfg.ticks_per_second = cfg.available_speeds[(idx + 1).min(max)];
        changed = true;
    }

    if changed {
        timer.0 = Timer::from_seconds(1.0 / cfg.ticks_per_second, TimerMode::Repeating);
        let label = format_speed(cfg.ticks_per_second);
        for mut text in speed_label.iter_mut() {
            *text = Text::new(&label);
        }
    }
}

pub fn handle_policy_key(
    keys:       Res<ButtonInput<KeyCode>>,
    mut bridge: ResMut<SimBridge>,
) {
    if keys.just_pressed(KeyCode::KeyP) {
        bridge.mode = bridge.mode.next();
        info!("Policy mode → {:?}", bridge.mode);
    }
}

pub fn sync_policy_mode_label(
    bridge:    Res<SimBridge>,
    mut query: Query<&mut Text, With<PolicyModeLabel>>,
) {
    if !bridge.is_changed() { return; }
    for mut text in query.iter_mut() {
        *text = Text::new(bridge.mode.label());
    }
}

pub fn handle_pause_button(
    mut cfg: ResMut<SimConfig>,
    btn_q:   Query<&Interaction, (Changed<Interaction>, With<PauseButtonMarker>)>,
    keys:    Res<ButtonInput<KeyCode>>,
) {
    for interaction in btn_q.iter() {
        if *interaction == Interaction::Pressed { cfg.paused = !cfg.paused; }
    }
    if keys.just_pressed(KeyCode::Space) { cfg.paused = !cfg.paused; }
}

pub fn sync_pause_visuals(
    cfg:        Res<SimConfig>,
    mut text_q: Query<(&mut Text, &mut TextColor), With<PauseButtonText>>,
    mut bg_q:   Query<&mut BackgroundColor, With<PauseButtonMarker>>,
) {
    if !cfg.is_changed() { return; }
    let (label, text_color, bg_color) = if cfg.paused {
        ("▶", GREEN_400, GREEN_500)
    } else {
        ("⏸", GOLD_500, GOLD_800)
    };
    for (mut text, mut color) in text_q.iter_mut() {
        *text  = Text::new(label);
        *color = TextColor(text_color);
    }
    for mut bg in bg_q.iter_mut() { bg.0 = bg_color; }
}
