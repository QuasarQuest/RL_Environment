// src/viz/hud/systems.rs

use bevy::prelude::*;
use crate::sim::config::SimConfig;
use crate::team::{Team, TeamScore};
use super::components::{TickLabelMarker, TimeLabelMarker, TeamScoreMarker};

pub fn update_tick_label(
    cfg: Res<SimConfig>,
    mut query: Query<&mut Text, With<TickLabelMarker>>,
) {
    if !cfg.is_changed() { return; }
    for mut text in query.iter_mut() {
        *text = Text::new(format!("{}", cfg.tick));
    }
}

pub fn update_time_label(
    cfg: Res<SimConfig>,
    mut query: Query<&mut Text, With<TimeLabelMarker>>,
) {
    if !cfg.is_changed() { return; }
    for mut text in query.iter_mut() {
        *text = Text::new(cfg.remaining_display());
    }
}

pub fn update_team_scores(
    team_score: Res<TeamScore>,
    mut query: Query<(&mut Text, &TeamScoreMarker)>,
) {
    if !team_score.is_changed() { return; }
    for (mut text, marker) in query.iter_mut() {
        let score = team_score.get(Team(marker.0));
        let name  = Team(marker.0).name();
        *text = Text::new(format!("{}: {}", name, score));
    }
}