// src/viz/hud/scoreboard.rs

use bevy::prelude::*;
use crate::agent::components::{AgentLabel, GoldCarried, Score};
use crate::style::{ThemeMode, ThemeColor, UiRoot, SIZE_SM, SIZE_MD};

#[derive(Component)] pub struct ScoreboardPanel;
#[derive(Component)] pub struct ScoreboardContent;

pub fn spawn_scoreboard(mut commands: Commands, theme: Res<ThemeMode>) {
    build_scoreboard(&mut commands, *theme);
}

pub fn build_scoreboard(commands: &mut Commands, mode: ThemeMode) {
    commands.spawn((
        UiRoot,
        ScoreboardPanel,
        Node {
            position_type:  PositionType::Absolute,
            bottom:         Val::Px(12.0),
            right:          Val::Px(12.0),
            flex_direction: FlexDirection::Column,
            min_width:      Val::Px(240.0),
            border:         UiRect::all(Val::Px(1.0)),
            border_radius:  BorderRadius::all(Val::Px(8.0)),
            padding:        UiRect::all(Val::Px(1.0)),
            ..default()
        },
        BackgroundColor(ThemeColor::Background.resolve(mode)),
        BorderColor::all(ThemeColor::Border.resolve(mode)),
    )).with_children(|panel| {
        panel.spawn(Node {
            flex_direction: FlexDirection::Row,
            padding:        UiRect::new(Val::Px(12.0), Val::Px(12.0), Val::Px(8.0), Val::Px(6.0)),
            column_gap:     Val::Px(8.0),
            ..default()
        }).with_children(|h| {
            cell(h, "AGENT", 120.0, ThemeColor::TextDim.resolve(mode),    SIZE_SM);
            cell(h, "Gold",   32.0, ThemeColor::AccentGold.resolve(mode), SIZE_SM);
            cell(h, "SCORE",  60.0, ThemeColor::TextDim.resolve(mode),    SIZE_SM);
        });

        panel.spawn((
            ScoreboardContent,
            Node { flex_direction: FlexDirection::Column, ..default() },
        ));
    });
}

pub fn update_scoreboard(
    agents:        Query<(&AgentLabel, &GoldCarried, &Score)>,
    changed_gold:  Query<(), Changed<GoldCarried>>,
    changed_score: Query<(), Changed<Score>>,
    content_q:     Query<Entity, With<ScoreboardContent>>,
    theme:         Res<ThemeMode>,
    mut commands:  Commands,
) {
    let data_changed = !changed_gold.is_empty() || !changed_score.is_empty();
    if !data_changed      { return; }
    if theme.is_changed() { return; }

    let Ok(content) = content_q.single() else { return };
    commands.entity(content).despawn_related::<Children>();

    let mut rows: Vec<_> = agents.iter().collect();
    rows.sort_by_key(|(label, _, _)| label.0.clone());

    commands.entity(content).with_children(|c| {
        for (i, (label, gold, score)) in rows.iter().enumerate() {
            let bg = if i % 2 == 0 {
                Color::NONE
            } else {
                ThemeColor::SurfaceHighlight.resolve(*theme)
            };
            c.spawn((
                Node {
                    flex_direction: FlexDirection::Row,
                    padding:        UiRect::axes(Val::Px(12.0), Val::Px(5.0)),
                    column_gap:     Val::Px(8.0),
                    border_radius:  BorderRadius::all(Val::Px(4.0)),
                    ..default()
                },
                BackgroundColor(bg),
            )).with_children(|row| {
                cell(row, &label.0,            120.0, ThemeColor::TextPrimary.resolve(*theme), SIZE_MD);
                cell(row, &gold.0.to_string(),  32.0, ThemeColor::AccentGold.resolve(*theme),  SIZE_MD);
                cell(row, &score.0.to_string(), 60.0, ThemeColor::SuccessText.resolve(*theme), SIZE_MD);
            });
        }
    });
}

fn cell(parent: &mut ChildSpawnerCommands, text: &str, width: f32, color: Color, size: f32) {
    parent.spawn((
        Text::new(text),
        TextFont  { font_size: size, ..default() },
        TextColor(color),
        Node { width: Val::Px(width), ..default() },
    ));
}