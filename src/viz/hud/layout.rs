// src/viz/hud/layout.rs

use bevy::prelude::*;
use crate::style::{ThemeColor, UiRoot, SIZE_SM, SIZE_MD, SIZE_LG, SIZE_XL, TOOLBAR_H};
use crate::style::color::team_color;
use crate::viz::core_ui::button::spawn_icon_button;
use crate::viz::core_ui::panel::spawn_button_group;
use crate::viz::core_ui::text::{spawn_label, spawn_marked_label};
use super::components::{
    TickLabelMarker, TimeLabelMarker, TeamScoreMarker,
    SpeedDecreaseButton, SpeedIncreaseButton, SpeedResetButton,
    CurrentSpeedLabel, PauseButtonMarker, PauseButtonText,
};

pub fn spawn_hud(mut commands: Commands) {
    let bg         = ThemeColor::Background.resolve();
    let border     = ThemeColor::Border.resolve();
    let dim        = ThemeColor::TextDim.resolve();
    let red_color  = team_color(0);
    let blue_color = team_color(1);

    commands.spawn((
        UiRoot,
        Node {
            width:           Val::Percent(100.0),
            height:          Val::Px(TOOLBAR_H),
            position_type:   PositionType::Absolute,
            top:             Val::Px(0.0),
            left:            Val::Px(0.0),
            flex_direction:  FlexDirection::Row,
            align_items:     AlignItems::Center,
            justify_content: JustifyContent::SpaceBetween,
            padding:         UiRect::axes(Val::Px(16.0), Val::Px(0.0)),
            border:          UiRect::bottom(Val::Px(1.0)),
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(border),
        ZIndex(100),
    )).with_children(|top_bar| {

        // ── Left: Football scoreboard  RED 0 | 0:00 | 0 BLUE ─────────────────
        top_bar.spawn(Node {
            flex_direction: FlexDirection::Row,
            align_items:    AlignItems::Stretch,
            ..default()
        }).with_children(|left| {
            // Team 0
            left.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items:    AlignItems::Center,
                column_gap:     Val::Px(8.0),
                padding:        UiRect::horizontal(Val::Px(16.0)),
                ..default()
            }).with_children(|t0| {
                t0.spawn((
                    Text::new("RED"),
                    TextFont  { font_size: SIZE_SM, ..default() },
                    TextColor(red_color.with_alpha(0.7)),
                ));
                t0.spawn((
                    Text::new("0"),
                    TextFont  { font_size: SIZE_XL, ..default() },
                    TextColor(red_color),
                    TeamScoreMarker(0),
                ));
            });

            vdivider(left, border);

            // Time
            left.spawn(Node {
                flex_direction:  FlexDirection::Column,
                align_items:     AlignItems::Center,
                justify_content: JustifyContent::Center,
                padding:         UiRect::horizontal(Val::Px(20.0)),
                row_gap:         Val::Px(1.0),
                ..default()
            }).with_children(|time_col| {
                time_col.spawn((
                    Text::new("0 / 10000"),
                    TextFont  { font_size: SIZE_XL, ..default() },
                    TextColor(ThemeColor::TextPrimary.resolve()),
                    TimeLabelMarker,
                ));
                time_col.spawn((
                    Text::new("TICK"),
                    TextFont  { font_size: SIZE_SM, ..default() },
                    TextColor(dim),
                ));
            });

            vdivider(left, border);

            // Team 1
            left.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items:    AlignItems::Center,
                column_gap:     Val::Px(8.0),
                padding:        UiRect::horizontal(Val::Px(16.0)),
                ..default()
            }).with_children(|t1| {
                t1.spawn((
                    Text::new("0"),
                    TextFont  { font_size: SIZE_XL, ..default() },
                    TextColor(blue_color),
                    TeamScoreMarker(1),
                ));
                t1.spawn((
                    Text::new("BLUE"),
                    TextFont  { font_size: SIZE_SM, ..default() },
                    TextColor(blue_color.with_alpha(0.7)),
                ));
            });
        });

        // ── Right: Speed + Pause + Tick ───────────────────────────────────────
        top_bar.spawn(Node {
            flex_direction: FlexDirection::Row,
            align_items:    AlignItems::Center,
            column_gap:     Val::Px(16.0),
            ..default()
        }).with_children(|right| {
            spawn_button_group(right, |grp| {
                spawn_icon_button(grp, "-", SpeedDecreaseButton);
                spawn_speed_label(grp);
                spawn_icon_button(grp, "+", SpeedIncreaseButton);
            });

            right.spawn((
                Button,
                PauseButtonMarker,
                Node {
                    padding:       UiRect::axes(Val::Px(18.0), Val::Px(8.0)),
                    border:        UiRect::all(Val::Px(1.0)),
                    align_items:   AlignItems::Center,
                    border_radius: BorderRadius::all(Val::Px(6.0)),
                    ..default()
                },
                BackgroundColor(ThemeColor::Success.resolve()),
                BorderColor::all(ThemeColor::Border.resolve()),
            )).with_children(|btn| {
                btn.spawn((
                    Text::new("|>"),
                    TextFont  { font_size: SIZE_MD, ..default() },
                    TextColor(ThemeColor::SuccessText.resolve()),
                    PauseButtonText,
                ));
            });

            right.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items:    AlignItems::Center,
                column_gap:     Val::Px(6.0),
                ..default()
            }).with_children(|tick| {
                spawn_label(tick, "TICK", dim, SIZE_SM);
                spawn_marked_label(tick, "0",
                                   ThemeColor::TextPrimary.resolve(), SIZE_LG, TickLabelMarker);
            });
        });
    });
}

fn vdivider(parent: &mut ChildSpawnerCommands, color: Color) {
    parent.spawn((
        Node {
            width:      Val::Px(0.0),
            height:     Val::Percent(60.0),
            border:     UiRect::left(Val::Px(1.0)),
            align_self: AlignSelf::Center,
            ..default()
        },
        BorderColor::all(color),
    ));
}

fn spawn_speed_label(parent: &mut ChildSpawnerCommands) {
    parent.spawn((
        Button,
        SpeedResetButton,
        Node {
            padding:         UiRect::axes(Val::Px(12.0), Val::Px(6.0)),
            justify_content: JustifyContent::Center,
            align_items:     AlignItems::Center,
            border_radius:   BorderRadius::all(Val::Px(4.0)),
            min_width:       Val::Px(48.0),
            ..default()
        },
        BackgroundColor(ThemeColor::ButtonIdle.resolve()),
        BorderColor::all(ThemeColor::Border.resolve()),
    )).with_children(|btn| {
        btn.spawn((
            Text::new("10x"),
            TextFont  { font_size: SIZE_MD, ..default() },
            TextColor(ThemeColor::TextPrimary.resolve()),
            CurrentSpeedLabel,
        ));
    });
}