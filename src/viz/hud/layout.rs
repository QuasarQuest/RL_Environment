// src/viz/hud/layout.rs

use bevy::prelude::*;
use crate::style::{ThemeMode, ThemeColor, UiRoot, SIZE_SM, SIZE_MD, SIZE_LG, TOOLBAR_H};
use crate::viz::core_ui::button::{spawn_icon_button, spawn_labeled_button};
use crate::viz::core_ui::panel::spawn_button_group;
use crate::viz::core_ui::text::{spawn_label, spawn_marked_label};
use super::components::TickLabelMarker;
use crate::viz::menu::components::{
    HamburgerButton, ThemeToggleButton,
    SpeedDecreaseButton, SpeedIncreaseButton, SpeedResetButton,
    CurrentSpeedLabel, PauseButtonMarker,
};

pub fn spawn_hud(mut commands: Commands, theme: Res<ThemeMode>) {
    build_hud(&mut commands, *theme);
}

pub fn build_hud(commands: &mut Commands, mode: ThemeMode) {
    let bg     = ThemeColor::Background.resolve(mode);
    let border = ThemeColor::Border.resolve(mode);

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

        // ── Left: Hamburger & Theme Toggle ────────────────────────────────────
        top_bar.spawn(Node {
            flex_direction: FlexDirection::Row,
            align_items:    AlignItems::Center,
            column_gap:     Val::Px(8.0),
            ..default()
        }).with_children(|left| {
            spawn_icon_button(left, mode, "=", HamburgerButton);
            let theme_text = match mode {
                ThemeMode::Dark  => "Light",
                ThemeMode::Light => "Dark",
            };
            spawn_labeled_button(left, mode, theme_text, ThemeColor::ButtonIdle, ThemeColor::TextPrimary, ThemeToggleButton);
        });

        // ── Center: Speed Controls ────────────────────────────────────────────
        spawn_button_group(top_bar, mode, |center| {
            spawn_icon_button(center, mode, "-", SpeedDecreaseButton);
            spawn_speed_label(center, mode);
            spawn_icon_button(center, mode, "+", SpeedIncreaseButton);
        });

        // ── Right: Pause & Tick Counter ───────────────────────────────────────
        top_bar.spawn(Node {
            flex_direction: FlexDirection::Row,
            align_items:    AlignItems::Center,
            column_gap:     Val::Px(16.0),
            ..default()
        }).with_children(|right| {
            spawn_labeled_button(right, mode, "Running", ThemeColor::Success, ThemeColor::SuccessText, PauseButtonMarker);

            right.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items:    AlignItems::Center,
                column_gap:     Val::Px(6.0),
                ..default()
            }).with_children(|tick| {
                spawn_label(tick, "TICK", ThemeColor::TextDim.resolve(mode), SIZE_SM);
                spawn_marked_label(tick, "0", ThemeColor::TextPrimary.resolve(mode), SIZE_LG, TickLabelMarker);
            });
        });
    });
}

fn spawn_speed_label(parent: &mut ChildSpawnerCommands, mode: ThemeMode) {
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
        BackgroundColor(ThemeColor::ButtonIdle.resolve(mode)),
        BorderColor::all(ThemeColor::Border.resolve(mode)),
    )).with_children(|btn| {
        btn.spawn((
            Text::new("10x"),
            TextFont  { font_size: SIZE_MD, ..default() },
            TextColor(ThemeColor::TextPrimary.resolve(mode)),
            CurrentSpeedLabel,
        ));
    });
}