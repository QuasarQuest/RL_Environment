// src/viz/help_overlay.rs
//
// Press H → toggle shortcut reference overlay.

use bevy::prelude::*;
use crate::style::{ThemeMode, ThemeColor, UiRoot, SIZE_SM, SIZE_MD, TOOLBAR_H};

#[derive(Component)]
pub struct HelpOverlay;

const SHORTCUTS: &[(&str, &str)] = &[
    ("TAB",          "Hold — scoreboard"),
    ("H",            "Toggle this help"),
    ("Space",        "Pause / Resume"),
    ("F",            "Speed up"),
    ("S",            "Speed down"),
    ("Scroll wheel", "Zoom in / out"),
    ("Middle mouse", "Pan camera"),
    ("Left click",   "Toggle agent debug viz (hover agent first)"),
];

pub fn spawn_help_overlay(mut commands: Commands, theme: Res<ThemeMode>) {
    build_help_overlay(&mut commands, *theme);
}

pub fn build_help_overlay(commands: &mut Commands, mode: ThemeMode) {
    let bg     = ThemeColor::Background.resolve(mode);
    let border = ThemeColor::Border.resolve(mode);
    let dim    = ThemeColor::TextDim.resolve(mode);
    let body   = ThemeColor::TextPrimary.resolve(mode);

    commands.spawn((
        UiRoot,
        HelpOverlay,
        Node {
            display:        Display::None,
            position_type:  PositionType::Absolute,
            top:            Val::Px(TOOLBAR_H + 8.0),
            right:          Val::Px(16.0),
            flex_direction: FlexDirection::Column,
            min_width:      Val::Px(340.0),
            border:         UiRect::all(Val::Px(1.0)),
            border_radius:  BorderRadius::all(Val::Px(10.0)),
            padding:        UiRect::all(Val::Px(16.0)),
            row_gap:        Val::Px(8.0),
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(border),
        ZIndex(300),
    )).with_children(|panel| {
        // Title
        panel.spawn((
            Text::new("Shortcuts"),
            TextFont  { font_size: SIZE_MD, ..default() },
            TextColor(body),
        ));

        // Divider gap
        panel.spawn(Node {
            height: Val::Px(4.0),
            ..default()
        });

        // Rows
        for (key, desc) in SHORTCUTS {
            panel.spawn(Node {
                flex_direction: FlexDirection::Row,
                column_gap:     Val::Px(12.0),
                align_items:    AlignItems::Center,
                ..default()
            }).with_children(|row| {
                // Key badge
                row.spawn((
                    Node {
                        min_width:       Val::Px(110.0),
                        padding:         UiRect::axes(Val::Px(8.0), Val::Px(3.0)),
                        border:          UiRect::all(Val::Px(1.0)),
                        border_radius:   BorderRadius::all(Val::Px(4.0)),
                        justify_content: JustifyContent::Center,
                        ..default()
                    },
                    BackgroundColor(ThemeColor::ButtonIdle.resolve(mode)),
                    BorderColor::all(border),
                )).with_children(|badge| {
                    badge.spawn((
                        Text::new(*key),
                        TextFont  { font_size: SIZE_SM, ..default() },
                        TextColor(body),
                    ));
                });

                // Description
                row.spawn((
                    Text::new(*desc),
                    TextFont  { font_size: SIZE_SM, ..default() },
                    TextColor(dim),
                ));
            });
        }
    });
}

pub fn toggle_help_overlay(
    keys:  Res<ButtonInput<KeyCode>>,
    mut q: Query<&mut Node, With<HelpOverlay>>,
) {
    if !keys.just_pressed(KeyCode::KeyH) { return; }
    for mut node in q.iter_mut() {
        node.display = if node.display == Display::None {
            Display::Flex
        } else {
            Display::None
        };
    }
}