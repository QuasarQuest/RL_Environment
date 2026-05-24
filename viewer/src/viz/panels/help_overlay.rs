use bevy::prelude::*;
use crate::style::{ThemeColor, UiRoot, SIZE_SM, SIZE_MD, TOOLBAR_H};

#[derive(Component)]
pub struct HelpOverlay;

const SHORTCUTS: &[(&str, &str)] = &[
    ("TAB",          "Hold — scoreboard"),
    ("H",            "Toggle this help"),
    ("D",            "Toggle policy debug"),
    ("P",            "Cycle policy  ONNX → RANDOM → BT"),
    ("Space",        "Pause / Resume"),
    ("↑",            "Speed up"),
    ("↓",            "Speed down"),
    ("Scroll wheel", "Zoom in / out"),
    ("Middle mouse", "Pan camera"),
];

pub fn spawn_help_overlay(mut commands: Commands) {
    let bg     = ThemeColor::Background.resolve();
    let border = ThemeColor::Border.resolve();
    let dim    = ThemeColor::TextDim.resolve();
    let body   = ThemeColor::TextPrimary.resolve();

    commands.spawn((
        UiRoot,
        HelpOverlay,
        Node {
            display:        Display::None,
            position_type:  PositionType::Absolute,
            top:            Val::Px(TOOLBAR_H + 8.0),
            right:          Val::Px(16.0),
            flex_direction: FlexDirection::Column,
            min_width:      Val::Px(300.0),
            border:         UiRect::all(Val::Px(1.0)),
            border_radius:  BorderRadius::all(Val::Px(10.0)),
            padding:        UiRect::all(Val::Px(16.0)),
            row_gap:        Val::Px(6.0),
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(border),
        ZIndex(300),
    )).with_children(|panel| {
        panel.spawn((Text::new("Shortcuts"), TextFont { font_size: SIZE_MD, ..default() }, TextColor(body)));
        panel.spawn((Node { height: Val::Px(1.0), margin: UiRect::vertical(Val::Px(4.0)), ..default() }, BackgroundColor(border)));

        for (key, desc) in SHORTCUTS {
            panel.spawn(Node {
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::SpaceBetween,
                column_gap:     Val::Px(16.0),
                ..default()
            }).with_children(|row| {
                row.spawn((Text::new(*key),  TextFont { font_size: SIZE_SM, ..default() }, TextColor(body)));
                row.spawn((Text::new(*desc), TextFont { font_size: SIZE_SM, ..default() }, TextColor(dim)));
            });
        }
    });
}

pub fn toggle_help_overlay(
    keys:      Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut Node, With<HelpOverlay>>,
) {
    if keys.just_pressed(KeyCode::KeyH) {
        for mut node in query.iter_mut() {
            node.display = if node.display == Display::None { Display::Flex } else { Display::None };
        }
    }
}
