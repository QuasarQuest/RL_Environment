use bevy::prelude::*;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use crate::style::{ThemeColor, UiRoot, SIZE_SM, SIZE_MD, TOOLBAR_H};
use crate::style::color::{
    team_color, ITEM_LEGEND, TILE_OBSTACLE, SPAWN_POCKET_BORDER, SPAWN_POCKET_FILL,
};

#[derive(Component)]
pub struct LegendOverlay;

/// One legend row: colour swatch (optional border) + label.
fn legend_row(
    panel:  &mut ChildSpawnerCommands,
    fill:   Color,
    border: Option<Color>,
    label:  &str,
    text:   Color,
) {
    panel.spawn(Node {
        flex_direction: FlexDirection::Row,
        align_items:    AlignItems::Center,
        column_gap:     Val::Px(10.0),
        ..default()
    }).with_children(|row| {
        let mut swatch = row.spawn((
            Node {
                width:         Val::Px(14.0),
                height:        Val::Px(14.0),
                border:        UiRect::all(Val::Px(if border.is_some() { 1.0 } else { 0.0 })),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                ..default()
            },
            BackgroundColor(fill),
        ));
        if let Some(b) = border {
            swatch.insert(BorderColor::all(b));
        }
        row.spawn((Text::new(label), TextFont { font_size: SIZE_SM, ..default() }, TextColor(text)));
    });
}

fn section_header(panel: &mut ChildSpawnerCommands, title: &str, body: Color, border: Color) {
    panel.spawn((Text::new(title), TextFont { font_size: SIZE_MD, ..default() }, TextColor(body)));
    panel.spawn((
        Node { height: Val::Px(1.0), margin: UiRect::vertical(Val::Px(4.0)), ..default() },
        BackgroundColor(border),
    ));
}

pub fn spawn_legend_overlay(mut commands: Commands) {
    let bg     = ThemeColor::Background.resolve();
    let border = ThemeColor::Border.resolve();
    let dim    = ThemeColor::TextDim.resolve();
    let body   = ThemeColor::TextPrimary.resolve();

    commands.spawn((
        UiRoot,
        LegendOverlay,
        Node {
            display:        Display::None,
            position_type:  PositionType::Absolute,
            top:            Val::Px(TOOLBAR_H + 8.0),
            left:           Val::Px(16.0),
            flex_direction: FlexDirection::Column,
            min_width:      Val::Px(260.0),
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
        // ── Map ────────────────────────────────────────────────────────────
        section_header(panel, "Map", body, border);
        legend_row(panel, team_color(0),        None,                       "Agent",    dim);
        legend_row(panel, team_color(0),        None,                       "Base",     dim);
        legend_row(panel, TILE_OBSTACLE,        None,                       "Obstacle", dim);
        legend_row(panel, SPAWN_POCKET_FILL,    Some(SPAWN_POCKET_BORDER),  "Spawn pocket (no obstacles)", dim);

        // ── Items ──────────────────────────────────────────────────────────
        panel.spawn(Node { height: Val::Px(10.0), ..default() });
        section_header(panel, "Items", body, border);
        for (label, swatch) in ITEM_LEGEND {
            legend_row(panel, *swatch, None, label, dim);
        }
    });
}

pub fn toggle_legend_overlay(
    keys:      Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut Node, With<LegendOverlay>>,
) {
    if keys.just_pressed(KeyCode::KeyI) {
        for mut node in query.iter_mut() {
            node.display = if node.display == Display::None { Display::Flex } else { Display::None };
        }
    }
}
