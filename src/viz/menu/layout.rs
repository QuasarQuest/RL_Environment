// src/viz/menu/layout.rs

use bevy::prelude::*;
use crate::style::{ThemeMode, ThemeColor, UiRoot, SIZE_SM, SIZE_LG, TOOLBAR_H, DRAWER_W};
use crate::viz::core_ui::button::spawn_icon_button;
use crate::viz::core_ui::text::spawn_label;
use super::components::*;

pub fn build_drawer(commands: &mut Commands, mode: ThemeMode) {
    let bg     = ThemeColor::Background.resolve(mode);
    let border = ThemeColor::Border.resolve(mode);
    let dim    = ThemeColor::TextDim.resolve(mode);
    let text   = ThemeColor::TextPrimary.resolve(mode);

    commands.spawn((
        UiRoot,
        DrawerOverlay,
        Interaction::default(),
        Node {
            position_type: PositionType::Absolute,
            top:           Val::Px(TOOLBAR_H),
            left:          Val::Px(0.0),
            width:         Val::Percent(100.0),
            height:        Val::Percent(100.0),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.4)),
        ZIndex(98),
    ));

    commands.spawn((
        UiRoot,
        DrawerPanel,
        Node {
            position_type:  PositionType::Absolute,
            top:            Val::Px(TOOLBAR_H),
            left:           Val::Px(0.0),
            width:          Val::Px(DRAWER_W),
            height:         Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            padding:        UiRect::all(Val::Px(12.0)),
            row_gap:        Val::Px(8.0),
            border:         UiRect::right(Val::Px(1.0)),
            ..default()
        },
        BackgroundColor(bg.with_alpha(0.85)),
        BorderColor::all(border),
        ZIndex(99),
    )).with_children(|drawer| {
        drawer.spawn(Node {
            flex_direction:  FlexDirection::Row,
            align_items:     AlignItems::Center,
            justify_content: JustifyContent::SpaceBetween,
            padding:         UiRect::bottom(Val::Px(12.0)),
            border:          UiRect::bottom(Val::Px(1.0)),
            margin:          UiRect::bottom(Val::Px(8.0)),
            ..default()
        }).with_children(|header| {
            header.spawn((
                Text::new("Menu"),
                TextFont  { font_size: SIZE_LG, ..default() },
                TextColor(text),
            ));
            spawn_icon_button(header, mode, "X", HamburgerButton);
        });

        drawer_section(drawer, "Simulation", dim);
        spawn_label(drawer, "Environment controls coming soon.", dim, SIZE_SM);

        drawer_section(drawer, "Agents", dim);
        spawn_label(drawer, "Spawn controls coming soon.", dim, SIZE_SM);
    });
}

fn drawer_section(parent: &mut ChildSpawnerCommands, label: &str, color: Color) {
    parent.spawn(Node {
        flex_direction: FlexDirection::Column,
        padding:        UiRect::top(Val::Px(6.0)),
        ..default()
    }).with_children(|s| {
        s.spawn((
            Text::new(label),
            TextFont  { font_size: SIZE_SM, ..default() },
            TextColor(color),
        ));
    });
}