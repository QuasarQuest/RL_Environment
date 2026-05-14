// src/viz/core_ui/panel.rs

use bevy::prelude::*;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use crate::style::{ThemeColor, ThemeMode, UiRoot};

/// Spawns a standardized floating panel at the specified absolute position.
pub fn spawn_floating_panel(
    commands: &mut Commands,
    mode: ThemeMode, // <-- Accept the current theme mode
    top: Val, right: Val, bottom: Val, left: Val,
    build_children: impl FnOnce(&mut ChildSpawnerCommands)
) {
    commands.spawn((
        UiRoot, // <-- The React Marker!
        Node {
            position_type: PositionType::Absolute,
            top, right, bottom, left,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(16.0),
            padding: UiRect::axes(Val::Px(16.0), Val::Px(8.0)),
            border_radius: BorderRadius::all(Val::Px(8.0)),
            border: UiRect::all(Val::Px(1.0)),
            ..default()
        },
        BackgroundColor(ThemeColor::Background.resolve(mode)), // <-- Resolved dynamically!
        BorderColor::all(ThemeColor::Border.resolve(mode)),    // <-- Resolved dynamically!
    )).with_children(build_children);
}

/// A wrapper for grouping buttons together (like the speed controls)
pub fn spawn_button_group(
    parent: &mut ChildSpawnerCommands,
    mode: ThemeMode, // <-- Accept the current theme mode
    build_children: impl FnOnce(&mut ChildSpawnerCommands)
) {
    parent.spawn((
        Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(4.0),
            padding: UiRect::all(Val::Px(4.0)),
            border_radius: BorderRadius::all(Val::Px(8.0)),
            border: UiRect::all(Val::Px(1.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.2)), // Keeps its slight transparency
        BorderColor::all(ThemeColor::Border.resolve(mode)), // <-- Resolved dynamically!
    )).with_children(build_children);
}