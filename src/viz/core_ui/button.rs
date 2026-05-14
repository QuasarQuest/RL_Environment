// src/viz/core_ui/button.rs

use bevy::prelude::*;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use crate::style::{ThemeColor, ThemeMode, SIZE_MD, SIZE_XL};

pub fn spawn_labeled_button<M: Component>(
    parent: &mut ChildSpawnerCommands,
    mode: ThemeMode,
    text: &str,
    bg_color: ThemeColor,
    text_color: ThemeColor,
    btn_marker: M,
) {
    parent.spawn((
        Button,
        btn_marker,
        Node {
            padding: UiRect::axes(Val::Px(18.0), Val::Px(8.0)),
            border: UiRect::all(Val::Px(1.0)),
            align_items: AlignItems::Center,
            border_radius: BorderRadius::all(Val::Px(6.0)),
            ..default()
        },
        BackgroundColor(bg_color.resolve(mode)),
        BorderColor::all(ThemeColor::Border.resolve(mode)),
    )).with_children(|btn| {
        btn.spawn((
            Text::new(text),
            TextFont { font_size: SIZE_MD, ..default() },
            TextColor(text_color.resolve(mode)),
        ));
    });
}

pub fn spawn_icon_button<M: Component>(
    parent: &mut ChildSpawnerCommands,
    mode: ThemeMode,
    icon: &str,
    marker: M,
) {
    parent.spawn((
        Button,
        marker,
        Node {
            padding: UiRect::axes(Val::Px(12.0), Val::Px(6.0)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            border_radius: BorderRadius::all(Val::Px(4.0)),
            ..default()
        },
        BackgroundColor(ThemeColor::ButtonIdle.resolve(mode)),
        BorderColor::all(ThemeColor::Border.resolve(mode)),
    )).with_children(|btn| {
        btn.spawn((
            Text::new(icon),
            TextFont { font_size: SIZE_XL, ..default() },
            TextColor(ThemeColor::TextPrimary.resolve(mode)),
        ));
    });
}