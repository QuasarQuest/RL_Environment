use bevy::prelude::*;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use crate::style::{ThemeColor, SIZE_XL};

pub fn spawn_icon_button<M: Component>(
    parent: &mut ChildSpawnerCommands,
    icon:   &str,
    marker: M,
) {
    parent.spawn((
        Button,
        marker,
        Node {
            padding:         UiRect::axes(Val::Px(12.0), Val::Px(6.0)),
            justify_content: JustifyContent::Center,
            align_items:     AlignItems::Center,
            border_radius:   BorderRadius::all(Val::Px(4.0)),
            ..default()
        },
        BackgroundColor(ThemeColor::ButtonIdle.resolve()),
        BorderColor::all(ThemeColor::Border.resolve()),
    )).with_children(|btn| {
        btn.spawn((
            Text::new(icon),
            TextFont  { font_size: SIZE_XL, ..default() },
            TextColor(ThemeColor::TextPrimary.resolve()),
        ));
    });
}
