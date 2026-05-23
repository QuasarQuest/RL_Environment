use bevy::prelude::*;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use crate::style::ThemeColor;

pub fn spawn_button_group(
    parent:         &mut ChildSpawnerCommands,
    build_children: impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent.spawn((
        Node {
            flex_direction: FlexDirection::Row,
            align_items:    AlignItems::Center,
            column_gap:     Val::Px(4.0),
            padding:        UiRect::all(Val::Px(4.0)),
            border_radius:  BorderRadius::all(Val::Px(8.0)),
            border:         UiRect::all(Val::Px(1.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.2)),
        BorderColor::all(ThemeColor::Border.resolve()),
    )).with_children(build_children);
}
