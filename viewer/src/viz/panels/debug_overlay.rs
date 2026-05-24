use bevy::prelude::*;
use atb::rl::action::ACTION_SIZE;
use crate::sim_bridge::SimBridge;
use crate::style::{ThemeColor, UiRoot, SIZE_SM, TOOLBAR_H};

const ACTION_NAMES: [&str; ACTION_SIZE] = [
    "Move N", "Move S", "Move E", "Move W", "Move NE", "Move NW", "Move SE", "Move SW",
    "Atk N",  "Atk S",  "Atk E",  "Atk W",  "Atk NE",  "Atk NW",  "Atk SE",  "Atk SW",
    "Rng N",  "Rng S",  "Rng E",  "Rng W",  "Rng NE",  "Rng NW",  "Rng SE",  "Rng SW",
    "Drop",   "Wait",
];

#[derive(Component)] pub struct DebugOverlay;
#[derive(Component)] pub struct DebugLastAction;
#[derive(Component)] pub struct DebugReward;
#[derive(Component)] pub struct DebugDist;

pub fn spawn_debug_overlay(mut commands: Commands) {
    let bg     = ThemeColor::Background.resolve();
    let border = ThemeColor::Border.resolve();
    let dim    = ThemeColor::TextDim.resolve();
    let body   = ThemeColor::TextPrimary.resolve();

    commands.spawn((
        UiRoot,
        DebugOverlay,
        Node {
            display:        Display::None,
            position_type:  PositionType::Absolute,
            top:            Val::Px(TOOLBAR_H + 8.0),
            left:           Val::Px(12.0),
            flex_direction: FlexDirection::Column,
            min_width:      Val::Px(220.0),
            border:         UiRect::all(Val::Px(1.0)),
            border_radius:  BorderRadius::all(Val::Px(8.0)),
            padding:        UiRect::all(Val::Px(12.0)),
            row_gap:        Val::Px(4.0),
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(border),
        ZIndex(200),
    )).with_children(|p| {
        p.spawn((Text::new("Policy Debug"), TextFont { font_size: SIZE_SM, ..default() }, TextColor(dim)));
        p.spawn((Node { height: Val::Px(1.0), margin: UiRect::vertical(Val::Px(2.0)), ..default() }, BackgroundColor(border)));

        kv_row(p, "Action", body, dim, DebugLastAction);
        kv_row(p, "Reward", body, dim, DebugReward);

        p.spawn((Node { height: Val::Px(1.0), margin: UiRect::vertical(Val::Px(2.0)), ..default() }, BackgroundColor(border)));
        p.spawn((
            Text::new("—"),
            TextFont { font_size: SIZE_SM, ..default() },
            TextColor(dim),
            DebugDist,
        ));
    });
}

fn kv_row<M: Component>(
    parent: &mut ChildSpawnerCommands,
    label:  &str,
    body:   Color,
    dim:    Color,
    marker: M,
) {
    parent.spawn(Node {
        flex_direction:  FlexDirection::Row,
        justify_content: JustifyContent::SpaceBetween,
        column_gap:      Val::Px(12.0),
        ..default()
    }).with_children(|row| {
        row.spawn((Text::new(label), TextFont { font_size: SIZE_SM, ..default() }, TextColor(dim)));
        row.spawn((Text::new("—"), TextFont { font_size: SIZE_SM, ..default() }, TextColor(body), marker));
    });
}

pub fn toggle_debug_overlay(
    keys:  Res<ButtonInput<KeyCode>>,
    mut q: Query<&mut Node, With<DebugOverlay>>,
) {
    if keys.just_pressed(KeyCode::KeyD) {
        for mut node in q.iter_mut() {
            node.display = if node.display == Display::None { Display::Flex } else { Display::None };
        }
    }
}

pub fn update_debug_overlay(
    bridge:      Res<SimBridge>,
    mut action_q: Query<&mut Text, (With<DebugLastAction>, Without<DebugReward>, Without<DebugDist>)>,
    mut reward_q: Query<&mut Text, (With<DebugReward>, Without<DebugLastAction>, Without<DebugDist>)>,
    mut dist_q:   Query<&mut Text, (With<DebugDist>, Without<DebugLastAction>, Without<DebugReward>)>,
) {
    if !bridge.is_changed() { return; }

    // Last action name.
    let action_name = ACTION_NAMES.get(bridge.last_action as usize).copied().unwrap_or("?");
    for mut t in action_q.iter_mut() { *t = Text::new(action_name); }

    // Cumulative episode reward.
    for mut t in reward_q.iter_mut() { *t = Text::new(format!("{:.3}", bridge.episode_reward)); }

    // Action distribution: top 5 by count.
    let total: u32 = bridge.action_counts.iter().sum();
    if total == 0 { return; }

    let mut ranked: Vec<(usize, u32)> = bridge.action_counts
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, c)| c > 0)
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));

    let lines: String = ranked.iter()
        .take(5)
        .map(|&(idx, count)| {
            let pct = 100.0 * count as f32 / total as f32;
            format!("{:<8} {:5.1}%  ({})", ACTION_NAMES[idx], pct, count)
        })
        .collect::<Vec<_>>()
        .join("\n");

    for mut t in dist_q.iter_mut() { *t = Text::new(lines.clone()); }
}
