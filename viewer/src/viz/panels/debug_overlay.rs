use bevy::prelude::*;
use atb::rl::action::{int_to_rl_action, RlAction};
use crate::sim_bridge::SimBridge;
use crate::style::{ThemeColor, UiRoot, SIZE_SM, TOOLBAR_H};
use crate::style::color::{CYAN_400, LIME_400, VIOLET_400, AMBER_400, ORANGE_400, GRAY_400};

fn action_color(action: RlAction) -> Color {
    match action {
        RlAction::NavigateToCluster(0) => CYAN_400,
        RlAction::NavigateToCluster(1) => LIME_400,
        RlAction::NavigateToCluster(2) => VIOLET_400,
        RlAction::NavigateToCluster(_) => AMBER_400,
        RlAction::NavigateToBase        => ORANGE_400,
        RlAction::NavigateToSpeed       => LIME_400,
        RlAction::NavigateToMultiplier  => VIOLET_400,
        RlAction::NavigateToNearestGold => CYAN_400,
        RlAction::Wait                  => GRAY_400,
    }
}

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
    bridge:       Res<SimBridge>,
    mut action_q: Query<
        (&mut Text, &mut TextColor),
        (With<DebugLastAction>, Without<DebugReward>, Without<DebugDist>),
    >,
    mut reward_q: Query<&mut Text, (With<DebugReward>, Without<DebugLastAction>, Without<DebugDist>)>,
    mut dist_q:   Query<&mut Text, (With<DebugDist>, Without<DebugLastAction>, Without<DebugReward>)>,
) {
    if !bridge.is_changed() { return; }

    let action     = int_to_rl_action(bridge.last_action);
    let action_col = action_color(action);

    for (mut t, mut tc) in action_q.iter_mut() {
        *t  = Text::new(action.to_string());
        *tc = TextColor(action_col);
    }

    for mut t in reward_q.iter_mut() { *t = Text::new(format!("{:.3}", bridge.episode_reward)); }

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
            let name = int_to_rl_action(idx as u32).to_string();
            let pct  = 100.0 * count as f32 / total as f32;
            format!("{name:<14} {pct:5.1}%  ({count})")
        })
        .collect::<Vec<_>>()
        .join("\n");

    for mut t in dist_q.iter_mut() { *t = Text::new(lines.clone()); }
}
