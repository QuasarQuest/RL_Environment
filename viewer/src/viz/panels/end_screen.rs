use bevy::prelude::*;
use crate::sim_bridge::SimBridge;
use crate::style::{ThemeColor, UiRoot, SIZE_LG, SIZE_XL};
use crate::viz::events::RestartPending;

const MODAL_W: f32 = 400.0;
const MODAL_H: f32 = 320.0;

#[derive(Component)] pub struct EndScreen;
#[derive(Component)] pub struct WinnerLabel;
#[derive(Component)] pub struct QuitButton;
#[derive(Component)] pub struct RestartButton;

pub fn spawn_end_screen(mut commands: Commands) {
    let bg     = ThemeColor::Background.resolve();
    let border = ThemeColor::Border.resolve();
    let dim    = ThemeColor::TextDim.resolve();

    commands.spawn((
        UiRoot,
        EndScreen,
        Node {
            display:        Display::None,
            position_type:  PositionType::Absolute,
            left:           Val::Percent(50.0),
            top:            Val::Percent(50.0),
            margin:         UiRect { left: Val::Px(-MODAL_W / 2.0), top: Val::Px(-MODAL_H / 2.0), ..default() },
            flex_direction: FlexDirection::Column,
            align_items:    AlignItems::Center,
            min_width:      Val::Px(MODAL_W),
            padding:        UiRect::all(Val::Px(36.0)),
            row_gap:        Val::Px(16.0),
            border:         UiRect::all(Val::Px(1.0)),
            border_radius:  BorderRadius::all(Val::Px(16.0)),
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(border),
        ZIndex(500),
    )).with_children(|panel| {
        panel.spawn((Text::new("MATCH OVER"), TextFont { font_size: 28.0, ..default() }, TextColor(ThemeColor::TextPrimary.resolve())));
        panel.spawn((Text::new(""), TextFont { font_size: SIZE_XL, ..default() }, TextColor(dim), WinnerLabel));

        // Buttons row
        panel.spawn(Node {
            flex_direction: FlexDirection::Row,
            column_gap:     Val::Px(12.0),
            margin:         UiRect::top(Val::Px(8.0)),
            ..default()
        }).with_children(|btns| {
            action_button(btns, "Restart", ThemeColor::SuccessText.resolve(), RestartButton);
            action_button(btns, "Quit",    ThemeColor::TextDim.resolve(),     QuitButton);
        });
    });
}

fn action_button<M: Component>(
    parent: &mut ChildSpawnerCommands,
    label:  &str,
    color:  Color,
    marker: M,
) {
    parent.spawn((
        Button,
        marker,
        Node {
            padding:       UiRect::axes(Val::Px(24.0), Val::Px(10.0)),
            border_radius: BorderRadius::all(Val::Px(6.0)),
            ..default()
        },
        BackgroundColor(ThemeColor::ButtonIdle.resolve()),
        BorderColor::all(ThemeColor::Border.resolve()),
    )).with_children(|btn| {
        btn.spawn((Text::new(label), TextFont { font_size: SIZE_LG, ..default() }, TextColor(color)));
    });
}

pub fn show_end_screen(
    bridge:    Res<SimBridge>,
    mut query: Query<&mut Node, With<EndScreen>>,
    mut label: Query<&mut Text, With<WinnerLabel>>,
) {
    if !bridge.is_changed() { return; }
    let game_over = bridge.game_over;
    let want = if game_over { Display::Flex } else { Display::None };
    for mut node in query.iter_mut() {
        // Write only on transition (unconditional writes relayout every frame).
        if node.display != want { node.display = want; }
    }
    if game_over {
        // Find leading team.
        let n_teams = bridge.n_teams();
        let winner = (0..n_teams)
            .max_by_key(|&t| bridge.team_score(t));
        let msg = winner.map(|t| format!("{} wins!", crate::team::Team(t).name())).unwrap_or_default();
        for mut text in label.iter_mut() { *text = Text::new(&msg); }
    }
}

pub fn handle_restart_button(
    btn_q:          Query<&Interaction, (Changed<Interaction>, With<RestartButton>)>,
    mut restart:    ResMut<RestartPending>,
) {
    for interaction in btn_q.iter() {
        if *interaction == Interaction::Pressed { restart.0 = true; }
    }
}

pub fn handle_quit_button(
    btn_q:    Query<&Interaction, (Changed<Interaction>, With<QuitButton>)>,
    mut exit: MessageWriter<AppExit>,
) {
    for interaction in btn_q.iter() {
        // AppExit runs Bevy's shutdown (Drop impls, window teardown) —
        // std::process::exit would skip all of it.
        if *interaction == Interaction::Pressed { exit.write(AppExit::Success); }
    }
}
