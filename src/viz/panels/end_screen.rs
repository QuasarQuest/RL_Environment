// src/viz/panels/end_screen.rs
use bevy::prelude::*;
use crate::sim::config::SimConfig;
use crate::team::{Team, TeamScore};
use crate::agent::components::{DeathCount, KillCount, Score};
use crate::style::{ThemeColor, UiRoot, SIZE_SM, SIZE_MD, SIZE_LG, SIZE_XL};
use crate::style::color::team_color;
use crate::viz::events::RestartEvent;

#[derive(Component)] pub struct EndScreen;
#[derive(Component)] pub struct WinnerLabel;
#[derive(Component)] pub struct TeamStatsContainer;
#[derive(Component)] pub struct CardsPopulated;
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
            display:       Display::None,
            position_type: PositionType::Absolute,
            left:          Val::Percent(50.0),
            top:           Val::Percent(50.0),
            margin:        UiRect {
                left: Val::Px(-260.0),
                top:  Val::Px(-200.0),
                ..default()
            },
            flex_direction: FlexDirection::Column,
            align_items:    AlignItems::Center,
            min_width:      Val::Px(520.0),
            padding:        UiRect::all(Val::Px(36.0)),
            row_gap:        Val::Px(20.0),
            border:         UiRect::all(Val::Px(1.0)),
            border_radius:  BorderRadius::all(Val::Px(16.0)),
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(border),
        ZIndex(500),
    )).with_children(|panel| {
        panel.spawn((
            Text::new("MATCH OVER"),
            TextFont  { font_size: 28.0, ..default() },
            TextColor(ThemeColor::TextPrimary.resolve()),
        ));
        panel.spawn((
            Text::new(""),
            TextFont  { font_size: SIZE_XL, ..default() },
            TextColor(dim),
            WinnerLabel,
        ));
        divider(panel, border);
        panel.spawn((
            TeamStatsContainer,
            Node {
                flex_direction:  FlexDirection::Row,
                justify_content: JustifyContent::SpaceAround,
                width:           Val::Percent(100.0),
                column_gap:      Val::Px(32.0),
                ..default()
            },
        ));
        divider(panel, border);
        panel.spawn(Node {
            flex_direction: FlexDirection::Row,
            column_gap:     Val::Px(16.0),
            ..default()
        }).with_children(|row| {
            row.spawn((
                Button, RestartButton,
                Node {
                    padding:         UiRect::axes(Val::Px(36.0), Val::Px(12.0)),
                    justify_content: JustifyContent::Center,
                    align_items:     AlignItems::Center,
                    border:          UiRect::all(Val::Px(1.0)),
                    border_radius:   BorderRadius::all(Val::Px(8.0)),
                    ..default()
                },
                BackgroundColor(ThemeColor::ButtonIdle.resolve()),
                BorderColor::all(ThemeColor::SuccessText.resolve()),
            )).with_children(|btn| {
                btn.spawn((
                    Text::new("RESTART"),
                    TextFont  { font_size: SIZE_LG, ..default() },
                    TextColor(ThemeColor::SuccessText.resolve()),
                ));
            });
            row.spawn((
                Button, QuitButton,
                Node {
                    padding:         UiRect::axes(Val::Px(36.0), Val::Px(12.0)),
                    justify_content: JustifyContent::Center,
                    align_items:     AlignItems::Center,
                    border:          UiRect::all(Val::Px(1.0)),
                    border_radius:   BorderRadius::all(Val::Px(8.0)),
                    ..default()
                },
                BackgroundColor(ThemeColor::ButtonIdle.resolve()),
                BorderColor::all(border),
            )).with_children(|btn| {
                btn.spawn((
                    Text::new("QUIT"),
                    TextFont  { font_size: SIZE_LG, ..default() },
                    TextColor(ThemeColor::TextPrimary.resolve()),
                ));
            });
        });
    });
}

pub fn show_end_screen(
    sim:          Res<SimConfig>,
    team_score:   Res<TeamScore>,
    mut screen_q: Query<(&mut Node, &mut Visibility), With<EndScreen>>,
    mut winner_q: Query<&mut Text, With<WinnerLabel>>,
) {
    if !sim.is_changed() || !sim.game_over { return; }

    for (mut node, mut vis) in screen_q.iter_mut() {
        node.display = Display::Flex;
        *vis         = Visibility::Visible;
    }

    let team_ids: Vec<u8> = vec![0, 1];
    let winner_id = team_ids.iter()
        .max_by_key(|&&id| team_score.get(Team(id)));

    if let Ok(mut text) = winner_q.single_mut() {
        *text = match winner_id {
            Some(&id) => {
                let winning_score = team_score.get(Team(id));
                let tied = team_ids.iter()
                    .filter(|&&other| team_score.get(Team(other)) == winning_score)
                    .count() > 1;
                if tied {
                    Text::new("It's a draw!")
                } else {
                    Text::new(format!("Winner: {} Team  🏆", Team(id).name()))
                }
            }
            None => Text::new(""),
        };
    }
}

pub fn populate_end_screen_cards(
    sim:          Res<SimConfig>,
    team_score:   Res<TeamScore>,
    agents:       Query<(&Team, &Score, &KillCount, &DeathCount)>,
    container_q:  Query<Entity, (With<TeamStatsContainer>, Without<CardsPopulated>)>,
    mut commands: Commands,
) {
    if !sim.game_over { return; }
    let Ok(container) = container_q.single() else { return };
    commands.entity(container).insert(CardsPopulated);

    use std::collections::HashMap;
    struct TeamData { kills: u32, deaths: u32 }
    let mut map: HashMap<u8, TeamData> = HashMap::new();

    for (team, _score, kills, deaths) in agents.iter() {
        let d = map.entry(team.0).or_insert(TeamData { kills: 0, deaths: 0 });
        d.kills  += kills.0;
        d.deaths += deaths.0;
    }

    let mut team_ids: Vec<u8> = map.keys().copied().collect();
    for id in [0u8, 1u8] { if !team_ids.contains(&id) { team_ids.push(id); } }
    team_ids.sort();

    commands.entity(container).with_children(|c| {
        for &team_id in &team_ids {
            let tc     = team_color(team_id);
            let ts     = team_score.get(Team(team_id));
            let kills  = map.get(&team_id).map(|d| d.kills).unwrap_or(0);
            let deaths = map.get(&team_id).map(|d| d.deaths).unwrap_or(0);
            let kd     = if deaths == 0 { kills as f32 }
            else { kills as f32 / deaths as f32 };

            c.spawn(Node {
                flex_direction: FlexDirection::Column,
                align_items:    AlignItems::Center,
                row_gap:        Val::Px(8.0),
                padding:        UiRect::all(Val::Px(16.0)),
                border:         UiRect::all(Val::Px(1.0)),
                border_radius:  BorderRadius::all(Val::Px(8.0)),
                min_width:      Val::Px(180.0),
                ..default()
            }).with_children(|card| {
                card.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    align_items:    AlignItems::Center,
                    column_gap:     Val::Px(8.0),
                    ..default()
                }).with_children(|h| {
                    h.spawn((
                        Node {
                            width:         Val::Px(10.0),
                            height:        Val::Px(10.0),
                            border_radius: BorderRadius::all(Val::Px(5.0)),
                            ..default()
                        },
                        BackgroundColor(tc),
                    ));
                    h.spawn((
                        Text::new(Team(team_id).name().to_uppercase()),
                        TextFont  { font_size: SIZE_MD, ..default() },
                        TextColor(tc),
                    ));
                });
                stat_row(card, "Score",  &ts.to_string(),       ThemeColor::SuccessText.resolve());
                stat_row(card, "Kills",  &kills.to_string(),    ThemeColor::TextPrimary.resolve());
                stat_row(card, "Deaths", &deaths.to_string(),   ThemeColor::TextPrimary.resolve());
                stat_row(card, "K/D",    &format!("{:.2}", kd), ThemeColor::TextPrimary.resolve());
            });
        }
    });
}

pub fn handle_restart_button(
    mut restart:  MessageWriter<RestartEvent>,
    container_q:  Query<Entity, With<TeamStatsContainer>>,
    mut commands: Commands,
    buttons:      Query<&Interaction, (Changed<Interaction>, With<RestartButton>)>,
) {
    for interaction in buttons.iter() {
        if *interaction == Interaction::Pressed {
            if let Ok(container) = container_q.single() {
                commands.entity(container)
                    .remove::<CardsPopulated>()
                    .despawn_related::<Children>();
            }
            restart.write(RestartEvent);
        }
    }
}

pub fn handle_quit_button(
    mut app_exit: MessageWriter<AppExit>,
    buttons:      Query<&Interaction, (Changed<Interaction>, With<QuitButton>)>,
) {
    for interaction in buttons.iter() {
        if *interaction == Interaction::Pressed {
            app_exit.write(AppExit::Success);
        }
    }
}

fn divider(parent: &mut ChildSpawnerCommands, color: Color) {
    parent.spawn((
        Node { width: Val::Percent(100.0), height: Val::Px(1.0), ..default() },
        BackgroundColor(color),
    ));
}

fn stat_row(parent: &mut ChildSpawnerCommands, label: &str, value: &str, value_color: Color) {
    let dim = ThemeColor::TextDim.resolve();
    parent.spawn(Node {
        flex_direction:  FlexDirection::Row,
        justify_content: JustifyContent::SpaceBetween,
        width:           Val::Percent(100.0),
        column_gap:      Val::Px(16.0),
        ..default()
    }).with_children(|row| {
        row.spawn((
            Text::new(label),
            TextFont  { font_size: SIZE_SM, ..default() },
            TextColor(dim),
        ));
        row.spawn((
            Text::new(value),
            TextFont  { font_size: SIZE_MD, ..default() },
            TextColor(value_color),
        ));
    });
}