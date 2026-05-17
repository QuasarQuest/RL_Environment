// src/viz/hud/scoreboard.rs

use bevy::prelude::*;
use crate::agent::components::{Ammo, GoldCarried, Hearts, RespawnIn, Score};
use crate::viz::components::{AgentInfo, AgentLabel, HidePathViz, HideRangeViz};
use crate::style::{ThemeColor, UiRoot, SIZE_SM, SIZE_MD, SIZE_LG, TOOLBAR_H};
use crate::style::color::team_color;
use crate::team::Team;
use super::components::{
    TabScoreboard, TabScoreboardContent, ScoreboardRow,
    ScoreboardRowHp, ScoreboardRowAmmo, ScoreboardRowGold, ScoreboardRowScore,
    ScoreboardRowRangeLabel, ScoreboardRowPathLabel,
};

// ── Spawn ─────────────────────────────────────────────────────────────────────

pub fn spawn_tab_scoreboard(mut commands: Commands) {
    let bg     = ThemeColor::Background.resolve();
    let border = ThemeColor::Border.resolve();
    let dim    = ThemeColor::TextDim.resolve();

    commands.spawn((
        UiRoot,
        TabScoreboard,
        Node {
            display:        Display::None,
            position_type:  PositionType::Absolute,
            top:            Val::Px(TOOLBAR_H + 8.0),
            left:           Val::Percent(50.0),
            margin:         UiRect::left(Val::Px(-360.0)),
            flex_direction: FlexDirection::Column,
            min_width:      Val::Px(720.0),
            border:         UiRect::all(Val::Px(1.0)),
            border_radius:  BorderRadius::all(Val::Px(10.0)),
            overflow:       Overflow::clip(),
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(border),
        ZIndex(200),
    )).with_children(|panel| {
        // Header row
        panel.spawn(Node {
            flex_direction: FlexDirection::Row,
            padding:        UiRect::new(Val::Px(16.0), Val::Px(16.0), Val::Px(10.0), Val::Px(8.0)),
            border:         UiRect::bottom(Val::Px(1.0)),
            ..default()
        }).with_children(|h| {
            hdr(h, "TEAM",  50.0,  dim);
            hdr(h, "AGENT", 200.0, dim);
            hdr(h, "HP",    55.0,  dim);
            hdr(h, "AMMO",  50.0,  dim);
            hdr(h, "GOLD",  50.0,  ThemeColor::AccentGold.resolve());
            hdr(h, "SCORE", 60.0,  dim);
            hdr(h, "RANGE", 60.0,  dim);
            hdr(h, "PATH",  60.0,  dim);
        });

        panel.spawn((
            TabScoreboardContent,
            Node { flex_direction: FlexDirection::Column, ..default() },
        ));

        // Footer hint
        panel.spawn(Node {
            flex_direction:  FlexDirection::Row,
            justify_content: JustifyContent::Center,
            padding:         UiRect::axes(Val::Px(0.0), Val::Px(6.0)),
            border:          UiRect::top(Val::Px(1.0)),
            ..default()
        }).with_children(|f| {
            f.spawn((
                Text::new("Hold TAB  |  RANGE = combat rings  |  PATH = agent route  |  H for shortcuts"),
                TextFont  { font_size: SIZE_SM, ..default() },
                TextColor(dim),
            ));
        });
    });
}

// ── Tab toggle ────────────────────────────────────────────────────────────────

pub fn toggle_tab_scoreboard(
    keys:  Res<ButtonInput<KeyCode>>,
    mut q: Query<&mut Node, With<TabScoreboard>>,
) {
    let visible = keys.pressed(KeyCode::Tab);
    for mut node in q.iter_mut() {
        node.display = if visible { Display::Flex } else { Display::None };
    }
}

// ── Viz toggles ───────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct RangeVizToggleButton(pub Entity);

#[derive(Component)]
pub struct PathVizToggleButton(pub Entity);

pub fn handle_viz_toggle(
    mut commands:  Commands,
    range_buttons: Query<(&Interaction, &RangeVizToggleButton), Changed<Interaction>>,
    path_buttons:  Query<(&Interaction, &PathVizToggleButton),  Changed<Interaction>>,
    range_hidden:  Query<Has<HideRangeViz>>,
    path_hidden:   Query<Has<HidePathViz>>,
) {
    for (interaction, btn) in range_buttons.iter() {
        if *interaction != Interaction::Pressed { continue; }
        let entity = btn.0;
        if range_hidden.get(entity).unwrap_or(false) {
            commands.entity(entity).remove::<HideRangeViz>();
        } else {
            commands.entity(entity).insert(HideRangeViz);
        }
    }
    for (interaction, btn) in path_buttons.iter() {
        if *interaction != Interaction::Pressed { continue; }
        let entity = btn.0;
        if path_hidden.get(entity).unwrap_or(false) {
            commands.entity(entity).remove::<HidePathViz>();
        } else {
            commands.entity(entity).insert(HidePathViz);
        }
    }
}

// ── Build rows (once on open / agent count change) ────────────────────────────

pub fn build_scoreboard_rows(
    scoreboard_q: Query<&Node, With<TabScoreboard>>,
    content_q:    Query<Entity, With<TabScoreboardContent>>,
    row_q:        Query<(), With<ScoreboardRow>>,
    agents:       Query<(
        Entity, &AgentLabel, &AgentInfo, &Team,
        &Hearts, &Ammo, &GoldCarried, &Score,
        Option<&RespawnIn>, Has<HideRangeViz>, Has<HidePathViz>,
    )>,
    mut commands: Commands,
) {
    let Ok(node) = scoreboard_q.single() else { return };
    if node.display == Display::None { return; }
    if agents.iter().count() == row_q.iter().count() { return; }

    let Ok(content) = content_q.single() else { return };
    commands.entity(content).despawn_related::<Children>();

    let mut rows: Vec<_> = agents.iter().collect();
    rows.sort_by(|a, b| a.3.0.cmp(&b.3.0).then(b.7.0.cmp(&a.7.0)));

    let dim     = ThemeColor::TextDim.resolve();
    let primary = ThemeColor::TextPrimary.resolve();
    let gold_c  = ThemeColor::AccentGold.resolve();
    let score_c = ThemeColor::SuccessText.resolve();

    commands.entity(content).with_children(|c| {
        for (i, (agent_entity, label, info, team, hearts, ammo, gold, score, respawning, range_hidden, path_hidden))
        in rows.iter().enumerate()
        {
            let bg         = if i % 2 == 0 { Color::NONE } else { ThemeColor::SurfaceHighlight.resolve() };
            let tcolor     = team_color(team.0);
            let name_color = if respawning.is_some() { dim } else { primary };
            let hp_str     = hp_string(hearts, respawning);
            let hp_col     = hp_color_val(hearts, respawning);
            let info_text  = format!("{} · {}", info.strategy, info.planner);

            c.spawn((
                ScoreboardRow(*agent_entity),
                Node {
                    flex_direction: FlexDirection::Row,
                    padding:        UiRect::axes(Val::Px(16.0), Val::Px(6.0)),
                    align_items:    AlignItems::Center,
                    ..default()
                },
                BackgroundColor(bg),
            )).with_children(|row| {
                // Team dot
                row.spawn((
                    Node {
                        width:         Val::Px(10.0),
                        height:        Val::Px(10.0),
                        border_radius: BorderRadius::all(Val::Px(5.0)),
                        margin:        UiRect::right(Val::Px(40.0)),
                        ..default()
                    },
                    BackgroundColor(tcolor),
                ));

                // Name + strategy (static)
                row.spawn(Node {
                    flex_direction: FlexDirection::Column,
                    width:          Val::Px(200.0),
                    row_gap:        Val::Px(2.0),
                    ..default()
                }).with_children(|col| {
                    col.spawn((
                        Text::new(&label.0),
                        TextFont  { font_size: SIZE_MD, ..default() },
                        TextColor(name_color),
                    ));
                    col.spawn((
                        Text::new(&info_text),
                        TextFont  { font_size: 9.0, ..default() },
                        TextColor(dim),
                    ));
                });

                // Data cells (updated each frame by refresh_scoreboard_rows)
                row.spawn((
                    Text::new(hp_str),
                    TextFont  { font_size: SIZE_MD, ..default() },
                    TextColor(hp_col),
                    Node { width: Val::Px(55.0), ..default() },
                    ScoreboardRowHp(*agent_entity),
                ));
                row.spawn((
                    Text::new(ammo.0.to_string()),
                    TextFont  { font_size: SIZE_MD, ..default() },
                    TextColor(primary),
                    Node { width: Val::Px(50.0), ..default() },
                    ScoreboardRowAmmo(*agent_entity),
                ));
                row.spawn((
                    Text::new(gold.0.to_string()),
                    TextFont  { font_size: SIZE_MD, ..default() },
                    TextColor(gold_c),
                    Node { width: Val::Px(50.0), ..default() },
                    ScoreboardRowGold(*agent_entity),
                ));
                row.spawn((
                    Text::new(score.0.to_string()),
                    TextFont  { font_size: SIZE_LG, ..default() },
                    TextColor(score_c),
                    Node { width: Val::Px(60.0), ..default() },
                    ScoreboardRowScore(*agent_entity),
                ));

                // RANGE toggle button
                viz_button(row, *agent_entity, *range_hidden, dim, score_c,
                           RangeVizToggleButton(*agent_entity),
                           ScoreboardRowRangeLabel(*agent_entity),
                );

                // PATH toggle button
                viz_button(row, *agent_entity, *path_hidden, dim, score_c,
                           PathVizToggleButton(*agent_entity),
                           ScoreboardRowPathLabel(*agent_entity),
                );
            });
        }
    });
}

// ── Refresh cells every frame while open ─────────────────────────────────────

pub fn refresh_scoreboard_rows(
    scoreboard_q: Query<&Node, With<TabScoreboard>>,
    agents:       Query<(
        Entity, &Hearts, &Ammo, &GoldCarried, &Score,
        Option<&RespawnIn>, Has<HideRangeViz>, Has<HidePathViz>,
    )>,
    mut hp_q:    Query<(&mut Text, &mut TextColor, &ScoreboardRowHp)>,
    mut ammo_q:  Query<(&mut Text, &ScoreboardRowAmmo),
        Without<ScoreboardRowHp>>,
    mut gold_q:  Query<(&mut Text, &ScoreboardRowGold),
        (Without<ScoreboardRowHp>, Without<ScoreboardRowAmmo>)>,
    mut score_q: Query<(&mut Text, &ScoreboardRowScore),
        (Without<ScoreboardRowHp>, Without<ScoreboardRowAmmo>, Without<ScoreboardRowGold>)>,
    mut range_q: Query<(&mut Text, &mut TextColor, &ScoreboardRowRangeLabel),
        (Without<ScoreboardRowHp>, Without<ScoreboardRowAmmo>, Without<ScoreboardRowGold>, Without<ScoreboardRowScore>)>,
    mut path_q:  Query<(&mut Text, &mut TextColor, &ScoreboardRowPathLabel),
        (Without<ScoreboardRowHp>, Without<ScoreboardRowAmmo>, Without<ScoreboardRowGold>, Without<ScoreboardRowScore>, Without<ScoreboardRowRangeLabel>)>,
) {
    let Ok(node) = scoreboard_q.single() else { return };
    if node.display == Display::None { return; }

    let dim     = ThemeColor::TextDim.resolve();
    let score_c = ThemeColor::SuccessText.resolve();

    for (entity, hearts, ammo, gold, score, respawning, range_hidden, path_hidden) in agents.iter() {
        let hp_str = hp_string(hearts, &respawning);
        let hp_col = hp_color_val(hearts, &respawning);

        for (mut text, mut color, marker) in hp_q.iter_mut() {
            if marker.0 != entity { continue; }
            *text  = Text::new(&hp_str);
            *color = TextColor(hp_col);
        }
        for (mut text, marker) in ammo_q.iter_mut() {
            if marker.0 == entity { *text = Text::new(ammo.0.to_string()); }
        }
        for (mut text, marker) in gold_q.iter_mut() {
            if marker.0 == entity { *text = Text::new(gold.0.to_string()); }
        }
        for (mut text, marker) in score_q.iter_mut() {
            if marker.0 == entity { *text = Text::new(score.0.to_string()); }
        }
        for (mut text, mut color, marker) in range_q.iter_mut() {
            if marker.0 != entity { continue; }
            *text  = Text::new(if range_hidden { "OFF" } else { "ON" });
            *color = TextColor(if range_hidden { dim } else { score_c });
        }
        for (mut text, mut color, marker) in path_q.iter_mut() {
            if marker.0 != entity { continue; }
            *text  = Text::new(if path_hidden { "OFF" } else { "ON" });
            *color = TextColor(if path_hidden { dim } else { score_c });
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn viz_button(
    row:         &mut ChildSpawnerCommands,
    _agent:      Entity,
    is_hidden:   bool,
    dim:         Color,
    on_color:    Color,
    btn_marker:  impl Bundle,
    text_marker: impl Bundle,
) {
    let label = if is_hidden { "OFF" } else { "ON" };
    let color = if is_hidden { dim } else { on_color };
    row.spawn((
        Button,
        btn_marker,
        Node {
            width:           Val::Px(44.0),
            height:          Val::Px(22.0),
            justify_content: JustifyContent::Center,
            align_items:     AlignItems::Center,
            border:          UiRect::all(Val::Px(1.0)),
            border_radius:   BorderRadius::all(Val::Px(4.0)),
            margin:          UiRect::right(Val::Px(8.0)),
            ..default()
        },
        BackgroundColor(ThemeColor::ButtonIdle.resolve()),
        BorderColor::all(ThemeColor::Border.resolve()),
    )).with_children(|btn| {
        btn.spawn((
            Text::new(label),
            TextFont  { font_size: SIZE_SM, ..default() },
            TextColor(color),
            text_marker,
        ));
    });
}

fn hp_string(hearts: &Hearts, respawning: &Option<&RespawnIn>) -> String {
    if respawning.is_some() { "dead".into() }
    else { format!("{}/{}", hearts.0, crate::config::AGENT_MAX_HEARTS) }
}

fn hp_color_val(hearts: &Hearts, respawning: &Option<&RespawnIn>) -> Color {
    if respawning.is_some() { return Color::srgb(0.6, 0.6, 0.6); }
    match hearts.0 {
        0 => Color::srgb(0.6,  0.6,  0.6),
        1 => Color::srgb(0.85, 0.25, 0.20),
        2 => Color::srgb(0.95, 0.60, 0.10),
        _ => Color::srgb(0.20, 0.75, 0.35),
    }
}

fn hdr(parent: &mut ChildSpawnerCommands, text: &str, width: f32, color: Color) {
    parent.spawn((
        Text::new(text),
        TextFont  { font_size: SIZE_SM, ..default() },
        TextColor(color),
        Node { width: Val::Px(width), ..default() },
    ));
}