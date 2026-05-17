// src/viz/hud/scoreboard.rs

use bevy::prelude::*;
use crate::agent::components::{
    Ammo, DeathCount, GoldCarried, Hearts, KillCount, RespawnIn, Score,
};
use crate::viz::components::{AgentInfo, AgentLabel, HidePathViz, HideRangeViz};
use crate::style::{ThemeColor, UiRoot, SIZE_SM, SIZE_LG, SIZE_XL, TOOLBAR_H};
use crate::style::color::{team_color, SURFACE_HIGHLIGHT};
use crate::team::{Team, TeamScore};
use super::components::{
    TabScoreboard, TabScoreboardContent, ScoreboardRow,
    ScoreboardTeamScore,
    ScoreboardAvgScore, ScoreboardAvgKills,
    ScoreboardAvgDeaths, ScoreboardAvgKd,
    ScoreboardRowHp, ScoreboardRowAmmo, ScoreboardRowGold, ScoreboardRowScore,
    ScoreboardRowKills, ScoreboardRowDeaths, ScoreboardRowKd,
    ScoreboardRowRangeLabel, ScoreboardRowPathLabel,
};

// ── Layout constants ──────────────────────────────────────────────────────────

const F_AGENT: f32 = 15.0;
const F_SUB:   f32 = 11.0;
const F_STAT:  f32 = 15.0;
const F_SCORE: f32 = 18.0;
const F_BTN:   f32 = 12.0;
const F_AVG:   f32 = 16.0;

const W_AGENT: f32 = 230.0;
const W_HP:    f32 =  62.0;
const W_AMMO:  f32 =  56.0;
const W_GOLD:  f32 =  56.0;
const W_SCORE: f32 =  72.0;
const W_K:     f32 =  44.0;
const W_D:     f32 =  44.0;
const W_KD:    f32 =  60.0;
const W_RANGE: f32 =  64.0;
const W_PATH:  f32 =  64.0;
const SEP_W:   f32 =  13.0;
const TOTAL_W: f32 = 16.0 + W_AGENT + W_HP + W_AMMO + W_GOLD + W_SCORE
    + W_K + W_D + W_KD + SEP_W + W_RANGE + W_PATH + 16.0;

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
            margin:         UiRect::left(Val::Px(-TOTAL_W / 2.0)),
            flex_direction: FlexDirection::Column,
            min_width:      Val::Px(TOTAL_W),
            border:         UiRect::all(Val::Px(1.0)),
            border_radius:  BorderRadius::all(Val::Px(10.0)),
            overflow:       Overflow::clip(),
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(border),
        ZIndex(200),
    )).with_children(|panel| {
        // ── Column header — XL font ───────────────────────────────────────────
        panel.spawn((
            Node {
                flex_direction: FlexDirection::Row,
                align_items:    AlignItems::Center,
                padding:        UiRect::new(Val::Px(16.0), Val::Px(16.0), Val::Px(10.0), Val::Px(8.0)),
                border:         UiRect::bottom(Val::Px(1.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.02)),
            BorderColor::all(border),
        )).with_children(|h| {
            hdr(h, "AGENT",  W_AGENT, dim,                              SIZE_XL);
            hdr(h, "HP",     W_HP,    dim,                              SIZE_XL);
            hdr(h, "AMMO",   W_AMMO,  dim,                              SIZE_XL);
            hdr(h, "GOLD",   W_GOLD,  ThemeColor::AccentGold.resolve(), SIZE_XL);
            hdr(h, "SCORE",  W_SCORE, dim,                              SIZE_XL);
            hdr(h, "K",      W_K,     dim,                              SIZE_XL);
            hdr(h, "D",      W_D,     dim,                              SIZE_XL);
            hdr(h, "K/D",    W_KD,    dim,                              SIZE_XL);
            h.spawn(Node { width: Val::Px(SEP_W), ..default() });
            hdr(h, "RANGE",  W_RANGE, dim,                              SIZE_XL);
            hdr(h, "PATH",   W_PATH,  dim,                              SIZE_XL);
        });

        panel.spawn((
            TabScoreboardContent,
            Node { flex_direction: FlexDirection::Column, ..default() },
        ));

        // Footer
        panel.spawn((
            Node {
                flex_direction:  FlexDirection::Row,
                justify_content: JustifyContent::Center,
                padding:         UiRect::axes(Val::Px(0.0), Val::Px(6.0)),
                border:          UiRect::top(Val::Px(1.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.02)),
            BorderColor::all(border),
        )).with_children(|f| {
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

// ── Build rows ────────────────────────────────────────────────────────────────

pub fn build_scoreboard_rows(
    scoreboard_q: Query<&Node, With<TabScoreboard>>,
    content_q:    Query<Entity, With<TabScoreboardContent>>,
    row_q:        Query<(), With<ScoreboardRow>>,
    agents:       Query<(
        Entity, &AgentLabel, &AgentInfo, &Team,
        &Hearts, &Ammo, &GoldCarried, &Score,
        &KillCount, &DeathCount,
        Option<&RespawnIn>, Has<HideRangeViz>, Has<HidePathViz>,
    )>,
    team_score:   Res<TeamScore>,
    mut commands: Commands,
) {
    let Ok(node) = scoreboard_q.single() else { return };
    if node.display == Display::None { return; }
    if agents.iter().count() == row_q.iter().count() { return; }

    let Ok(content) = content_q.single() else { return };
    commands.entity(content).despawn_related::<Children>();

    let mut all_rows: Vec<_> = agents.iter().collect();
    all_rows.sort_by(|a, b| a.3.0.cmp(&b.3.0).then(b.7.0.cmp(&a.7.0)));

    let mut team_ids: Vec<u8> = all_rows.iter().map(|(_, _, _, t, ..)| t.0).collect();
    team_ids.sort();
    team_ids.dedup();

    let dim     = ThemeColor::TextDim.resolve();
    let primary = ThemeColor::TextPrimary.resolve();
    let gold_c  = ThemeColor::AccentGold.resolve();
    let score_c = ThemeColor::SuccessText.resolve();
    let border  = ThemeColor::Border.resolve();

    commands.entity(content).with_children(|c| {
        for team_id in &team_ids {
            let tcolor    = team_color(*team_id);
            let team_name = Team(*team_id).name().to_uppercase();
            let ts        = team_score.get(Team(*team_id));

            let team_rows: Vec<_> = all_rows.iter()
                .filter(|(_, _, _, t, ..)| t.0 == *team_id)
                .collect();

            // Compute initial avgs for spawn-time values
            let count  = team_rows.len() as f32;
            let avg_k  = team_rows.iter().map(|(_, _, _, _, _, _, _, _, k, ..)| k.0 as f32).sum::<f32>() / count;
            let avg_d  = team_rows.iter().map(|(_, _, _, _, _, _, _, _, _, d, ..)| d.0 as f32).sum::<f32>() / count;
            let avg_kd = if avg_d == 0.0 { avg_k } else { avg_k / avg_d };

            // ── Team header with embedded avg ─────────────────────────────────
            c.spawn((
                Node {
                    flex_direction: FlexDirection::Row,
                    align_items:    AlignItems::Center,
                    padding:        UiRect::new(Val::Px(16.0), Val::Px(16.0), Val::Px(8.0), Val::Px(8.0)),
                    border:         UiRect::bottom(Val::Px(1.0)),
                    ..default()
                },
                BackgroundColor(tcolor.with_alpha(0.10)),
                BorderColor::all(tcolor.with_alpha(0.25)),
            )).with_children(|h| {
                // Left: dot + team name only
                h.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    align_items:    AlignItems::Center,
                    width:          Val::Px(W_AGENT),
                    column_gap:     Val::Px(10.0),
                    ..default()
                }).with_children(|left| {
                    left.spawn((
                        Node {
                            width:         Val::Px(12.0),
                            height:        Val::Px(12.0),
                            border_radius: BorderRadius::all(Val::Px(6.0)),
                            flex_shrink:   0.0,
                            ..default()
                        },
                        BackgroundColor(tcolor),
                    ));
                    left.spawn((
                        Text::new(&team_name),
                        TextFont  { font_size: SIZE_LG, ..default() },
                        TextColor(tcolor),
                    ));
                });

                // HP / Ammo / Gold — spacers only
                h.spawn(Node { width: Val::Px(W_HP),   ..default() });
                h.spawn(Node { width: Val::Px(W_AMMO), ..default() });
                h.spawn(Node { width: Val::Px(W_GOLD), ..default() });

                // Team score — F_SCORE so it's bigger than agent stat font
                h.spawn((
                    Text::new(ts.to_string()),
                    TextFont  { font_size: F_SCORE, ..default() },
                    TextColor(score_c),
                    Node { width: Val::Px(W_SCORE), ..default() },
                    ScoreboardTeamScore(*team_id),
                ));

                // Avg K
                avg_header_cell(h, &format!("{:.1}", avg_k),
                                W_K, tcolor.with_alpha(0.70), ScoreboardAvgKills(*team_id));
                // Avg D
                avg_header_cell(h, &format!("{:.1}", avg_d),
                                W_D, tcolor.with_alpha(0.70), ScoreboardAvgDeaths(*team_id));
                // Avg K/D
                avg_header_cell(h, &format!("{:.2}", avg_kd),
                                W_KD, tcolor.with_alpha(0.70), ScoreboardAvgKd(*team_id));

                // Separator gap + empty range/path space
                h.spawn(Node { width: Val::Px(SEP_W + W_RANGE + W_PATH), ..default() });
            });

            // ── Agent rows ────────────────────────────────────────────────────
            for (i, (agent_entity, label, info, _, hearts, ammo, gold, score,
                kills, deaths, respawning, range_hidden, path_hidden))
            in team_rows.iter().enumerate()
            {
                // Alternate: even = slightly lighter, odd = base dark
                let bg = if i % 2 == 0 {
                    SURFACE_HIGHLIGHT
                } else {
                    Color::NONE
                };
                let name_col  = if respawning.is_some() { dim } else { primary };
                let hp_str    = hp_string(hearts, respawning);
                let hp_col    = hp_color_val(hearts, respawning);
                let kd_str    = kd_ratio(kills.0, deaths.0);
                let info_text = format!("{} · {}", info.strategy, info.planner);

                c.spawn((
                    ScoreboardRow,
                    Node {
                        flex_direction: FlexDirection::Row,
                        align_items:    AlignItems::Center,
                        padding:        UiRect::axes(Val::Px(16.0), Val::Px(9.0)),
                        ..default()
                    },
                    BackgroundColor(bg),
                )).with_children(|row| {
                    row.spawn(Node {
                        flex_direction: FlexDirection::Column,
                        width:          Val::Px(W_AGENT),
                        row_gap:        Val::Px(3.0),
                        ..default()
                    }).with_children(|col| {
                        col.spawn((
                            Text::new(&label.0),
                            TextFont  { font_size: F_AGENT, ..default() },
                            TextColor(name_col),
                        ));
                        col.spawn((
                            Text::new(&info_text),
                            TextFont  { font_size: F_SUB, ..default() },
                            TextColor(dim),
                        ));
                    });

                    data_cell(row, &hp_str,               W_HP,    hp_col,  F_STAT,  ScoreboardRowHp(*agent_entity));
                    data_cell(row, &ammo.0.to_string(),   W_AMMO,  primary, F_STAT,  ScoreboardRowAmmo(*agent_entity));
                    data_cell(row, &gold.0.to_string(),   W_GOLD,  gold_c,  F_STAT,  ScoreboardRowGold(*agent_entity));
                    data_cell(row, &score.0.to_string(),  W_SCORE, score_c, F_SCORE, ScoreboardRowScore(*agent_entity));
                    data_cell(row, &kills.0.to_string(),  W_K,     primary, F_STAT,  ScoreboardRowKills(*agent_entity));
                    data_cell(row, &deaths.0.to_string(), W_D,     primary, F_STAT,  ScoreboardRowDeaths(*agent_entity));
                    data_cell(row, &kd_str,               W_KD,    primary, F_STAT,  ScoreboardRowKd(*agent_entity));

                    // Separator
                    row.spawn((
                        Node {
                            width:  Val::Px(1.0),
                            height: Val::Px(28.0),
                            margin: UiRect::horizontal(Val::Px(6.0)),
                            ..default()
                        },
                        BackgroundColor(border),
                    ));

                    viz_button(row, *range_hidden, dim, score_c,
                               RangeVizToggleButton(*agent_entity),
                               ScoreboardRowRangeLabel(*agent_entity), W_RANGE);
                    viz_button(row, *path_hidden, dim, score_c,
                               PathVizToggleButton(*agent_entity),
                               ScoreboardRowPathLabel(*agent_entity), W_PATH);
                });
            }

            // Team divider
            c.spawn((
                Node { height: Val::Px(2.0), ..default() },
                BackgroundColor(tcolor.with_alpha(0.25)),
            ));
        }
    });
}

// ── Refresh: agent stats + team score ────────────────────────────────────────

pub fn refresh_scoreboard_stats(
    scoreboard_q: Query<&Node, With<TabScoreboard>>,
    team_score:   Res<TeamScore>,
    agents:       Query<(
        Entity, &Hearts, &Ammo, &GoldCarried, &Score,
        &KillCount, &DeathCount, Option<&RespawnIn>,
    )>,
    mut cells: ParamSet<(
        Query<(&mut Text, &ScoreboardTeamScore)>,
        Query<(&mut Text, &mut TextColor, &ScoreboardRowHp)>,
        Query<(&mut Text, &ScoreboardRowAmmo)>,
        Query<(&mut Text, &ScoreboardRowGold)>,
        Query<(&mut Text, &ScoreboardRowScore)>,
        Query<(&mut Text, &ScoreboardRowKills)>,
        Query<(&mut Text, &ScoreboardRowDeaths)>,
        Query<(&mut Text, &ScoreboardRowKd)>,
    )>,
) {
    let Ok(node) = scoreboard_q.single() else { return };
    if node.display == Display::None { return; }

    for (mut text, marker) in cells.p0().iter_mut() {
        *text = Text::new(team_score.get(Team(marker.0)).to_string());
    }

    let data: Vec<_> = agents.iter().map(|(e, h, a, g, s, k, d, r)| {
        (e, hp_string(h, &r), hp_color_val(h, &r),
         a.0, g.0, s.0, k.0, d.0, kd_ratio(k.0, d.0))
    }).collect();

    for (entity, hp_str, hp_col, ammo, gold, score, kills, deaths, kd_str) in &data {
        for (mut text, mut color, marker) in cells.p1().iter_mut() {
            if marker.0 != *entity { continue; }
            *text  = Text::new(hp_str.as_str());
            *color = TextColor(*hp_col);
        }
        for (mut text, marker) in cells.p2().iter_mut() {
            if marker.0 == *entity { *text = Text::new(ammo.to_string()); }
        }
        for (mut text, marker) in cells.p3().iter_mut() {
            if marker.0 == *entity { *text = Text::new(gold.to_string()); }
        }
        for (mut text, marker) in cells.p4().iter_mut() {
            if marker.0 == *entity { *text = Text::new(score.to_string()); }
        }
        for (mut text, marker) in cells.p5().iter_mut() {
            if marker.0 == *entity { *text = Text::new(kills.to_string()); }
        }
        for (mut text, marker) in cells.p6().iter_mut() {
            if marker.0 == *entity { *text = Text::new(deaths.to_string()); }
        }
        for (mut text, marker) in cells.p7().iter_mut() {
            if marker.0 == *entity { *text = Text::new(kd_str.as_str()); }
        }
    }
}

// ── Refresh: viz toggle labels ────────────────────────────────────────────────

pub fn refresh_scoreboard_viz(
    scoreboard_q: Query<&Node, With<TabScoreboard>>,
    agents:       Query<(Entity, Has<HideRangeViz>, Has<HidePathViz>)>,
    mut viz: ParamSet<(
        Query<(&mut Text, &mut TextColor, &ScoreboardRowRangeLabel)>,
        Query<(&mut Text, &mut TextColor, &ScoreboardRowPathLabel)>,
    )>,
) {
    let Ok(node) = scoreboard_q.single() else { return };
    if node.display == Display::None { return; }

    let dim     = ThemeColor::TextDim.resolve();
    let score_c = ThemeColor::SuccessText.resolve();
    let data: Vec<_> = agents.iter().collect();

    for (entity, range_hidden, _) in &data {
        for (mut text, mut color, marker) in viz.p0().iter_mut() {
            if marker.0 != *entity { continue; }
            *text  = Text::new(if *range_hidden { "OFF" } else { "ON" });
            *color = TextColor(if *range_hidden { dim } else { score_c });
        }
    }
    for (entity, _, path_hidden) in &data {
        for (mut text, mut color, marker) in viz.p1().iter_mut() {
            if marker.0 != *entity { continue; }
            *text  = Text::new(if *path_hidden { "OFF" } else { "ON" });
            *color = TextColor(if *path_hidden { dim } else { score_c });
        }
    }
}

// ── Refresh: team avg cells in header ────────────────────────────────────────

pub fn refresh_scoreboard_avg(
    scoreboard_q: Query<&Node, With<TabScoreboard>>,
    agents:       Query<(&Team, &Score, &KillCount, &DeathCount)>,
    mut avg: ParamSet<(
        Query<(&mut Text, &ScoreboardAvgScore)>,
        Query<(&mut Text, &ScoreboardAvgKills)>,
        Query<(&mut Text, &ScoreboardAvgDeaths)>,
        Query<(&mut Text, &ScoreboardAvgKd)>,
    )>,
) {
    let Ok(node) = scoreboard_q.single() else { return };
    if node.display == Display::None { return; }

    use std::collections::HashMap;
    #[derive(Default)]
    struct Agg { score: f32, k: f32, d: f32, count: f32 }
    let mut map: HashMap<u8, Agg> = HashMap::new();
    for (team, score, kills, deaths) in agents.iter() {
        let a = map.entry(team.0).or_default();
        a.score += score.0  as f32;
        a.k     += kills.0  as f32;
        a.d     += deaths.0 as f32;
        a.count += 1.0;
    }

    for (team_id, agg) in &map {
        if agg.count == 0.0 { continue; }
        let avg_sc = agg.score / agg.count;
        let avg_k  = agg.k     / agg.count;
        let avg_d  = agg.d     / agg.count;
        let avg_kd = if avg_d == 0.0 { avg_k } else { avg_k / avg_d };

        for (mut text, marker) in avg.p0().iter_mut() {
            if marker.0 == *team_id { *text = Text::new(format!("{:.0}", avg_sc)); }
        }
        for (mut text, marker) in avg.p1().iter_mut() {
            if marker.0 == *team_id { *text = Text::new(format!("{:.1}", avg_k)); }
        }
        for (mut text, marker) in avg.p2().iter_mut() {
            if marker.0 == *team_id { *text = Text::new(format!("{:.1}", avg_d)); }
        }
        for (mut text, marker) in avg.p3().iter_mut() {
            if marker.0 == *team_id { *text = Text::new(format!("{:.2}", avg_kd)); }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn kd_ratio(kills: u32, deaths: u32) -> String {
    if deaths == 0 { format!("{:.2}", kills as f32) }
    else           { format!("{:.2}", kills as f32 / deaths as f32) }
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

fn viz_button(
    row:         &mut ChildSpawnerCommands,
    is_hidden:   bool,
    dim:         Color,
    on_color:    Color,
    btn_marker:  impl Bundle,
    text_marker: impl Bundle,
    width:       f32,
) {
    let label = if is_hidden { "OFF" } else { "ON" };
    let color = if is_hidden { dim } else { on_color };
    row.spawn((
        Button,
        btn_marker,
        Node {
            width:           Val::Px(width - 8.0),
            height:          Val::Px(24.0),
            justify_content: JustifyContent::Center,
            align_items:     AlignItems::Center,
            border:          UiRect::all(Val::Px(1.0)),
            border_radius:   BorderRadius::all(Val::Px(4.0)),
            margin:          UiRect::right(Val::Px(4.0)),
            ..default()
        },
        BackgroundColor(ThemeColor::ButtonIdle.resolve()),
        BorderColor::all(ThemeColor::Border.resolve()),
    )).with_children(|btn| {
        btn.spawn((
            Text::new(label),
            TextFont  { font_size: F_BTN, ..default() },
            TextColor(color),
            text_marker,
        ));
    });
}

fn data_cell(
    parent: &mut ChildSpawnerCommands,
    text:   &str,
    width:  f32,
    color:  Color,
    size:   f32,
    marker: impl Bundle,
) {
    parent.spawn((
        Text::new(text),
        TextFont  { font_size: size, ..default() },
        TextColor(color),
        Node { width: Val::Px(width), ..default() },
        marker,
    ));
}

fn avg_header_cell(
    parent: &mut ChildSpawnerCommands,
    text:   &str,
    width:  f32,
    color:  Color,
    marker: impl Bundle,
) {
    parent.spawn((
        Text::new(text),
        TextFont  { font_size: F_AVG, ..default() },
        TextColor(color),
        Node { width: Val::Px(width), ..default() },
        marker,
    ));
}

fn hdr(parent: &mut ChildSpawnerCommands, text: &str, width: f32, color: Color, size: f32) {
    parent.spawn((
        Text::new(text),
        TextFont  { font_size: size, ..default() },
        TextColor(color),
        Node { width: Val::Px(width), ..default() },
    ));
}