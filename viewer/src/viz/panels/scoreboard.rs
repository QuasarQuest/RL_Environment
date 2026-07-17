use bevy::prelude::*;
use bevy::color::Alpha;
use atb::engine::AgentState;
use crate::sim_bridge::SimBridge;
use crate::style::{ThemeColor, UiRoot, SIZE_SM, SIZE_XL, TOOLBAR_H};
use crate::style::color::{team_color, SURFACE_HIGHLIGHT, GREEN_400, VIOLET_400, ROW_STRIPE};
use crate::team::Team;
use super::components::{
    TabScoreboard, TabScoreboardContent, ScoreboardRow,
    ScoreboardTeamScore,
    ScoreboardRowSpeed, ScoreboardRowMult, ScoreboardRowGold, ScoreboardRowScore,
};

const F_AGENT: f32 = 15.0;
const F_STAT:  f32 = 15.0;
const F_SCORE: f32 = 18.0;
const F_AVG:   f32 = 16.0;

const W_AGENT: f32 = 180.0;
const W_SPEED: f32 =  60.0;
const W_MULT:  f32 =  60.0;
const W_GOLD:  f32 =  56.0;
const W_SCORE: f32 =  72.0;
const TOTAL_W: f32 = 16.0 + W_AGENT + W_SPEED + W_MULT + W_GOLD + W_SCORE + 16.0;

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
        // Column header row. Buff headers are tinted with their buff colour so the
        // columns double as a legend (speed=green, mult=violet).
        panel.spawn((
            Node {
                flex_direction: FlexDirection::Row,
                align_items:    AlignItems::Center,
                padding:        UiRect::new(Val::Px(16.0), Val::Px(16.0), Val::Px(10.0), Val::Px(8.0)),
                border:         UiRect::bottom(Val::Px(1.0)),
                ..default()
            },
            BackgroundColor(ROW_STRIPE),
            BorderColor::all(border),
        )).with_children(|hdr| {
            let gold_hdr = ThemeColor::AccentGold.resolve();
            hdr_cell(hdr, "AGENT", W_AGENT, dim);
            hdr_cell(hdr, "SPEED", W_SPEED, GREEN_400);
            hdr_cell(hdr, "MULT",  W_MULT,  VIOLET_400);
            hdr_cell(hdr, "GOLD",  W_GOLD,  gold_hdr);
            hdr_cell(hdr, "SCORE", W_SCORE, dim);
        });

        // Content container — rows are built by build_scoreboard_rows.
        panel.spawn((Node { flex_direction: FlexDirection::Column, ..default() }, TabScoreboardContent));

        // Footer
        panel.spawn((
            Node {
                flex_direction:  FlexDirection::Row,
                justify_content: JustifyContent::Center,
                padding:         UiRect::axes(Val::Px(0.0), Val::Px(6.0)),
                border:          UiRect::top(Val::Px(1.0)),
                ..default()
            },
            BackgroundColor(ROW_STRIPE),
            BorderColor::all(border),
        )).with_children(|f| {
            f.spawn((
                Text::new("Hold TAB · SPEED = ticks left · MULT = charge held"),
                TextFont  { font_size: SIZE_SM, ..default() },
                TextColor(dim),
            ));
        });
    });
}

fn hdr_cell(parent: &mut ChildSpawnerCommands, text: &str, width: f32, color: Color) {
    parent.spawn((
        Text::new(text),
        TextFont  { font_size: SIZE_XL, ..default() },
        TextColor(color),
        Node { width: Val::Px(width), ..default() },
    ));
}

/// Buff cell text: ticks remaining, or "–" when the buff is inactive.
fn buff_text(ticks: u16) -> String {
    if ticks > 0 { ticks.to_string() } else { "–".to_string() }
}

/// Buff cell colour: the buff's colour when active, dim when inactive.
fn buff_color(ticks: u16, active: Color, dim: Color) -> Color {
    if ticks > 0 { active } else { dim }
}

/// Multiplier cell: "×2" while a charge is held (to spend on the next deposit), else "–".
fn charge_text(charge: u8) -> String {
    if charge > 0 { "×2".to_string() } else { "–".to_string() }
}

fn stat_cell<M: Component>(parent: &mut ChildSpawnerCommands, text: &str, width: f32, color: Color, size: f32, marker: M) {
    parent.spawn((
        Text::new(text),
        TextFont  { font_size: size, ..default() },
        TextColor(color),
        Node { width: Val::Px(width), ..default() },
        marker,
    ));
}

pub fn build_scoreboard_rows(
    bridge:   Res<SimBridge>,
    cont:     Query<Entity, With<TabScoreboardContent>>,
    existing: Query<Entity, With<ScoreboardRow>>,
    mut commands: Commands,
) {
    if !existing.is_empty() { return; }
    let Ok(container) = cont.single() else { return };

    let agents  = bridge.agents();
    let primary = ThemeColor::TextPrimary.resolve();
    let dim     = ThemeColor::TextDim.resolve();
    let score_c = ThemeColor::SuccessText.resolve();
    let gold_c  = ThemeColor::AccentGold.resolve();
    let n_teams = bridge.n_teams();

    for team_id in 0..n_teams {
        let team_agents: Vec<(usize, &AgentState)> = agents
            .iter()
            .enumerate()
            .filter(|(_, a)| a.team == team_id)
            .collect();
        if team_agents.is_empty() { continue; }

        let tc = team_color(team_id);

        commands.entity(container).with_children(|cont| {
            // Team header
            cont.spawn((
                Node {
                    flex_direction:  FlexDirection::Row,
                    align_items:     AlignItems::Center,
                    justify_content: JustifyContent::SpaceBetween,
                    padding:         UiRect::new(Val::Px(16.0), Val::Px(16.0), Val::Px(8.0), Val::Px(8.0)),
                    border:          UiRect::bottom(Val::Px(1.0)),
                    ..default()
                },
                BackgroundColor(tc.with_alpha(0.10)),
                BorderColor::all(tc.with_alpha(0.25)),
            )).with_children(|h| {
                h.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    align_items:    AlignItems::Center,
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
                        BackgroundColor(tc),
                    ));
                    left.spawn((
                        Text::new(Team(team_id).name().to_uppercase()),
                        TextFont  { font_size: F_AVG, ..default() },
                        TextColor(tc),
                    ));
                });
                h.spawn((
                    Text::new(bridge.team_score(team_id).to_string()),
                    TextFont  { font_size: F_SCORE, ..default() },
                    TextColor(score_c),
                    ScoreboardTeamScore(team_id),
                ));
            });

            // Agent rows
            for (i, (agent_idx, agent)) in team_agents.iter().enumerate() {
                let agent_idx = *agent_idx;
                let bg = if i % 2 == 0 { SURFACE_HIGHLIGHT } else { Color::NONE };
                let gold_carry_color = if agent.gold_carried > 0 { gold_c } else { primary };

                cont.spawn((
                    ScoreboardRow,
                    Node {
                        flex_direction: FlexDirection::Row,
                        align_items:    AlignItems::Center,
                        padding:        UiRect::axes(Val::Px(16.0), Val::Px(9.0)),
                        ..default()
                    },
                    BackgroundColor(bg),
                )).with_children(|row| {
                    row.spawn((
                        Text::new(format!("Agent {}", agent_idx)),
                        TextFont  { font_size: F_AGENT, ..default() },
                        TextColor(tc),
                        Node { width: Val::Px(W_AGENT), ..default() },
                    ));
                    stat_cell(row, &buff_text(agent.speed_buff), W_SPEED, buff_color(agent.speed_buff, GREEN_400,  dim), F_STAT,  ScoreboardRowSpeed(agent_idx));
                    stat_cell(row, &charge_text(agent.mult_charge), W_MULT, if agent.mult_charge > 0 { VIOLET_400 } else { dim }, F_STAT, ScoreboardRowMult(agent_idx));
                    stat_cell(row, &agent.gold_carried.to_string(), W_GOLD,  gold_carry_color,       F_STAT,  ScoreboardRowGold(agent_idx));
                    stat_cell(row, &agent.score.to_string(),        W_SCORE, score_c,                F_SCORE, ScoreboardRowScore(agent_idx));
                });
            }

            // Team divider
            cont.spawn((
                Node { height: Val::Px(2.0), ..default() },
                BackgroundColor(tc.with_alpha(0.25)),
            ));
        });
    }
}

/// Rebuild the scoreboard after a config load — the new config's agent/team
/// composition may differ, and stale rows would silently keep rendering (their
/// per-agent lookups just miss). Clearing the content container despawns rows,
/// headers and dividers; build_scoreboard_rows refills it next frame.
pub fn clear_scoreboard_rows(
    mut ev:       MessageReader<crate::viz::events::ConfigLoaded>,
    cont:         Query<Entity, With<TabScoreboardContent>>,
    mut commands: Commands,
) {
    if ev.read().next().is_none() { return; }
    for e in cont.iter() { commands.entity(e).despawn_related::<Children>(); }
}

pub fn refresh_scoreboard_stats(
    bridge:          Res<SimBridge>,
    panel:           Query<&Node, With<TabScoreboard>>,
    mut was_visible: Local<bool>,
    mut qs: ParamSet<(
        Query<(&mut Text, &mut TextColor, &ScoreboardRowSpeed)>,
        Query<(&mut Text, &mut TextColor, &ScoreboardRowMult)>,
        Query<(&mut Text, &mut TextColor, &ScoreboardRowGold)>,
        Query<(&mut Text, &ScoreboardRowScore)>,
        Query<(&mut Text, &ScoreboardTeamScore)>,
    )>,
) {
    // Skip the text rebuild while hidden; refresh once on becoming visible so the
    // numbers are current even if the sim is paused.
    let visible = panel.iter().any(|n| n.display != Display::None);
    let just_shown = visible && !*was_visible;
    *was_visible = visible;
    if !visible || (!bridge.is_changed() && !just_shown) { return; }
    let agents  = bridge.agents();
    let gold_c  = ThemeColor::AccentGold.resolve();
    let primary = ThemeColor::TextPrimary.resolve();
    let dim     = ThemeColor::TextDim.resolve();

    for (mut text, mut color, marker) in qs.p0().iter_mut() {
        if let Some(a) = agents.get(marker.0) {
            *text  = Text::new(buff_text(a.speed_buff));
            *color = TextColor(buff_color(a.speed_buff, GREEN_400, dim));
        }
    }
    for (mut text, mut color, marker) in qs.p1().iter_mut() {
        if let Some(a) = agents.get(marker.0) {
            *text  = Text::new(charge_text(a.mult_charge));
            *color = TextColor(if a.mult_charge > 0 { VIOLET_400 } else { dim });
        }
    }
    for (mut text, mut color, marker) in qs.p2().iter_mut() {
        if let Some(a) = agents.get(marker.0) {
            *text  = Text::new(a.gold_carried.to_string());
            *color = TextColor(if a.gold_carried > 0 { gold_c } else { primary });
        }
    }
    for (mut text, marker) in qs.p3().iter_mut() {
        if let Some(a) = agents.get(marker.0) { *text = Text::new(a.score.to_string()); }
    }
    for (mut text, marker) in qs.p4().iter_mut() {
        *text = Text::new(bridge.team_score(marker.0).to_string());
    }
}

pub fn toggle_tab_scoreboard(
    keys:      Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut Node, With<TabScoreboard>>,
) {
    let want = if keys.pressed(KeyCode::Tab) { Display::Flex } else { Display::None };
    for mut node in query.iter_mut() {
        // Write only on transition — an unconditional Node write dirties UI
        // layout (taffy relayout) every frame.
        if node.display != want { node.display = want; }
    }
}
