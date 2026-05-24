use bevy::prelude::*;
use bevy::color::Alpha;
use atb::sim_core::AgentState;
use crate::sim_bridge::SimBridge;
use crate::style::{ThemeColor, UiRoot, SIZE_SM, TOOLBAR_H};
use crate::style::color::{team_color, SURFACE_HIGHLIGHT};
use crate::team::Team;
use super::components::{
    TabScoreboard, TabScoreboardContent, ScoreboardRow,
    ScoreboardTeamScore,
    ScoreboardRowHp, ScoreboardRowAmmo, ScoreboardRowGold, ScoreboardRowScore,
};

const F_AGENT: f32 = 14.0;
const F_STAT:  f32 = 14.0;
const F_AVG:   f32 = 15.0;

const W_AGENT: f32 = 160.0;
const W_HP:    f32 =  56.0;
const W_AMMO:  f32 =  56.0;
const W_GOLD:  f32 =  56.0;
const W_SCORE: f32 =  72.0;
const TOTAL_W: f32 = 16.0 + W_AGENT + W_HP + W_AMMO + W_GOLD + W_SCORE + 16.0;

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
        // Column header row
        panel.spawn((
            Node {
                flex_direction: FlexDirection::Row,
                align_items:    AlignItems::Center,
                padding:        UiRect::new(Val::Px(16.0), Val::Px(16.0), Val::Px(8.0), Val::Px(6.0)),
                border:         UiRect::bottom(Val::Px(1.0)),
                ..default()
            },
            BorderColor::all(border),
        )).with_children(|hdr| {
            col_label(hdr, "Agent",  W_AGENT, dim);
            col_label(hdr, "HP",     W_HP,    dim);
            col_label(hdr, "Ammo",   W_AMMO,  dim);
            col_label(hdr, "Gold",   W_GOLD,  dim);
            col_label(hdr, "Score",  W_SCORE, dim);
        });

        // Content container — rows are built here by build_scoreboard_rows.
        panel.spawn((Node { flex_direction: FlexDirection::Column, ..default() }, TabScoreboardContent));
    });
}

fn col_label(parent: &mut ChildSpawnerCommands, text: &str, width: f32, color: Color) {
    parent.spawn((
        Text::new(text),
        TextFont { font_size: SIZE_SM, ..default() },
        TextColor(color),
        Node { min_width: Val::Px(width), ..default() },
    ));
}

fn stat_cell<M: Component>(parent: &mut ChildSpawnerCommands, text: &str, width: f32, color: Color, marker: M) {
    parent.spawn((
        Text::new(text),
        TextFont { font_size: F_STAT, ..default() },
        TextColor(color),
        Node { min_width: Val::Px(width), ..default() },
        marker,
    ));
}

pub fn build_scoreboard_rows(
    bridge:   Res<SimBridge>,
    cont: Query<Entity, With<TabScoreboardContent>>,
    existing: Query<Entity, With<ScoreboardRow>>,
    mut commands: Commands,
) {
    // Only build once (agent count is fixed).
    if !existing.is_empty() { return; }
    let Ok(container) = cont.single() else { return };

    let agents = bridge.agents();
    let body   = ThemeColor::TextPrimary.resolve();
    let border = ThemeColor::Border.resolve();

    // Group by team.
    let n_teams = bridge.n_teams();
    for team_id in 0..n_teams {
        let team_agents: Vec<(usize, &AgentState)> = agents
            .iter()
            .enumerate()
            .filter(|(_, a)| a.team == team_id)
            .collect();
        if team_agents.is_empty() { continue; }

        let team_color_val = team_color(team_id);

        // Team header row.
        commands.entity(container).with_children(|cont| {
            cont.spawn((
                Node {
                    flex_direction:  FlexDirection::Row,
                    align_items:     AlignItems::Center,
                    justify_content: JustifyContent::SpaceBetween,
                    padding:         UiRect::new(Val::Px(16.0), Val::Px(16.0), Val::Px(6.0), Val::Px(6.0)),
                    border:          UiRect::bottom(Val::Px(1.0)),
                    ..default()
                },
                BackgroundColor(SURFACE_HIGHLIGHT),
                BorderColor::all(border),
            )).with_children(|hdr| {
                hdr.spawn((
                    Text::new(Team(team_id).name()),
                    TextFont { font_size: F_AVG, ..default() },
                    TextColor(team_color_val),
                ));
                hdr.spawn((
                    Text::new("0"),
                    TextFont { font_size: F_AVG, ..default() },
                    TextColor(team_color_val),
                    ScoreboardTeamScore(team_id),
                ));
            });

            // Agent rows.
            for (idx, agent) in &team_agents {
                let agent_idx = *idx;
                cont.spawn((
                    ScoreboardRow,
                    Node {
                        flex_direction: FlexDirection::Row,
                        align_items:    AlignItems::Center,
                        padding:        UiRect::new(Val::Px(16.0), Val::Px(16.0), Val::Px(5.0), Val::Px(5.0)),
                        border:         UiRect::bottom(Val::Px(1.0)),
                        ..default()
                    },
                    BorderColor::all(border.with_alpha(0.4)),
                )).with_children(|row| {
                    // Agent name
                    row.spawn((
                        Text::new(format!("Agent {}", agent_idx)),
                        TextFont { font_size: F_AGENT, ..default() },
                        TextColor(team_color_val),
                        Node { min_width: Val::Px(W_AGENT), ..default() },
                    ));
                    stat_cell(row, &format!("{}", agent.hearts), W_HP,    body, ScoreboardRowHp(agent_idx));
                    stat_cell(row, &format!("{}", agent.ammo),   W_AMMO,  body, ScoreboardRowAmmo(agent_idx));
                    stat_cell(row, &format!("{}", agent.gold_carried), W_GOLD, body, ScoreboardRowGold(agent_idx));
                    stat_cell(row, &format!("{}", agent.score),  W_SCORE, body, ScoreboardRowScore(agent_idx));
                });
            }
        });
    }
}

pub fn refresh_scoreboard_stats(
    bridge: Res<SimBridge>,
    mut qs: ParamSet<(
        Query<(&mut Text, &ScoreboardRowHp)>,
        Query<(&mut Text, &ScoreboardRowAmmo)>,
        Query<(&mut Text, &ScoreboardRowGold)>,
        Query<(&mut Text, &ScoreboardRowScore)>,
        Query<(&mut Text, &ScoreboardTeamScore)>,
    )>,
) {
    if !bridge.is_changed() { return; }
    let agents = bridge.agents();
    for (mut text, marker) in qs.p0().iter_mut() {
        if let Some(a) = agents.get(marker.0) { *text = Text::new(format!("{}", a.hearts)); }
    }
    for (mut text, marker) in qs.p1().iter_mut() {
        if let Some(a) = agents.get(marker.0) { *text = Text::new(format!("{}", a.ammo)); }
    }
    for (mut text, marker) in qs.p2().iter_mut() {
        if let Some(a) = agents.get(marker.0) { *text = Text::new(format!("{}", a.gold_carried)); }
    }
    for (mut text, marker) in qs.p3().iter_mut() {
        if let Some(a) = agents.get(marker.0) { *text = Text::new(format!("{}", a.score)); }
    }
    for (mut text, marker) in qs.p4().iter_mut() {
        *text = Text::new(bridge.team_score(marker.0).to_string());
    }
}

pub fn toggle_tab_scoreboard(
    keys:      Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut Node, With<TabScoreboard>>,
) {
    let show = keys.pressed(KeyCode::Tab);
    for mut node in query.iter_mut() {
        node.display = if show { Display::Flex } else { Display::None };
    }
}
