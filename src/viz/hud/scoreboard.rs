// src/viz/hud/scoreboard.rs

use bevy::prelude::*;
use crate::agent::components::{AgentInfo, AgentLabel, Ammo, GoldCarried, Hearts, RespawnIn, Score};
use crate::style::{ThemeColor, UiRoot, SIZE_SM, SIZE_MD, SIZE_LG, TOOLBAR_H};
use crate::style::color::team_color;
use crate::team::Team;
use super::components::{TabScoreboard, TabScoreboardContent, HideViz};

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
            margin:         UiRect::left(Val::Px(-300.0)),
            flex_direction: FlexDirection::Column,
            min_width:      Val::Px(620.0),
            border:         UiRect::all(Val::Px(1.0)),
            border_radius:  BorderRadius::all(Val::Px(10.0)),
            overflow:       Overflow::clip(),
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(border),
        ZIndex(200),
    )).with_children(|panel| {
        panel.spawn(Node {
            flex_direction: FlexDirection::Row,
            padding:        UiRect::new(Val::Px(16.0), Val::Px(16.0), Val::Px(10.0), Val::Px(8.0)),
            border:         UiRect::bottom(Val::Px(1.0)),
            ..default()
        }).with_children(|h| {
            hdr(h, "TEAM",  50.0,  dim);
            hdr(h, "AGENT", 220.0, dim);
            hdr(h, "HP",    60.0,  dim);
            hdr(h, "AMMO",  55.0,  dim);
            hdr(h, "GOLD",  55.0,  ThemeColor::AccentGold.resolve());
            hdr(h, "SCORE", 70.0,  dim);
            hdr(h, "VIZ",   60.0,  dim);
        });

        panel.spawn((
            TabScoreboardContent,
            Node { flex_direction: FlexDirection::Column, ..default() },
        ));

        panel.spawn(Node {
            flex_direction:  FlexDirection::Row,
            justify_content: JustifyContent::Center,
            padding:         UiRect::axes(Val::Px(0.0), Val::Px(6.0)),
            border:          UiRect::top(Val::Px(1.0)),
            ..default()
        }).with_children(|f| {
            f.spawn((
                Text::new("Hold TAB  |  Click VIZ to toggle path overlay  |  H for shortcuts"),
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

// ── Viz toggle ────────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct VizToggleButton(pub Entity);

pub fn handle_viz_toggle(
    mut commands: Commands,
    buttons:      Query<(&Interaction, &VizToggleButton), Changed<Interaction>>,
    hidden:       Query<Has<HideViz>>,
) {
    for (interaction, btn) in buttons.iter() {
        if *interaction != Interaction::Pressed { continue; }
        let entity    = btn.0;
        let is_hidden = hidden.get(entity).unwrap_or(false);
        if is_hidden { commands.entity(entity).remove::<HideViz>(); }
        else         { commands.entity(entity).insert(HideViz); }
    }
}

// ── Update content ────────────────────────────────────────────────────────────

pub fn update_tab_scoreboard(
    scoreboard_q: Query<&Node, With<TabScoreboard>>,
    content_q:    Query<Entity, With<TabScoreboardContent>>,
    agents:       Query<(
        Entity, &AgentLabel, &AgentInfo, &Team, &Hearts, &Ammo,
        &GoldCarried, &Score, Option<&RespawnIn>, Has<HideViz>,
    )>,
    mut commands: Commands,
) {
    let Ok(node) = scoreboard_q.single() else { return };
    if node.display == Display::None { return; }

    let Ok(content) = content_q.single() else { return };
    commands.entity(content).despawn_related::<Children>();

    let mut rows: Vec<_> = agents.iter().collect();
    rows.sort_by(|a, b| a.3.0.cmp(&b.3.0).then(b.7.0.cmp(&a.7.0)));

    let dim     = ThemeColor::TextDim.resolve();
    let primary = ThemeColor::TextPrimary.resolve();
    let gold_c  = ThemeColor::AccentGold.resolve();
    let score_c = ThemeColor::SuccessText.resolve();

    commands.entity(content).with_children(|c| {
        for (i, (agent_entity, label, info, team, hearts, ammo, gold, score, respawning, is_hidden))
        in rows.iter().enumerate()
        {
            let bg         = if i % 2 == 0 { Color::NONE }
            else { ThemeColor::SurfaceHighlight.resolve() };
            let tcolor     = team_color(team.0);
            let name_color = if respawning.is_some() { dim } else { primary };

            let hp_str   = format!("{}/{}", hearts.0, crate::config::AGENT_MAX_HEARTS);
            let hp_color = match hearts.0 {
                0 => Color::srgb(0.6, 0.6, 0.6),
                1 => Color::srgb(0.85, 0.25, 0.20),
                2 => Color::srgb(0.95, 0.60, 0.10),
                _ => Color::srgb(0.20, 0.75, 0.35),
            };

            let viz_label = if *is_hidden { "OFF" } else { "ON" };
            let viz_color = if *is_hidden { dim } else { score_c };
            let info_text = format!("{} · {}", info.strategy, info.planner);

            c.spawn((
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

                // Name + strategy/planner subtext stacked vertically
                row.spawn(Node {
                    flex_direction: FlexDirection::Column,
                    width:          Val::Px(220.0),
                    row_gap:        Val::Px(2.0),
                    ..default()
                }).with_children(|name_col| {
                    name_col.spawn((
                        Text::new(&label.0),
                        TextFont  { font_size: SIZE_MD, ..default() },
                        TextColor(name_color),
                    ));
                    name_col.spawn((
                        Text::new(&info_text),
                        TextFont  { font_size: 9.0, ..default() },
                        TextColor(dim),
                    ));
                });

                // HP
                let hp_display = if respawning.is_some() { "dead".to_string() } else { hp_str };
                cell(row, &hp_display, 60.0, hp_color, SIZE_MD);

                // Ammo
                cell(row, &ammo.0.to_string(), 55.0, primary, SIZE_MD);

                // Gold
                cell(row, &gold.0.to_string(), 55.0, gold_c, SIZE_MD);

                // Score
                cell(row, &score.0.to_string(), 70.0, score_c, SIZE_LG);

                // Viz toggle
                row.spawn((
                    Button,
                    VizToggleButton(*agent_entity),
                    Node {
                        width:           Val::Px(44.0),
                        height:          Val::Px(22.0),
                        justify_content: JustifyContent::Center,
                        align_items:     AlignItems::Center,
                        border:          UiRect::all(Val::Px(1.0)),
                        border_radius:   BorderRadius::all(Val::Px(4.0)),
                        ..default()
                    },
                    BackgroundColor(ThemeColor::ButtonIdle.resolve()),
                    BorderColor::all(ThemeColor::Border.resolve()),
                )).with_children(|btn| {
                    btn.spawn((
                        Text::new(viz_label),
                        TextFont  { font_size: SIZE_SM, ..default() },
                        TextColor(viz_color),
                    ));
                });
            });
        }
    });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn hdr(parent: &mut ChildSpawnerCommands, text: &str, width: f32, color: Color) {
    parent.spawn((
        Text::new(text),
        TextFont  { font_size: SIZE_SM, ..default() },
        TextColor(color),
        Node { width: Val::Px(width), ..default() },
    ));
}

fn cell(parent: &mut ChildSpawnerCommands, text: &str, width: f32, color: Color, size: f32) {
    parent.spawn((
        Text::new(text),
        TextFont  { font_size: size, ..default() },
        TextColor(color),
        Node { width: Val::Px(width), ..default() },
    ));
}