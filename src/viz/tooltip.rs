// src/viz/tooltip.rs

use bevy::prelude::*;
use crate::world::coords::GridPos;
use crate::agent::components::{Ammo, GoldCarried, Hearts, RespawnIn, Score};
use crate::viz::components::{AgentLabel, HideViz};
use crate::style::{ThemeColor, UiRoot, SIZE_SM};
use super::grid_offset::GridOffset;
use super::camera::MainCamera;

#[derive(Component)] pub struct TooltipPanel;
#[derive(Component)] pub struct TooltipName;
#[derive(Component)] pub struct TooltipHearts;
#[derive(Component)] pub struct TooltipAmmo;
#[derive(Component)] pub struct TooltipCarry;
#[derive(Component)] pub struct TooltipScore;
#[derive(Component)] pub struct TooltipPos;
#[derive(Component)] pub struct TooltipViz;
#[derive(Component)] pub struct TooltipStatus;

pub fn spawn_tooltip(mut commands: Commands) {
    let bg       = ThemeColor::TooltipBackground.resolve();
    let border   = ThemeColor::Border.resolve();
    let head     = ThemeColor::TextDim.resolve();
    let body     = ThemeColor::TextPrimary.resolve();
    let name_col = ThemeColor::SuccessText.resolve();

    commands.spawn((
        UiRoot,
        TooltipPanel,
        Node {
            position_type:  PositionType::Absolute,
            display:        Display::None,
            flex_direction: FlexDirection::Column,
            padding:        UiRect::all(Val::Px(12.0)),
            row_gap:        Val::Px(4.0),
            border:         UiRect::all(Val::Px(1.0)),
            border_radius:  BorderRadius::all(Val::Px(8.0)),
            min_width:      Val::Px(190.0),
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(border),
        ZIndex(150),
    )).with_children(|p| {
        p.spawn((
            Text::new("—"),
            TextFont  { font_size: 13.0, ..default() },
            TextColor(name_col),
            TooltipName,
        ));
        p.spawn((
            Text::new(""),
            TextFont  { font_size: SIZE_SM, ..default() },
            TextColor(head),
            TooltipStatus,
        ));
        tooltip_row(p, "HP:",       "3/3",    head, body, TooltipHearts);
        tooltip_row(p, "Ammo:",     "0",      head, body, TooltipAmmo);
        tooltip_row(p, "Gold:",     "0",      head, body, TooltipCarry);
        tooltip_row(p, "Score:",    "0",      head, body, TooltipScore);
        tooltip_row(p, "Position:", "(0, 0)", head, body, TooltipPos);
        tooltip_row(p, "Debug:",    "ON",     head, body, TooltipViz);
    });
}

fn tooltip_row(
    parent:     &mut ChildSpawnerCommands,
    label:      &str,
    value:      &str,
    head_color: Color,
    body_color: Color,
    marker:     impl Bundle,
) {
    parent.spawn(Node {
        flex_direction: FlexDirection::Row,
        column_gap:     Val::Px(6.0),
        ..default()
    }).with_children(|r| {
        r.spawn((Text::new(label), TextFont { font_size: SIZE_SM, ..default() }, TextColor(head_color)));
        r.spawn((Text::new(value), TextFont { font_size: SIZE_SM, ..default() }, TextColor(body_color), marker));
    });
}

pub fn update_tooltip(
    windows:      Query<&Window>,
    mouse:        Res<ButtonInput<MouseButton>>,
    mut commands: Commands,
    agents:       Query<(
        Entity, &AgentLabel, &Hearts, &Ammo,
        &GoldCarried, &Score, &GridPos, &Transform,
        Has<HideViz>, Option<&RespawnIn>,
    )>,
    offset:       Res<GridOffset>,
    cam_q:        Query<(&Transform, &Projection), With<MainCamera>>,
    mut panel_q:  Query<(&mut Node, &mut Visibility), With<TooltipPanel>>,
    mut name_q:   Query<&mut Text, (With<TooltipName>,   Without<TooltipHearts>, Without<TooltipAmmo>, Without<TooltipCarry>, Without<TooltipScore>, Without<TooltipPos>, Without<TooltipViz>, Without<TooltipStatus>)>,
    mut status_q: Query<&mut Text, (With<TooltipStatus>, Without<TooltipName>,   Without<TooltipHearts>, Without<TooltipAmmo>, Without<TooltipCarry>, Without<TooltipScore>, Without<TooltipPos>, Without<TooltipViz>)>,
    mut hearts_q: Query<&mut Text, (With<TooltipHearts>, Without<TooltipName>,   Without<TooltipStatus>, Without<TooltipAmmo>,  Without<TooltipCarry>, Without<TooltipScore>, Without<TooltipPos>, Without<TooltipViz>)>,
    mut ammo_q:   Query<&mut Text, (With<TooltipAmmo>,   Without<TooltipName>,   Without<TooltipStatus>, Without<TooltipHearts>,Without<TooltipCarry>, Without<TooltipScore>, Without<TooltipPos>, Without<TooltipViz>)>,
    mut carry_q:  Query<&mut Text, (With<TooltipCarry>,  Without<TooltipName>,   Without<TooltipStatus>, Without<TooltipHearts>,Without<TooltipAmmo>,  Without<TooltipScore>, Without<TooltipPos>, Without<TooltipViz>)>,
    mut score_q:  Query<&mut Text, (With<TooltipScore>,  Without<TooltipName>,   Without<TooltipStatus>, Without<TooltipHearts>,Without<TooltipAmmo>,  Without<TooltipCarry>, Without<TooltipPos>, Without<TooltipViz>)>,
    mut pos_q:    Query<&mut Text, (With<TooltipPos>,    Without<TooltipName>,   Without<TooltipStatus>, Without<TooltipHearts>,Without<TooltipAmmo>,  Without<TooltipCarry>, Without<TooltipScore>,Without<TooltipViz>)>,
    mut viz_q:    Query<&mut Text, (With<TooltipViz>,    Without<TooltipName>,   Without<TooltipStatus>, Without<TooltipHearts>,Without<TooltipAmmo>,  Without<TooltipCarry>, Without<TooltipScore>,Without<TooltipPos>)>,
) {
    let Ok(window)              = windows.single()     else { return };
    let Ok((mut node, mut vis)) = panel_q.single_mut() else { return };

    let Some(cursor_screen) = window.cursor_position() else {
        hide(&mut node, &mut vis); return;
    };

    let cursor_world = {
        let Ok((cam_tf, projection)) = cam_q.single() else {
            hide(&mut node, &mut vis); return;
        };
        let Projection::Orthographic(ref ortho) = *projection else {
            hide(&mut node, &mut vis); return;
        };
        let win = Vec2::new(window.width(), window.height());
        let ndc = (cursor_screen / win - 0.5) * 2.0;
        cam_tf.translation.truncate()
            + Vec2::new(ndc.x * win.x / 2.0 * ortho.scale,
                        -ndc.y * win.y / 2.0 * ortho.scale)
    };

    let hover_r = offset.step * 0.6;
    let hovered = agents.iter().find(|(_, _, _, _, _, _, _, t, _, _)| {
        cursor_world.distance(t.translation.truncate()) < hover_r
    });

    if let Some((entity, label, hearts, ammo, gold, score, pos, _, is_hidden, respawn)) = hovered {
        node.display = Display::Flex;
        *vis         = Visibility::Visible;
        node.left    = Val::Px((cursor_screen.x + 14.0).min(window.width() - 210.0));
        node.top     = Val::Px((cursor_screen.y - 10.0).max(0.0));

        let hp_str = format!("{}/{}", hearts.0, crate::config::AGENT_MAX_HEARTS);

        if let Ok(mut t) = name_q.single_mut()   { *t = Text::new(&label.0); }
        if let Ok(mut t) = status_q.single_mut() {
            *t = Text::new(if respawn.is_some() { "RESPAWNING" } else { "" });
        }
        if let Ok(mut t) = hearts_q.single_mut() { *t = Text::new(hp_str); }
        if let Ok(mut t) = ammo_q.single_mut()   { *t = Text::new(ammo.0.to_string()); }
        if let Ok(mut t) = carry_q.single_mut()  { *t = Text::new(gold.0.to_string()); }
        if let Ok(mut t) = score_q.single_mut()  { *t = Text::new(score.0.to_string()); }
        if let Ok(mut t) = pos_q.single_mut()    {
            *t = Text::new(format!("({}, {})", pos.x, pos.y));
        }
        if let Ok(mut t) = viz_q.single_mut()    {
            *t = Text::new(if is_hidden { "OFF (click)" } else { "ON (click)" });
        }

        if mouse.just_pressed(MouseButton::Left) {
            if is_hidden { commands.entity(entity).remove::<HideViz>(); }
            else         { commands.entity(entity).insert(HideViz); }
        }
    } else {
        hide(&mut node, &mut vis);
    }
}

fn hide(node: &mut Node, vis: &mut Visibility) {
    node.display = Display::None;
    *vis         = Visibility::Hidden;
}