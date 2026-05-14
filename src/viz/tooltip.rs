// src/viz/tooltip.rs

use bevy::prelude::*;
use crate::world::coords::GridPos;
use crate::agent::components::{AgentLabel, GoldCarried, Score};
use crate::viz::menu::components::HideViz;
use crate::style::{ThemeMode, ThemeColor, UiRoot, SIZE_SM};
use super::grid_offset::GridOffset;
use super::camera::MainCamera;

#[derive(Component)] pub struct TooltipPanel;
#[derive(Component)] pub struct TooltipName;
#[derive(Component)] pub struct TooltipCarry;
#[derive(Component)] pub struct TooltipScore;
#[derive(Component)] pub struct TooltipPos;
#[derive(Component)] pub struct TooltipViz;

pub fn spawn_tooltip(mut commands: Commands, theme: Res<ThemeMode>) {
    build_tooltip(&mut commands, *theme);
}

pub fn build_tooltip(commands: &mut Commands, mode: ThemeMode) {
    let bg           = ThemeColor::TooltipBackground.resolve(mode);
    let border       = ThemeColor::Border.resolve(mode);
    let text_head    = ThemeColor::TextDim.resolve(mode);
    let text_body    = ThemeColor::TextPrimary.resolve(mode);
    let accent_green = ThemeColor::SuccessText.resolve(mode);

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
            min_width:      Val::Px(180.0),
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(border),
        ZIndex(100),
    )).with_children(|p| {
        p.spawn((
            Text::new("—"),
            TextFont  { font_size: 13.0, ..default() },
            TextColor(accent_green),
            TooltipName,
        ));
        tooltip_row(p, "Gold:",      "0",      text_head, text_body, TooltipCarry);
        tooltip_row(p, "Score:",     "0",      text_head, text_body, TooltipScore);
        tooltip_row(p, "Position:",  "(0, 0)", text_head, text_body, TooltipPos);
        tooltip_row(p, "Debug Viz:", "ON",     text_head, text_body, TooltipViz);
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
    windows:       Query<&Window>,
    mouse:         Res<ButtonInput<MouseButton>>,
    mut commands:  Commands,
    agents:        Query<(Entity, &AgentLabel, &GoldCarried, &Score, &GridPos, &Transform, Has<HideViz>)>,
    offset:        Res<GridOffset>,
    cam_q:         Query<(&Transform, &Projection), With<MainCamera>>,
    mut panel_q:   Query<(&mut Node, &mut Visibility), With<TooltipPanel>>,
    mut name_q:    Query<&mut Text, (With<TooltipName>,  Without<TooltipCarry>, Without<TooltipScore>, Without<TooltipPos>, Without<TooltipViz>)>,
    mut carry_q:   Query<&mut Text, (With<TooltipCarry>, Without<TooltipName>,  Without<TooltipScore>, Without<TooltipPos>, Without<TooltipViz>)>,
    mut score_q:   Query<&mut Text, (With<TooltipScore>, Without<TooltipName>,  Without<TooltipCarry>, Without<TooltipPos>, Without<TooltipViz>)>,
    mut pos_q:     Query<&mut Text, (With<TooltipPos>,   Without<TooltipName>,  Without<TooltipCarry>, Without<TooltipScore>, Without<TooltipViz>)>,
    mut viz_q:     Query<&mut Text, (With<TooltipViz>,   Without<TooltipName>,  Without<TooltipCarry>, Without<TooltipScore>, Without<TooltipPos>)>,
) {
    let Ok(window)              = windows.single()     else { return };
    let Ok((mut node, mut vis)) = panel_q.single_mut() else { return };

    let Some(cursor_screen) = window.cursor_position() else {
        hide(&mut node, &mut vis);
        return;
    };

    let cursor_world = {
        let Ok((cam_tf, projection)) = cam_q.single() else {
            hide(&mut node, &mut vis);
            return;
        };
        let Projection::Orthographic(ref ortho) = *projection else {
            hide(&mut node, &mut vis);
            return;
        };
        let win   = Vec2::new(window.width(), window.height());
        let ndc   = (cursor_screen / win - 0.5) * 2.0;
        let scale = ortho.scale;
        cam_tf.translation.truncate()
            + Vec2::new(ndc.x * win.x / 2.0 * scale, -ndc.y * win.y / 2.0 * scale)
    };

    let hover_r = offset.step * 0.6;
    let hovered = agents.iter().find(|(_, _, _, _, _, t, _)| {
        cursor_world.distance(t.translation.truncate()) < hover_r
    });

    if let Some((entity, label, gold, score, pos, _, is_hidden)) = hovered {
        node.display = Display::Flex;
        *vis         = Visibility::Visible;
        node.left    = Val::Px((cursor_screen.x + 14.0).min(window.width() - 200.0));
        node.top     = Val::Px((cursor_screen.y - 10.0).max(0.0));

        if let Ok(mut t) = name_q.single_mut()  { *t = Text::new(&label.0); }
        if let Ok(mut t) = carry_q.single_mut() { *t = Text::new(gold.0.to_string()); }
        if let Ok(mut t) = score_q.single_mut() { *t = Text::new(score.0.to_string()); }
        if let Ok(mut t) = pos_q.single_mut()   { *t = Text::new(format!("({}, {})", pos.x, pos.y)); }
        if let Ok(mut t) = viz_q.single_mut()   {
            *t = Text::new(if is_hidden { "OFF (Click to toggle)" } else { "ON (Click to toggle)" });
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