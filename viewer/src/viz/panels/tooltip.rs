use bevy::prelude::*;
use crate::sim_bridge::{SimBridge, AgentIndex, AgentMarker};
use crate::style::{ThemeColor, UiRoot, SIZE_SM};
use crate::viz::camera::MainCamera;
use atb::config;

/// Fraction of TILE_SIZE used as the hover hit radius.
const HIT_RADIUS_FACTOR: f32 = 0.6;
/// Pixel offset of the tooltip panel relative to the cursor position.
const TOOLTIP_OFFSET_X: f32 = 14.0;
const TOOLTIP_OFFSET_Y: f32 = -10.0;

#[derive(Component)] pub struct TooltipPanel;
#[derive(Component)] pub struct TooltipName;
#[derive(Component)] pub struct TooltipStats;
#[derive(Component)] pub struct TooltipExtra;

pub fn spawn_tooltip(mut commands: Commands) {
    let bg     = ThemeColor::TooltipBackground.resolve();
    let border = ThemeColor::Border.resolve();
    let dim    = ThemeColor::TextDim.resolve();

    commands.spawn((
        UiRoot,
        TooltipPanel,
        Node {
            position_type:  PositionType::Absolute,
            display:        Display::None,
            flex_direction: FlexDirection::Column,
            padding:        UiRect::all(Val::Px(10.0)),
            row_gap:        Val::Px(3.0),
            border:         UiRect::all(Val::Px(1.0)),
            border_radius:  BorderRadius::all(Val::Px(8.0)),
            min_width:      Val::Px(160.0),
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(border),
        ZIndex(150),
    )).with_children(|p| {
        p.spawn((Text::new("—"), TextFont { font_size: 13.0, ..default() }, TextColor(ThemeColor::SuccessText.resolve()), TooltipName));
        p.spawn((Text::new(""), TextFont { font_size: SIZE_SM, ..default() }, TextColor(dim), TooltipStats));
        p.spawn((Text::new(""), TextFont { font_size: SIZE_SM, ..default() }, TextColor(dim), TooltipExtra));
    });
}

pub fn update_tooltip(
    bridge:    Res<SimBridge>,
    windows:   Query<&Window>,
    cam_q:     Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    agents_q:  Query<(&AgentIndex, &Transform), With<AgentMarker>>,
    mut panel: Query<(&mut Node, &mut Visibility), With<TooltipPanel>>,
    mut texts: ParamSet<(
        Query<&mut Text, With<TooltipName>>,
        Query<&mut Text, With<TooltipStats>>,
        Query<&mut Text, With<TooltipExtra>>,
    )>,
) {
    let Ok(window) = windows.single() else { return };
    let Ok((camera, cam_tf)) = cam_q.single() else { return };
    let Ok((mut panel_node, mut vis)) = panel.single_mut() else { return };

    let Some(cursor_px) = window.cursor_position() else {
        panel_node.display = Display::None;
        *vis = Visibility::Hidden;
        return;
    };

    let hit_radius_sq = (config::TILE_SIZE * HIT_RADIUS_FACTOR).powi(2);
    let world_cursor = match camera.viewport_to_world_2d(cam_tf, cursor_px) {
        Ok(w)  => w,
        Err(_) => {
            panel_node.display = Display::None;
            *vis = Visibility::Hidden;
            return;
        }
    };

    let mut closest: Option<usize> = None;
    let mut best_dist = f32::MAX;
    for (idx, tf) in agents_q.iter() {
        let d = world_cursor.distance_squared(tf.translation.truncate());
        if d < hit_radius_sq && d < best_dist {
            best_dist = d;
            closest = Some(idx.0);
        }
    }

    if let Some(idx) = closest {
        if let Some(agent) = bridge.agents().get(idx) {
            let label = format!("Agent {} — {}", idx, crate::team::Team(agent.team).name());

            // Gold + score.
            let stats = format!(
                "Gold {}/{}  Score {}",
                agent.gold_carried, config::AGENT_MAX_GOLD,
                agent.score,
            );

            // Position + active buff timers (ticks remaining) + policy mode for agent 0.
            let mut buffs = String::new();
            if agent.speed_buff > 0 { buffs += &format!("  ⚡{}", agent.speed_buff); }
            if agent.slow_buff  > 0 { buffs += &format!("  🐌{}", agent.slow_buff); }
            if agent.mult_buff  > 0 { buffs += &format!("  ×2:{}", agent.mult_buff); }
            let mode_str = if idx == 0 {
                format!("  [{}]", bridge.mode.label())
            } else {
                String::new()
            };
            let extra = format!("({}, {}){}{}", agent.pos.x, agent.pos.y, buffs, mode_str);

            for mut t in texts.p0().iter_mut() { *t = Text::new(&label); }
            for mut t in texts.p1().iter_mut() { *t = Text::new(&stats); }
            for mut t in texts.p2().iter_mut() { *t = Text::new(&extra); }

            panel_node.display = Display::Flex;
            panel_node.left    = Val::Px(cursor_px.x + TOOLTIP_OFFSET_X);
            panel_node.top     = Val::Px(cursor_px.y + TOOLTIP_OFFSET_Y);
            *vis = Visibility::Visible;
            return;
        }
    }

    panel_node.display = Display::None;
    *vis = Visibility::Hidden;
}
