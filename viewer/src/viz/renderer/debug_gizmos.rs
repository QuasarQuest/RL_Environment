// Gizmo overlays: A* path polyline and combat-range rings.
//
// Keys:
//   V — toggle path gizmo (BT / GOAP modes only)
//   R — toggle melee + ranged range rings around every agent

use bevy::prelude::*;
use atb::config::{MELEE_RANGE, RANGED_RANGE, TILE_SIZE};
use crate::sim_bridge::{SimBridge, PolicyMode, AgentIndex, AgentMarker};
use crate::style::color::{RED_500, GOLD_500};
use crate::viz::grid_offset::GridOffset;

const PATH_COLOR:   Color = Color::srgba(0.20, 0.90, 0.60, 0.85);
const GOAL_COLOR:   Color = Color::srgba(0.20, 0.90, 0.60, 0.45);
const MELEE_ALPHA:  f32   = 0.18;
const RANGED_ALPHA: f32   = 0.13;

#[derive(Resource, Default)]
pub struct DebugGizmoFlags {
    pub show_path:   bool,
    pub show_ranges: bool,
}

pub fn toggle_path_viz(
    keys:     Res<ButtonInput<KeyCode>>,
    mut flags: ResMut<DebugGizmoFlags>,
) {
    if keys.just_pressed(KeyCode::KeyV) {
        flags.show_path = !flags.show_path;
    }
}

pub fn toggle_range_viz(
    keys:     Res<ButtonInput<KeyCode>>,
    mut flags: ResMut<DebugGizmoFlags>,
) {
    if keys.just_pressed(KeyCode::KeyR) {
        flags.show_ranges = !flags.show_ranges;
    }
}

/// Draw the current A* path for the RL agent when using BT or GOAP mode.
pub fn draw_path_gizmo(
    bridge: Res<SimBridge>,
    flags:  Res<DebugGizmoFlags>,
    offset: Res<GridOffset>,
    mut gizmos: Gizmos,
) {
    if !flags.show_path { return; }

    let waypoints = match bridge.mode {
        PolicyMode::BehaviorTree => bridge.bt_path(),
        PolicyMode::Goap         => bridge.goap_path(),
        _                        => return,
    };
    if waypoints.is_empty() { return; }

    let Some(agent) = bridge.agents().first() else { return };

    // Polyline from agent pos through every cached waypoint.
    let start = offset.world_pos(agent.pos.x, agent.pos.y);
    let mut pts: Vec<Vec2> = Vec::with_capacity(waypoints.len() + 1);
    pts.push(start);
    for wp in waypoints {
        pts.push(offset.world_pos(wp.x, wp.y));
    }

    for w in pts.windows(2) {
        gizmos.line_2d(w[0], w[1], PATH_COLOR);
    }

    // Goal marker at the last waypoint.
    if let Some(&dest) = pts.last() {
        let half = TILE_SIZE * 0.30;
        gizmos.rect_2d(
            Isometry2d::from_translation(dest),
            Vec2::splat(half * 2.0),
            GOAL_COLOR,
        );
    }
}

/// Draw melee and ranged rings centred on every agent.
pub fn draw_range_gizmos(
    flags:     Res<DebugGizmoFlags>,
    offset:    Res<GridOffset>,
    agents_q:  Query<(&AgentIndex, &Transform), With<AgentMarker>>,
    mut gizmos: Gizmos,
) {
    if !flags.show_ranges { return; }

    let step = offset.step;

    for (_, tf) in agents_q.iter() {
        let world = tf.translation.truncate();
        let iso   = Isometry2d::from_translation(world);

        gizmos.circle_2d(
            iso,
            MELEE_RANGE as f32 * step + step * 0.5,
            RED_500.with_alpha(MELEE_ALPHA),
        );

        gizmos.circle_2d(
            iso,
            RANGED_RANGE as f32 * step + step * 0.5,
            GOLD_500.with_alpha(RANGED_ALPHA),
        );
    }
}
