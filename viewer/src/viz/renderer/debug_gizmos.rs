// Gizmo overlays: A* path polyline and the spawn-pocket border.
//
// Keys:
//   V — toggle path gizmo (all policy modes except Random)

use bevy::prelude::*;
use atb::config::TILE_SIZE;
use atb::world::config::SPAWN_POCKET_RADIUS;
use crate::sim_bridge::{SimBridge, PolicyMode};
use crate::style::color::{PATH_GOAL, PATH_LINE, SPAWN_POCKET_BORDER};
use crate::viz::grid_offset::GridOffset;

#[derive(Resource, Default)]
pub struct DebugGizmoFlags {
    pub show_path: bool,
}

pub fn toggle_path_viz(
    keys:     Res<ButtonInput<KeyCode>>,
    mut flags: ResMut<DebugGizmoFlags>,
) {
    if keys.just_pressed(KeyCode::KeyV) {
        flags.show_path = !flags.show_path;
    }
}

/// Draw the RL agent's current A* path (available for all non-random modes).
pub fn draw_path_gizmo(
    bridge: Res<SimBridge>,
    flags:  Res<DebugGizmoFlags>,
    offset: Res<GridOffset>,
    mut gizmos: Gizmos,
) {
    if !flags.show_path { return; }
    if bridge.mode == PolicyMode::Random { return; }

    let waypoints = bridge.agent_path();
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
        gizmos.line_2d(w[0], w[1], PATH_LINE);
    }

    // Goal marker at the last waypoint.
    if let Some(&dest) = pts.last() {
        let half = TILE_SIZE * 0.30;
        gizmos.rect_2d(
            Isometry2d::from_translation(dest),
            Vec2::splat(half * 2.0),
            PATH_GOAL,
        );
    }
}

/// Draw the border around the obstacle-free spawn pocket (7×7 around the base).
/// Pairs with the translucent fill sprite in `agent_renderer`. Always on; follows
/// the per-episode randomised base.
pub fn draw_spawn_pocket_border(
    bridge: Res<SimBridge>,
    offset: Res<GridOffset>,
    mut gizmos: Gizmos,
) {
    let Some(agent) = bridge.agents().first() else { return };
    let centre = offset.world_pos(agent.base_pos.x, agent.base_pos.y);
    let side   = (2 * SPAWN_POCKET_RADIUS + 1) as f32 * offset.step;
    gizmos.rect_2d(
        Isometry2d::from_translation(centre),
        Vec2::splat(side),
        SPAWN_POCKET_BORDER,
    );
}
