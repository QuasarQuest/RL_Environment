// src/viz/algorithm/range.rs
//
// HideRangeViz-gated: melee ring (always) + ranged ring (always).
// Toggled per-agent via the RANGE button in the scoreboard.

use bevy::prelude::*;
use crate::agent::brain::AgentBrain;
use crate::agent::components::GridPos;
use crate::viz::components::HideRangeViz;
use crate::viz::grid_offset::GridOffset;
use crate::style::color::RED_500;
use crate::config::{MELEE_RANGE, RANGED_RANGE};

const RING_ALPHA: f32 = 0.18;

pub fn draw_agent_range(
    mut gizmos: Gizmos,
    offset:     Res<GridOffset>,
    query:      Query<&GridPos, (With<AgentBrain>, Without<HideRangeViz>)>,
) {
    let step = offset.step;

    for pos in query.iter() {
        let world = offset.world_pos(pos.x, pos.y);

        // Melee ring
        gizmos.circle_2d(
            Isometry2d::from_translation(world),
            MELEE_RANGE as f32 * step + step * 0.5,
            RED_500.with_alpha(RING_ALPHA),
        );

        // Ranged ring — shown regardless of current ammo count
        gizmos.circle_2d(
            Isometry2d::from_translation(world),
            RANGED_RANGE as f32 * step + step * 0.5,
            Color::srgba(1.0, 0.45, 0.10, RING_ALPHA),
        );
    }
}