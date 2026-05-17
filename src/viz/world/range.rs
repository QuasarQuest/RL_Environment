// src/viz/world/range.rs
//
// HideRangeViz-gated: melee ring + ranged ring.
// Ranges read from MapConfig so they match combat behaviour exactly.

use bevy::prelude::*;
use crate::agent::brain::AgentBrain;
use crate::agent::components::GridPos;
use crate::viz::components::HideRangeViz;
use crate::viz::grid_offset::GridOffset;
use crate::world::config::WorldConfig;
use crate::style::color::RED_500;

const RING_ALPHA: f32 = 0.18;

pub fn draw_agent_range(
    mut gizmos: Gizmos,
    offset:     Res<GridOffset>,
    map:        Res<WorldConfig>,
    query:      Query<&GridPos, (With<AgentBrain>, Without<HideRangeViz>)>,
) {
    let step = offset.step;

    for pos in query.iter() {
        let world = offset.world_pos(pos.x, pos.y);

        gizmos.circle_2d(
            Isometry2d::from_translation(world),
            map.melee_range as f32 * step + step * 0.5,
            RED_500.with_alpha(RING_ALPHA),
        );

        gizmos.circle_2d(
            Isometry2d::from_translation(world),
            map.ranged_range as f32 * step + step * 0.5,
            Color::srgba(1.0, 0.45, 0.10, RING_ALPHA),
        );
    }
}