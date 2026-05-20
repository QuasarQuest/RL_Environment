// src/viz/agent_renderer.rs
//
// Changes from original:
//   - assign_agent_colours() removed.
//
//     The original code had a conflict: spawn.rs sets each agent's Sprite::color
//     from agent_color() (team colour). assign_agent_colours() then ran on
//     Added<AgentLabel> and overwrote that colour with a hash-derived value,
//     silently defeating the team-colouring logic in registry.rs.
//
//     Colour assignment belongs entirely at spawn time (spawn.rs). If per-algorithm
//     colouring is wanted instead of per-team, change agent_color() in registry.rs —
//     don't fight spawn with a post-spawn override.
//
//     If you want a separate debug colour ring or icon on top of the team sprite,
//     spawn a child entity from spawn.rs or the viz layer — don't mutate the
//     primary Sprite after spawn.

use bevy::prelude::*;
use crate::world::coords::GridPos;
use crate::viz::grid_offset::GridOffset;

/// Syncs GridPos → world Transform every frame.
pub fn sync_agent_transforms(
    offset:    Res<GridOffset>,
    mut query: Query<(&GridPos, &mut Transform)>,
) {
    for (pos, mut transform) in query.iter_mut() {
        let world            = offset.world_pos(pos.x, pos.y);
        transform.translation = Vec3::new(world.x, world.y, 1.0);
    }
}