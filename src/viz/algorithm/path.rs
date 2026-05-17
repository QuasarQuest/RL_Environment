// src/viz/algorithm/path.rs
//
// HidePathViz-gated: path polyline + destination rect from AgentBrain::debug_draw().
// Toggled per-agent via the PATH button in the scoreboard.

use bevy::prelude::*;
use crate::agent::brain::AgentBrain;
use crate::agent::components::GridPos;
use crate::viz::components::HidePathViz;
use crate::viz::grid_offset::GridOffset;

pub fn draw_agent_path(
    mut gizmos: Gizmos,
    offset:     Res<GridOffset>,
    query:      Query<(&GridPos, &AgentBrain), Without<HidePathViz>>,
) {
    for (pos, brain) in query.iter() {
        let Some(drawer) = brain.debug_draw() else { continue };

        // Path polyline
        let lines = drawer.draw_lines(*pos);
        if lines.is_empty() { continue; }

        let mut pts = Vec::with_capacity(lines.len() + 1);
        pts.push(offset.world_pos(lines[0].start.x, lines[0].start.y));
        for line in &lines {
            pts.push(offset.world_pos(line.end.x, line.end.y));
        }
        gizmos.linestrip_2d(pts, lines[0].color);
    }
}