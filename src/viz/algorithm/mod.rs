// src/viz/algorithm/mod.rs

use bevy::prelude::*;
use crate::agent::brain::AgentBrain; // 1. Import AgentBrain instead of Brain
use crate::agent::components::GridPos;
use crate::viz::grid_offset::GridOffset;
use crate::viz::menu::components::HideViz;
use crate::config;

pub fn draw_agent_debug(
    mut gizmos: Gizmos,
    offset: Res<GridOffset>,
    query: Query<(&GridPos, &AgentBrain), Without<HideViz>>, // 2. Query for AgentBrain
) {
    let half = config::TILE_SIZE * 0.45;

    for (pos, brain) in query.iter() {
        // Note: If you didn't delegate this method on AgentBrain,
        // you may need to call brain.0.debug_draw() instead.
        if let Some(drawer) = brain.debug_draw() {

            // Render Rectangles (Open/Closed sets, Obstacles)
            for rect in drawer.draw_rects() {
                let world_pos = offset.world_pos(rect.pos.x, rect.pos.y);
                gizmos.rect_2d(Isometry2d::from_translation(world_pos), Vec2::splat(half), rect.color);
            }

            // Render Lines (Paths)
            let lines = drawer.draw_lines(*pos);
            if !lines.is_empty() {
                let mut pts = Vec::with_capacity(lines.len() + 1);

                // Form a continuous linestrip from the line segments
                pts.push(offset.world_pos(lines[0].start.x, lines[0].start.y));
                for line in lines.iter() {
                    pts.push(offset.world_pos(line.end.x, line.end.y));
                }

                let color = lines.first().map(|l| l.color).unwrap_or(Color::WHITE);
                gizmos.linestrip_2d(pts, color);
            }
        }
    }
}