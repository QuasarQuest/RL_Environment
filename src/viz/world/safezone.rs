// src/viz/world/safezone.rs
//
// Draws a solid border outline around each team's safe zone.
// Always visible — not togglable, it's permanent world information.

use bevy::prelude::*;
use bevy::color::Alpha;
use crate::world::config::WorldConfig;
use crate::world::Grid;
use crate::viz::grid_offset::GridOffset;
use crate::style::color::team_color;

pub fn draw_safe_zone_borders(
    mut gizmos: Gizmos,
    offset:     Res<GridOffset>,
    map:        Res<WorldConfig>,
    grid:       Res<Grid>,
) {
    let step   = offset.step;
    let half   = step * 0.5;
    let radius = map.base_safe_radius as i32;

    // Find each base centre and draw one rectangle around the entire safe zone
    for fixed in &map.fixed {
        let team_id = match fixed.tile {
            crate::world::config::TileKind::BaseRed  |
            crate::world::config::TileKind::Base      => 0u8,
            crate::world::config::TileKind::BaseBlue  => 1u8,
            _                                          => continue,
        };

        let color = team_color(team_id).with_alpha(0.60);

        let bx = fixed.x as i32;
        let by = fixed.y as i32;

        // Corner tiles of the safe zone
        let min_x = (bx - radius).max(0);
        let min_y = (by - radius).max(0);
        let max_x = (bx + radius).min(grid.width  as i32 - 1);
        let max_y = (by + radius).min(grid.height as i32 - 1);

        // World-space corners of the bounding rect
        let tl = offset.world_pos(min_x, max_y) + Vec2::new(-half,  half);
        let tr = offset.world_pos(max_x, max_y) + Vec2::new( half,  half);
        let bl = offset.world_pos(min_x, min_y) + Vec2::new(-half, -half);
        let br = offset.world_pos(max_x, min_y) + Vec2::new( half, -half);

        // Draw 4 border lines
        gizmos.line_2d(tl, tr, color); // top
        gizmos.line_2d(bl, br, color); // bottom
        gizmos.line_2d(tl, bl, color); // left
        gizmos.line_2d(tr, br, color); // right

        // Inner double-line for a thicker feel — offset inward by 1 pixel
        let inset = 2.0;
        let tl2 = tl + Vec2::new( inset, -inset);
        let tr2 = tr + Vec2::new(-inset, -inset);
        let bl2 = bl + Vec2::new( inset,  inset);
        let br2 = br + Vec2::new(-inset,  inset);
        let inner_color = team_color(team_id).with_alpha(0.25);
        gizmos.line_2d(tl2, tr2, inner_color);
        gizmos.line_2d(bl2, br2, inner_color);
        gizmos.line_2d(tl2, bl2, inner_color);
        gizmos.line_2d(tr2, br2, inner_color);
    }
}