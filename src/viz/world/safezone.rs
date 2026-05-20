// src/viz/world/safezone.rs
//
// Draws a border outline around each team's safe zone.
// Reads ResolvedLayout (base positions) and WorldConfig (safe_zone_fraction)
// instead of the deleted `fixed` / `base_safe_radius` fields.

use bevy::prelude::*;
use bevy::color::Alpha;
use crate::world::config::WorldConfig;
use crate::world::Grid;
use crate::world::layout::ResolvedLayout;
use crate::viz::grid_offset::GridOffset;
use crate::style::color::team_color;

pub fn draw_safe_zone_borders(
    mut gizmos: Gizmos,
    offset:     Res<GridOffset>,
    cfg:        Res<WorldConfig>,
    grid:       Res<Grid>,
    layout:     Res<ResolvedLayout>,
) {
    let step   = offset.step;
    let half   = step * 0.5;
    // Derive radius from the same formula WorldConfig uses so the drawn border
    // exactly matches the tile stamping done at world generation.
    let radius = cfg.safe_zone_radius();

    for base in &layout.bases {
        let color = team_color(base.team).with_alpha(0.60);

        // Clamp to grid bounds (same as stamp_safe_zone in world/plugin.rs)
        let min_x = (base.x - radius).max(0);
        let min_y = (base.y - radius).max(0);
        let max_x = (base.x + radius).min(grid.width  as i32 - 1);
        let max_y = (base.y + radius).min(grid.height as i32 - 1);

        // World-space corners of the bounding rectangle
        let tl = offset.world_pos(min_x, max_y) + Vec2::new(-half,  half);
        let tr = offset.world_pos(max_x, max_y) + Vec2::new( half,  half);
        let bl = offset.world_pos(min_x, min_y) + Vec2::new(-half, -half);
        let br = offset.world_pos(max_x, min_y) + Vec2::new( half, -half);

        // Outer border
        gizmos.line_2d(tl, tr, color);
        gizmos.line_2d(bl, br, color);
        gizmos.line_2d(tl, bl, color);
        gizmos.line_2d(tr, br, color);

        // Inner double-line for a thicker feel
        let inset       = 2.0;
        let inner_color = team_color(base.team).with_alpha(0.25);
        let tl2 = tl + Vec2::new( inset, -inset);
        let tr2 = tr + Vec2::new(-inset, -inset);
        let bl2 = bl + Vec2::new( inset,  inset);
        let br2 = br + Vec2::new(-inset,  inset);
        gizmos.line_2d(tl2, tr2, inner_color);
        gizmos.line_2d(bl2, br2, inner_color);
        gizmos.line_2d(tl2, bl2, inner_color);
        gizmos.line_2d(tr2, br2, inner_color);
    }
}