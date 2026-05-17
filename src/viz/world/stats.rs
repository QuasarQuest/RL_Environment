// src/viz/world/stats.rs
//
// Always-visible agent stat overlays: heart pips left, ammo bar right.
// Not gated by any HideViz variant.

use bevy::prelude::*;
use crate::agent::components::{Ammo, GridPos, Hearts, RespawnIn};
use crate::agent::brain::AgentBrain;
use crate::viz::grid_offset::GridOffset;
use crate::style::color::{RED_500, BLUE_500, GRAY_400};
use crate::config::{AGENT_MAX_AMMO, AGENT_MAX_HEARTS, TILE_SIZE};

const HEART_COL_X: f32 = -(TILE_SIZE * 0.70);
const HEART_R:     f32 = TILE_SIZE * 0.10;
const HEART_STEP:  f32 = TILE_SIZE * 0.28;

const AMMO_COL_X:  f32 = TILE_SIZE * 0.70;
const AMMO_BAR_H:  f32 = TILE_SIZE * 0.40;
const AMMO_BAR_W:  f32 = TILE_SIZE * 0.12;

pub fn draw_agent_stats(
    mut gizmos: Gizmos,
    offset:     Res<GridOffset>,
    query:      Query<(&GridPos, &Hearts, &Ammo), (With<AgentBrain>, Without<RespawnIn>)>,
) {
    for (pos, hearts, ammo) in query.iter() {
        let world = offset.world_pos(pos.x, pos.y);

        // ── Hearts — vertical column, left of agent ───────────────────────────
        let max_h   = AGENT_MAX_HEARTS as usize;
        let total_h = (max_h as f32 - 1.0) * HEART_STEP;
        let base_y  = world.y - total_h / 2.0;

        for i in 0..max_h {
            let pip = Vec2::new(world.x + HEART_COL_X, base_y + i as f32 * HEART_STEP);
            let color = if i < hearts.0 as usize {
                RED_500.with_alpha(0.90)
            } else {
                GRAY_400.with_alpha(0.35)
            };
            gizmos.circle_2d(Isometry2d::from_translation(pip), HEART_R, color);
        }

        // ── Ammo — vertical fill bar, right of agent ──────────────────────────
        let fill_ratio = ammo.0 as f32 / AGENT_MAX_AMMO as f32;
        let bar_cx     = world.x + AMMO_COL_X;
        let bar_bot    = world.y - AMMO_BAR_H;

        // Trough
        gizmos.rect_2d(
            Isometry2d::from_translation(Vec2::new(bar_cx, world.y)),
            Vec2::new(AMMO_BAR_W, AMMO_BAR_H * 2.0),
            GRAY_400.with_alpha(0.25),
        );

        // Fill (grows upward from bottom)
        if fill_ratio > 0.0 {
            let fill_h  = AMMO_BAR_H * 2.0 * fill_ratio;
            let fill_cy = bar_bot + fill_h / 2.0;
            gizmos.rect_2d(
                Isometry2d::from_translation(Vec2::new(bar_cx, fill_cy)),
                Vec2::new(AMMO_BAR_W, fill_h),
                BLUE_500.with_alpha(0.85),
            );
        }
    }
}