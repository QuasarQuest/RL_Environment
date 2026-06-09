// src/engine/obs.rs
//
// Builds the CNN observation for the RL agent.
// Writes into a caller-supplied buffer — no per-step heap allocation.
//
// Single-agent gold rush: the agent observes gold, the base, obstacles and the
// two flavour items (speed boost, score multiplier), plus its own carried gold
// and active buff timers.
//
// Buffer layout (OBS_TOTAL floats):
//   [0 .. OBS_DIM)                  egocentric crop  (OBS_CHANNELS × 25 × 25)
//   [OBS_DIM .. OBS_DIM+MM_DIM)     minimap          (MM_CHANNELS  × 17 × 17)
//   [OBS_DIM+MM_DIM .. OBS_TOTAL)   cluster features (CLUSTER_K × 4)
//
// Channel layout is defined in rl/obs.rs (authoritative constants).

use crate::config::{AGENT_MAX_GOLD, SPEED_BUFF_MAX};
use crate::entity::AgentState;
use crate::entity::item::{ItemKind, ItemState};
use crate::engine::clusters::GoldCluster;
use crate::rl::action::CLUSTER_K;
use crate::rl::obs::{
    CH_BASE, CH_BASE_DX, CH_BASE_DY, CH_CARRYING, CH_GOLD, CH_MULT,
    CH_MULT_CHARGE, CH_OBSTACLE, CH_OOB, CH_SPEED,
    CH_SPEED_REMAINING, CH_TIME_REMAINING,
    MM_CH_GOLD, MM_CH_MULT, MM_CH_OBSTACLE, MM_CH_SPEED,
    MM_CHANNELS, MM_DIM, MM_SIZE, OBS_CROP_SIZE, OBS_DIM, OBS_TOTAL,
};
use crate::world::coords::GridPos;
use crate::world::grid::Grid;
use crate::world::tile::Tile;

// Normaliser for cluster gold count (agent rarely sees more than this in one cluster).
const CLUSTER_COUNT_NORM: f32 = 25.0;

pub fn build_obs_into(
    buf:            &mut [f32],
    agent:          &AgentState,
    gold_positions: &[GridPos],
    items:          &[ItemState],
    grid:           &Grid,
    clusters:       &[Option<GoldCluster>; CLUSTER_K],
    time_remaining: f32,
) {
    debug_assert_eq!(buf.len(), OBS_TOTAL,
        "buf must be exactly OBS_TOTAL={OBS_TOTAL} floats");
    buf.fill(0.0);

    let centre = (OBS_CROP_SIZE / 2) as i32;
    let (ax, ay) = (agent.pos.x, agent.pos.y);
    let (gw, gh) = (grid.width as i32, grid.height as i32);
    let plane    = OBS_CROP_SIZE * OBS_CROP_SIZE;

    // ── Broadcast: agent scalars ──────────────────────────────────────────────

    buf[CH_CARRYING * plane..(CH_CARRYING + 1) * plane]
        .fill(agent.gold_carried as f32 / AGENT_MAX_GOLD as f32);

    let base_dx = (agent.base_pos.x - ax) as f32 / gw as f32;
    let base_dy = (agent.base_pos.y - ay) as f32 / gh as f32;
    buf[CH_BASE_DX * plane..(CH_BASE_DX + 1) * plane].fill(base_dx);
    buf[CH_BASE_DY * plane..(CH_BASE_DY + 1) * plane].fill(base_dy);

    buf[CH_TIME_REMAINING * plane..(CH_TIME_REMAINING + 1) * plane]
        .fill(time_remaining.clamp(0.0, 1.0));

    // Active buff timers, normalised by their longest possible window.
    buf[CH_SPEED_REMAINING * plane..(CH_SPEED_REMAINING + 1) * plane]
        .fill((agent.speed_buff as f32 / SPEED_BUFF_MAX as f32).min(1.0));
    // Multiplier is a held charge (not a timer): 1.0 while one is banked-up to spend.
    buf[CH_MULT_CHARGE * plane..(CH_MULT_CHARGE + 1) * plane]
        .fill((agent.mult_charge > 0) as u8 as f32);

    // ── Tile scan: OOB + base + obstacles ────────────────────────────────────

    for cy in 0..OBS_CROP_SIZE as i32 {
        let wy      = ay + cy - centre;
        let row_oob = wy < 0 || wy >= gh;
        for cx in 0..OBS_CROP_SIZE as i32 {
            let wx = ax + cx - centre;
            if row_oob || wx < 0 || wx >= gw {
                buf[pixel(CH_OOB, cx, cy)] = 1.0;
            } else {
                let tile = unsafe { grid.get_unchecked(wx, wy) };
                match tile {
                    Tile::Base(_)  => buf[pixel(CH_BASE,     cx, cy)] = 1.0,
                    Tile::Obstacle => buf[pixel(CH_OBSTACLE, cx, cy)] = 1.0,
                    _ => {}
                }
            }
        }
    }

    // ── Gold (crop) ────────────────────────────────────────────────────────────

    for &gpos in gold_positions {
        let cx = gpos.x - ax + centre;
        let cy = gpos.y - ay + centre;
        if in_crop(cx, cy) {
            buf[pixel(CH_GOLD, cx, cy)] = 1.0;
        }
    }

    // ── Flavour items (crop): speed / multiplier ─────────────────────────────

    for it in items {
        let cx = it.pos.x - ax + centre;
        let cy = it.pos.y - ay + centre;
        if !in_crop(cx, cy) { continue; }
        match it.kind {
            ItemKind::Speed      => buf[pixel(CH_SPEED, cx, cy)] = 1.0,
            ItemKind::Multiplier => buf[pixel(CH_MULT, cx, cy)] = 1.0,
            ItemKind::Gold       => {} // already drawn from gold_positions
        }
    }

    // ── Minimap (17×17 cells, ~3×3 tiles/cell for 50×50 map) ─────────────────

    build_minimap(&mut buf[OBS_DIM..OBS_DIM + MM_DIM], gold_positions, items, grid);

    // ── Cluster features (CLUSTER_K × 4 floats after minimap) ─────────────────
    // [dx_norm, dy_norm, pathdist_norm, count_norm] per fixed region. The "nearest
    // gold" used for direction + distance is the PATH-nearest (BFS around walls),
    // so the policy's distance perception is correct around obstacles.

    let dist = crate::engine::nav::dist_field(grid, agent.pos);
    let path_norm = (gw + gh) as f32; // longest plausible corridor distance
    let cluster_start = OBS_DIM + MM_DIM;
    for (k, maybe_cluster) in clusters.iter().enumerate().take(CLUSTER_K) {
        let base = cluster_start + k * 4;
        if let Some(c) = maybe_cluster {
            // Pick the region's gold with the smallest true path distance; fall back
            // to the Chebyshev-nearest for direction if none is reachable.
            let path_nearest = c.golds.iter()
                .filter_map(|&g| crate::engine::nav::dist_at(&dist, grid, g).map(|d| (d, g)))
                .min_by_key(|&(d, _)| d);
            let (target, pathdist) = match path_nearest {
                Some((d, g)) => (Some(g), (d as f32 / path_norm).min(1.0)),
                None         => (c.nearest_gold(agent.pos), 1.0), // all walled off
            };
            if let Some(g) = target {
                buf[base]     = (g.x - ax) as f32 / gw as f32;
                buf[base + 1] = (g.y - ay) as f32 / gh as f32;
            }
            buf[base + 2] = pathdist;
            buf[base + 3] = (c.count() as f32 / CLUSTER_COUNT_NORM).min(1.0);
        }
    }
}

// ── Minimap builder ───────────────────────────────────────────────────────────

fn build_minimap(
    mm:             &mut [f32],
    gold_positions: &[GridPos],
    items:          &[ItemState],
    grid:           &Grid,
) {
    debug_assert_eq!(mm.len(), MM_CHANNELS * MM_SIZE * MM_SIZE);

    let mm_plane = MM_SIZE * MM_SIZE;
    let gw = grid.width  as f32;
    let gh = grid.height as f32;

    let to_mm = |wx: i32, wy: i32| -> (usize, usize) {
        let mx = ((wx as f32 / gw) * MM_SIZE as f32) as usize;
        let my = ((wy as f32 / gh) * MM_SIZE as f32) as usize;
        (mx.min(MM_SIZE - 1), my.min(MM_SIZE - 1))
    };

    let mm_pixel = |ch: usize, mx: usize, my: usize| -> usize {
        ch * mm_plane + my * MM_SIZE + mx
    };

    // Obstacles — scan full grid once.
    for (wx, wy, tile) in grid.iter() {
        if tile == Tile::Obstacle {
            let (mx, my) = to_mm(wx as i32, wy as i32);
            mm[mm_pixel(MM_CH_OBSTACLE, mx, my)] = 1.0;
        }
    }

    // Gold positions (pre-computed vec).
    for &gpos in gold_positions {
        let (mx, my) = to_mm(gpos.x, gpos.y);
        mm[mm_pixel(MM_CH_GOLD, mx, my)] = 1.0;
    }

    // Flavour items.
    for it in items {
        let ch = match it.kind {
            ItemKind::Speed      => MM_CH_SPEED,
            ItemKind::Multiplier => MM_CH_MULT,
            ItemKind::Gold       => continue,
        };
        let (mx, my) = to_mm(it.pos.x, it.pos.y);
        mm[mm_pixel(ch, mx, my)] = 1.0;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline(always)]
fn in_crop(cx: i32, cy: i32) -> bool {
    let size = OBS_CROP_SIZE as i32;
    cx >= 0 && cx < size && cy >= 0 && cy < size
}

#[inline(always)]
fn pixel(ch: usize, cx: i32, cy: i32) -> usize {
    ch * OBS_CROP_SIZE * OBS_CROP_SIZE + cy as usize * OBS_CROP_SIZE + cx as usize
}
