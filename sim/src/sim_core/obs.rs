// src/sim_core/obs.rs
//
// Builds the CNN observation for the RL agent.
//
// Channels (CHW, 5 × 25 × 25):
//   0  Out-of-bounds mask
//   1  Own base
//   2  Gold items
//   3  Self at centre
//   4  Carrying gold (broadcast plane)
//
// Writes into a caller-supplied buffer to avoid per-step heap allocation.
//
// Gold channel performance
// ------------------------
// Previously the gold loop iterated over all ItemState entries and filtered
// by kind. With multiple item types (Ammo, Health, SpeedBoost) this wastes
// cycles on non-gold items every step.
//
// Now the caller passes a pre-filtered `gold_positions: &[GridPos]` slice
// maintained by SimCore. The slice is updated only on pickup (O(1) swap_remove
// mirror) — the hot obs path becomes O(gold_count) with no branching.

use crate::rl::obs::{OBS_CROP_SIZE, OBS_TOTAL};
use crate::world::coords::GridPos;
use crate::world::grid::Grid;
use crate::world::tile::Tile;
use super::state::AgentState;

pub fn build_obs_into(
    buf:            &mut [f32],
    agent:          &AgentState,
    gold_positions: &[GridPos],
    grid:           &Grid,
) {
    debug_assert_eq!(buf.len(), OBS_TOTAL);
    buf.fill(0.0);

    let centre = (OBS_CROP_SIZE / 2) as i32;
    let (ax, ay) = (agent.pos.x, agent.pos.y);
    let (gw, gh) = (grid.width as i32, grid.height as i32);

    // Ch 4: carrying gold — broadcast entire plane
    if agent.gold_carried > 0 {
        let base = 4 * OBS_CROP_SIZE * OBS_CROP_SIZE;
        buf[base..base + OBS_CROP_SIZE * OBS_CROP_SIZE].fill(1.0);
    }

    // Ch 3: self at crop centre
    buf[pixel(3, centre, centre)] = 1.0;

    // Ch 0 (OOB) + Ch 1 (own base) — tight inner loop using unchecked indexing
    for cy in 0..OBS_CROP_SIZE as i32 {
        let wy = ay + cy - centre;
        let row_oob = wy < 0 || wy >= gh;
        for cx in 0..OBS_CROP_SIZE as i32 {
            let wx = ax + cx - centre;
            if row_oob || wx < 0 || wx >= gw {
                buf[pixel(0, cx, cy)] = 1.0;
            } else {
                // SAFETY: row_oob is false and wx/wy are in [0, gw/gh)
                let tile = unsafe { grid.get_unchecked(wx, wy) };
                if let Tile::Base(t) = tile {
                    if t == agent.team {
                        buf[pixel(1, cx, cy)] = 1.0;
                    }
                }
            }
        }
    }

    // Ch 2: gold positions inside the crop window.
    // gold_positions is pre-filtered — no kind check needed here.
    for &gpos in gold_positions {
        let cx = gpos.x - ax + centre;
        let cy = gpos.y - ay + centre;
        if cx >= 0 && cx < OBS_CROP_SIZE as i32 && cy >= 0 && cy < OBS_CROP_SIZE as i32 {
            buf[pixel(2, cx, cy)] = 1.0;
        }
    }
}

#[inline(always)]
fn pixel(ch: usize, cx: i32, cy: i32) -> usize {
    ch * OBS_CROP_SIZE * OBS_CROP_SIZE + cy as usize * OBS_CROP_SIZE + cx as usize
}