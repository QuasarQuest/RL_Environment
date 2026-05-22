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

use crate::item::ItemKind;
use crate::rl::obs::{OBS_CROP_SIZE, OBS_TOTAL};
use crate::world::grid::Grid;
use crate::world::tile::Tile;
use super::state::{AgentState, ItemState};

pub fn build_obs(agent: &AgentState, items: &[ItemState], grid: &Grid) -> Vec<f32> {
    let mut buf = vec![0.0f32; OBS_TOTAL];
    let centre  = (OBS_CROP_SIZE / 2) as i32;
    let (ax, ay) = (agent.pos.x, agent.pos.y);
    let (gw, gh) = (grid.width as i32, grid.height as i32);

    // Ch 4: carrying gold — broadcast plane
    if agent.gold_carried > 0 {
        let base = 4 * OBS_CROP_SIZE * OBS_CROP_SIZE;
        buf[base..base + OBS_CROP_SIZE * OBS_CROP_SIZE].fill(1.0);
    }

    // Ch 3: self at centre
    buf[idx(3, centre, centre)] = 1.0;

    // Ch 0 (OOB) + Ch 1 (own base)
    for cy in 0..OBS_CROP_SIZE as i32 {
        let wy = ay + cy - centre;
        for cx in 0..OBS_CROP_SIZE as i32 {
            let wx = ax + cx - centre;
            if wx < 0 || wx >= gw || wy < 0 || wy >= gh {
                buf[idx(0, cx, cy)] = 1.0;
            } else if let Some(Tile::Base(t)) = grid.get(wx, wy) {
                if t == agent.team {
                    buf[idx(1, cx, cy)] = 1.0;
                }
            }
        }
    }

    // Ch 2: gold items in crop window
    for item in items {
        if item.kind != ItemKind::Gold { continue; }
        let cx = item.pos.x - ax + centre;
        let cy = item.pos.y - ay + centre;
        if (0..OBS_CROP_SIZE as i32).contains(&cx) && (0..OBS_CROP_SIZE as i32).contains(&cy) {
            buf[idx(2, cx, cy)] = 1.0;
        }
    }

    buf
}

#[inline]
fn idx(ch: usize, cx: i32, cy: i32) -> usize {
    ch * OBS_CROP_SIZE * OBS_CROP_SIZE + cy as usize * OBS_CROP_SIZE + cx as usize
}
