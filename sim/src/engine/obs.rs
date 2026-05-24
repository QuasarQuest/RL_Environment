// src/engine/obs.rs
//
// Builds the CNN observation for the RL agent each tick.
// Writes into a caller-supplied buffer — no per-step heap allocation.
//
// Channel layout is defined in rl/obs.rs (authoritative constants).

use crate::config::{AGENT_MAX_AMMO, AGENT_MAX_HEARTS};
use crate::entity::item::ItemKind;
use crate::entity::{AgentState, ItemState};
use crate::rl::obs::{
    CH_AMMO, CH_BASE, CH_CARRYING, CH_ENEMY, CH_GOLD,
    CH_HEALTH, CH_ITEMS, CH_OBSTACLE, CH_OOB,
    ITEM_AMMO, ITEM_HEALTH, ITEM_SPEED,
    OBS_CROP_SIZE, OBS_TOTAL,
};
use crate::world::grid::Grid;
use crate::world::tile::Tile;

pub fn build_obs_into(
    buf:    &mut [f32],
    agent:  &AgentState,
    items:  &[ItemState],
    agents: &[AgentState],
    grid:   &Grid,
) {
    debug_assert_eq!(buf.len(), OBS_TOTAL);
    buf.fill(0.0);

    let centre = (OBS_CROP_SIZE / 2) as i32;
    let (ax, ay) = (agent.pos.x, agent.pos.y);
    let (gw, gh) = (grid.width as i32, grid.height as i32);
    let plane    = OBS_CROP_SIZE * OBS_CROP_SIZE;

    // ── Broadcast channels ────────────────────────────────────────────────────

    if agent.gold_carried > 0 {
        buf[CH_CARRYING * plane..(CH_CARRYING + 1) * plane].fill(1.0);
    }
    buf[CH_HEALTH * plane..(CH_HEALTH + 1) * plane]
        .fill(agent.hearts as f32 / AGENT_MAX_HEARTS as f32);
    buf[CH_AMMO * plane..(CH_AMMO + 1) * plane]
        .fill(agent.ammo as f32 / AGENT_MAX_AMMO as f32);

    // ── Tile scan: OOB + base + obstacles ─────────────────────────────────────

    for cy in 0..OBS_CROP_SIZE as i32 {
        let wy      = ay + cy - centre;
        let row_oob = wy < 0 || wy >= gh;
        for cx in 0..OBS_CROP_SIZE as i32 {
            let wx = ax + cx - centre;
            if row_oob || wx < 0 || wx >= gw {
                buf[pixel(CH_OOB, cx, cy)] = 1.0;
            } else {
                // SAFETY: bounds checked above.
                let tile = unsafe { grid.get_unchecked(wx, wy) };
                match tile {
                    Tile::Base(t) if t == agent.team => buf[pixel(CH_BASE,     cx, cy)] = 1.0,
                    Tile::Obstacle                    => buf[pixel(CH_OBSTACLE, cx, cy)] = 1.0,
                    _ => {}
                }
            }
        }
    }

    // ── Items ─────────────────────────────────────────────────────────────────

    for item in items {
        let cx = item.pos.x - ax + centre;
        let cy = item.pos.y - ay + centre;
        if cx < 0 || cx >= OBS_CROP_SIZE as i32 || cy < 0 || cy >= OBS_CROP_SIZE as i32 {
            continue;
        }
        match item.kind {
            ItemKind::Gold       => buf[pixel(CH_GOLD,  cx, cy)] = 1.0,
            ItemKind::Health     => buf[pixel(CH_ITEMS, cx, cy)] = ITEM_HEALTH,
            ItemKind::Ammo       => buf[pixel(CH_ITEMS, cx, cy)] = ITEM_AMMO,
            ItemKind::SpeedBoost => buf[pixel(CH_ITEMS, cx, cy)] = ITEM_SPEED,
        }
    }

    // ── Enemies ───────────────────────────────────────────────────────────────

    for other in agents {
        if other.team == agent.team { continue; }
        let cx = other.pos.x - ax + centre;
        let cy = other.pos.y - ay + centre;
        if cx >= 0 && cx < OBS_CROP_SIZE as i32 && cy >= 0 && cy < OBS_CROP_SIZE as i32 {
            buf[pixel(CH_ENEMY, cx, cy)] = 1.0;
        }
    }
}

#[inline(always)]
fn pixel(ch: usize, cx: i32, cy: i32) -> usize {
    ch * OBS_CROP_SIZE * OBS_CROP_SIZE + cy as usize * OBS_CROP_SIZE + cx as usize
}
