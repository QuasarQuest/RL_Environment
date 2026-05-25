// src/engine/obs.rs
//
// Builds the CNN observation for the RL agent each tick.
// Writes into a caller-supplied buffer — no per-step heap allocation.
//
// Buffer layout (OBS_TOTAL floats):
//   [0 .. OBS_DIM)       main egocentric crop  (OBS_CHANNELS × 25 × 25)
//   [OBS_DIM .. OBS_TOTAL) minimap              (MM_CHANNELS  ×  7 ×  7)
//
// Channel layout is defined in rl/obs.rs (authoritative constants).

use crate::config::{AGENT_MAX_AMMO, AGENT_MAX_GOLD, AGENT_MAX_HEARTS};
use crate::entity::item::ItemKind;
use crate::entity::{AgentState, ItemState};
use crate::rl::obs::{
    CH_AMMO, CH_BASE, CH_BASE_DX, CH_BASE_DY, CH_CARRYING, CH_ENEMY,
    CH_ENEMY_AMMO, CH_ENEMY_HP, CH_GOLD, CH_HEALTH, CH_ITEMS, CH_OBSTACLE, CH_OOB,
    ITEM_AMMO, ITEM_HEALTH, ITEM_SPEED,
    MM_CH_ENEMY, MM_CH_GOLD, MM_CH_OBSTACLE,
    MM_CHANNELS, MM_SIZE,
    OBS_CHANNELS, OBS_CROP_SIZE, OBS_DIM, OBS_TOTAL,
};
use crate::world::coords::GridPos;
use crate::world::grid::Grid;
use crate::world::tile::Tile;

pub fn build_obs_into(
    buf:            &mut [f32],
    agent:          &AgentState,
    items:          &[ItemState],
    agents:         &[AgentState],
    gold_positions: &[GridPos],
    grid:           &Grid,
) {
    // Explicit size check — no magic numbers.
    // OBS_TOTAL = OBS_CHANNELS * OBS_CROP_SIZE² + MM_CHANNELS * MM_SIZE²
    debug_assert_eq!(
        buf.len(),
        OBS_CHANNELS * OBS_CROP_SIZE * OBS_CROP_SIZE + MM_CHANNELS * MM_SIZE * MM_SIZE,
        "buf must be exactly OBS_CHANNELS*OBS_CROP_SIZE²+MM_CHANNELS*MM_SIZE² floats"
    );
    debug_assert_eq!(buf.len(), OBS_TOTAL); // cross-check against const
    buf.fill(0.0);

    let centre = (OBS_CROP_SIZE / 2) as i32;
    let (ax, ay) = (agent.pos.x, agent.pos.y);
    let (gw, gh) = (grid.width as i32, grid.height as i32);
    let plane    = OBS_CROP_SIZE * OBS_CROP_SIZE;

    // ── Broadcast: agent scalars ──────────────────────────────────────────────

    buf[CH_CARRYING * plane..(CH_CARRYING + 1) * plane]
        .fill(agent.gold_carried as f32 / AGENT_MAX_GOLD as f32);
    buf[CH_HEALTH * plane..(CH_HEALTH + 1) * plane]
        .fill(agent.hearts as f32 / AGENT_MAX_HEARTS as f32);
    buf[CH_AMMO * plane..(CH_AMMO + 1) * plane]
        .fill(agent.ammo as f32 / AGENT_MAX_AMMO as f32);

    // ── Broadcast: base direction (always points to own base) ─────────────────
    //
    // Always encodes direction to BASE — not a goal-switch.
    // Gold navigation comes from CH_GOLD in the spatial crop.
    // Normalised by map dimensions → ∈ [-1, 1] regardless of map size.

    let base_dx = (agent.base_pos.x - ax) as f32 / gw as f32;
    let base_dy = (agent.base_pos.y - ay) as f32 / gh as f32;
    buf[CH_BASE_DX * plane..(CH_BASE_DX + 1) * plane].fill(base_dx);
    buf[CH_BASE_DY * plane..(CH_BASE_DY + 1) * plane].fill(base_dy);

    // ── Tile scan: OOB + base + obstacles ────────────────────────────────────

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
        if !in_crop(cx, cy) { continue; }
        match item.kind {
            ItemKind::Gold       => buf[pixel(CH_GOLD,  cx, cy)] = 1.0,
            ItemKind::Health     => buf[pixel(CH_ITEMS, cx, cy)] = ITEM_HEALTH,
            ItemKind::Ammo       => buf[pixel(CH_ITEMS, cx, cy)] = ITEM_AMMO,
            ItemKind::SpeedBoost => buf[pixel(CH_ITEMS, cx, cy)] = ITEM_SPEED,
        }
    }

    // ── Enemies — position + HP + ammo ───────────────────────────────────────
    //
    // CH_ENEMY:      1.0 at enemy pixel (binary presence)
    // CH_ENEMY_HP:   hearts/max at enemy pixel, 0 elsewhere
    // CH_ENEMY_AMMO: ammo/max   at enemy pixel, 0 elsewhere
    //
    // All three are 0 in stages 1–3 (no enemy present) — no stage branching needed.

    for other in agents {
        if other.team == agent.team { continue; }
        let cx = other.pos.x - ax + centre;
        let cy = other.pos.y - ay + centre;
        if in_crop(cx, cy) {
            buf[pixel(CH_ENEMY,      cx, cy)] = 1.0;
            buf[pixel(CH_ENEMY_HP,   cx, cy)] = other.hearts as f32 / AGENT_MAX_HEARTS as f32;
            buf[pixel(CH_ENEMY_AMMO, cx, cy)] = other.ammo   as f32 / AGENT_MAX_AMMO   as f32;
        }
    }

    // ── Minimap (appended after main crop) ───────────────────────────────────
    //
    // 7×7 grid where each cell covers ~3.5×3.5 map tiles (for 25×25 maps).
    // Each cell takes the MAX over the covered tiles (presence semantics).
    // Channels: obstacles, enemy, gold.
    //
    // Written into buf[OBS_DIM .. OBS_TOTAL].

    build_minimap(&mut buf[OBS_DIM..], agents, items, gold_positions, agent, grid);
}

// ── Minimap builder ───────────────────────────────────────────────────────────

fn build_minimap(
    mm:             &mut [f32],   // MM_DIM floats, pre-zeroed
    agents:         &[AgentState],
    items:          &[ItemState],
    gold_positions: &[GridPos],
    agent:          &AgentState,
    grid:           &Grid,
) {
    debug_assert_eq!(mm.len(), MM_CHANNELS * MM_SIZE * MM_SIZE);

    let mm_plane = MM_SIZE * MM_SIZE;
    let gw = grid.width  as f32;
    let gh = grid.height as f32;

    // Map a world coordinate to a minimap cell index (clamped).
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

    // Enemy positions.
    for other in agents {
        if other.team == agent.team { continue; }
        let (mx, my) = to_mm(other.pos.x, other.pos.y);
        mm[mm_pixel(MM_CH_ENEMY, mx, my)] = 1.0;
    }

    // Gold positions (use pre-computed vec, faster than scanning items).
    for &gpos in gold_positions {
        let (mx, my) = to_mm(gpos.x, gpos.y);
        mm[mm_pixel(MM_CH_GOLD, mx, my)] = 1.0;
    }

    // Suppress: mark own position so the agent knows where it is on the minimap.
    // (optional — leave as 0 for now, agent learns from base_dx/dy instead)
    let _ = items; // kept for future use (e.g. health items on minimap)
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