// src/rl/grid.rs
//
// Spatial (CNN-shaped) observation builder for the RL agent.
//
// Layout: flat Vec<f32> of length OBS_TOTAL, indexed in CHW order so that
// reshape to (C, H, W) on the Python side is a zero-cost view that matches
// PyTorch's default tensor layout.
//
// Index formula:
//   buf[c * H * W + y * W + x]
// where (x, y) are the crop-local coordinates, origin at top-left of the
// crop, x growing right, y growing down. The agent sits at the centre
// (CENTRE, CENTRE).
//
// Channels (0..OBS_CHANNELS):
//   0  Out-of-bounds mask  (1.0 where the crop falls outside the map)
//   1  Own base            (1.0 on Tile::Base(team) where team == agent.team)
//   2  Gold item           (1.0 where a Gold item entity sits)
//   3  Self at centre      (single 1.0 at (CENTRE, CENTRE), else 0.0)
//   4  Carrying gold       (entire plane = 1.0 if agent.gold > 0 else 0.0)
//
// Borrow discipline mirrors obs.rs: copy resource values into locals before
// running any query, and fully drain each query into owned data before the
// next one.

use bevy::prelude::*;
use crate::agent::components::GoldCarried;
use crate::item::{Item, ItemKind};
use crate::team::Team;
use crate::world::coords::GridPos;
use crate::world::tile::Tile;
use crate::world::Grid;
use super::marker::RlAgent;

// ── Public constants ──────────────────────────────────────────────────────────

pub const OBS_CHANNELS:  usize = 5;
pub const OBS_CROP_SIZE: usize = 25;
pub const OBS_TOTAL:     usize = OBS_CHANNELS * OBS_CROP_SIZE * OBS_CROP_SIZE;

/// Tuple form for Python — (C, H, W).
pub const OBS_SHAPE: (usize, usize, usize) =
    (OBS_CHANNELS, OBS_CROP_SIZE, OBS_CROP_SIZE);

// Half-width of the crop (radius from centre).
// For a 25-tile crop, CENTRE = 12, RADIUS = 12.
const CENTRE: i32 = (OBS_CROP_SIZE / 2) as i32;
const RADIUS: i32 = CENTRE;

// Channel indices — kept as named constants so the build code reads cleanly.
const CH_OOB:          usize = 0;
const CH_OWN_BASE:     usize = 1;
const CH_GOLD:         usize = 2;
const CH_SELF:         usize = 3;
const CH_CARRY:        usize = 4;

// ── Snapshots ─────────────────────────────────────────────────────────────────

struct AgentSnap {
    pos:  GridPos,
    team: u8,
    gold: u8,
}

// ── Public entry point ────────────────────────────────────────────────────────

pub fn build_obs_grid(world: &mut World) -> Vec<f32> {
    let mut obs = vec![0.0f32; OBS_TOTAL];

    // ── 1. Locate the RL agent ────────────────────────────────────────────
    let agent: Option<AgentSnap> = {
        let mut q = world.query_filtered::<
            (&GridPos, &Team, &GoldCarried),
            With<RlAgent>,
        >();
        q.single(world).ok().map(|(pos, team, gold)| AgentSnap {
            pos:  *pos,
            team: team.0,
            gold: gold.0,
        })
    };
    let Some(agent) = agent else { return obs };

    // ── 2. Channel 4 (carrying-gold) — broadcast plane ────────────────────
    // Cheaper than per-cell branching: set once, no per-tile work.
    if agent.gold > 0 {
        let start = CH_CARRY * OBS_CROP_SIZE * OBS_CROP_SIZE;
        let end   = start + OBS_CROP_SIZE * OBS_CROP_SIZE;
        for v in &mut obs[start..end] { *v = 1.0; }
    }

    // ── 3. Channel 3 (self at centre) ─────────────────────────────────────
    obs[idx(CH_SELF, CENTRE, CENTRE)] = 1.0;

    // ── 4. Tile-derived channels (OOB, own base) ──────────────────────────
    // We hold Grid via resource() (shared borrow) and never query during this
    // block, so this is safe. After the block, the borrow is dropped.
    {
        let grid       = world.resource::<Grid>();
        let grid_w     = grid.width  as i32;
        let grid_h     = grid.height as i32;
        let own_team   = agent.team;
        let ax         = agent.pos.x;
        let ay         = agent.pos.y;

        for cy in 0..OBS_CROP_SIZE as i32 {
            let wy = ay + (cy - CENTRE);          // world y for this crop row
            for cx in 0..OBS_CROP_SIZE as i32 {
                let wx = ax + (cx - CENTRE);      // world x for this crop col

                if wx < 0 || wx >= grid_w || wy < 0 || wy >= grid_h {
                    obs[idx(CH_OOB, cx, cy)] = 1.0;
                    continue;
                }

                // Safe to unwrap: we just bounds-checked above.
                if let Some(Tile::Base(t)) = grid.get(wx, wy) {
                    if t == own_team {
                        obs[idx(CH_OWN_BASE, cx, cy)] = 1.0;
                    }
                }
            }
        }
    }

    // ── 5. Gold items ─────────────────────────────────────────────────────
    // Collect first to avoid holding the query borrow during writes.
    let gold_positions: Vec<GridPos> = {
        let mut q = world.query::<(&GridPos, &Item)>();
        q.iter(world)
            .filter(|(_, item)| item.kind == ItemKind::Gold)
            .map(|(pos, _)| *pos)
            .collect()
    };

    for gp in gold_positions {
        let cx = gp.x - agent.pos.x + CENTRE;
        let cy = gp.y - agent.pos.y + CENTRE;
        if cx >= 0 && cx < OBS_CROP_SIZE as i32
            && cy >= 0 && cy < OBS_CROP_SIZE as i32
        {
            obs[idx(CH_GOLD, cx, cy)] = 1.0;
        }
    }

    obs
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Convert (channel, crop_x, crop_y) into the flat CHW index.
/// Caller must ensure cx, cy ∈ [0, OBS_CROP_SIZE).
#[inline]
fn idx(ch: usize, cx: i32, cy: i32) -> usize {
    debug_assert!(cx >= 0 && (cx as usize) < OBS_CROP_SIZE);
    debug_assert!(cy >= 0 && (cy as usize) < OBS_CROP_SIZE);
    ch * OBS_CROP_SIZE * OBS_CROP_SIZE
        + (cy as usize) * OBS_CROP_SIZE
        + (cx as usize)
}

// Discard unused imports under cfg analysis — RADIUS is documentation,
// keep it referenced so it isn't dead-code-warned.
const _: i32 = RADIUS;