// src/algorithm/path_planning/graph_utils.rs

use crate::entity::agent::Dir;
use crate::world::coords::GridPos;

// ── Cost constants ────────────────────────────────────────────────────────────
// Cardinal moves cost 10, diagonals cost 14 (~10*sqrt(2)).
// This integer approximation gives algorithms a strong preference for straight paths
// and eliminates zigzag artefacts that appear when all moves cost the same.

pub const CARDINAL: i32 = 10;
pub const DIAGONAL: i32 = 14;

/// Octile distance — the correct heuristic for 8-directional movement
/// with the CARDINAL/DIAGONAL cost model.
///
///   h = CARDINAL * (dx + dy) + (DIAGONAL - 2*CARDINAL) * min(dx, dy)
///     = 10*(dx+dy) - 6*min(dx,dy)
///
/// This is admissible: it never overestimates the true cost.
#[inline]
pub fn octile(from: GridPos, to: GridPos) -> i32 {
    let dx = (to.x - from.x).abs();
    let dy = (to.y - from.y).abs();
    CARDINAL * (dx + dy) + (DIAGONAL - 2 * CARDINAL) * dx.min(dy)
}

/// Returns the `Dir` that moves from `from` to `to` in one step, or `None`
/// if the two positions are not exactly one step apart in any of the 8 directions.
///
/// Extracted here so `AStarAgent` and `DStarAgent` share one implementation.
#[inline]
pub fn dir_to(from: GridPos, to: GridPos) -> Option<Dir> {
    let delta = (to.x - from.x, to.y - from.y);
    Dir::all().iter().find(|d| d.delta() == delta).copied()
}