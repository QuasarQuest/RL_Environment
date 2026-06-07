// src/engine/clusters.rs
//
// Fixed-grid spatial gold "clusters".
//
// The map is partitioned into a fixed REGION_COLS × REGION_ROWS grid of regions
// (3×3 = 9 by default). Each region is a STABLE slot: region k always refers to
// the same physical area of the map, regardless of where the agent is or how the
// gold is distributed. Gold pieces are binned into the region that contains them.
//
// WHY fixed regions (vs the previous distance-sorted union-find):
//   The old scheme sorted clusters by distance-to-agent every tick, so slot 0 was
//   *always* the nearest gold. The optimal policy under that encoding is the
//   trivial "always pick slot 0 = nearest" — slots 1..K carried no decision value,
//   and the policy collapsed to a single greedy action. Worse, slot identity
//   reshuffled every tick (a pile that was slot 1 became slot 0 once the agent
//   moved), so the LSTM could never form a stable "I'm heading to region k"
//   intention and PPO credit assignment couldn't connect a region choice to its
//   delayed payoff.
//
//   With FIXED regions, slot k is a stable spatial address. The per-region
//   features (direction + gold mass) let the policy reason about the real
//   tradeoff: "region 4 is farther but holds 3× the gold — worth the trip." That
//   choice is now learnable because the slot means the same thing across ticks.
//
// Within a chosen region the low-level controller still collects greedily
// (navigate to the region's nearest gold via A*), so the RL policy only makes the
// strategic call — which region, and when to bank — not the micro-pathing.
//
// NOTE: CLUSTER_K here must match CLUSTER_K in rl/action.rs and rl/obs.rs, and
// ACTION_SIZE in rl/action.rs is sized as CLUSTER_K + 4 (base, speed, mult, wait).

use crate::world::coords::GridPos;

/// Spatial region grid dimensions. CLUSTER_K = REGION_COLS * REGION_ROWS.
pub const REGION_COLS: i32 = 3;
pub const REGION_ROWS: i32 = 3;

/// Number of gold region slots exposed to the RL agent.
/// Must match CLUSTER_K in rl/action.rs and rl/obs.rs.
pub const CLUSTER_K: usize = (REGION_COLS * REGION_ROWS) as usize;

// ── GoldCluster ───────────────────────────────────────────────────────────────

#[derive(Clone, Default)]
pub struct GoldCluster {
    pub golds: Vec<GridPos>,
}

impl GoldCluster {
    pub fn count(&self) -> usize { self.golds.len() }

    /// Nearest gold piece in the cluster to `pos` by Chebyshev distance.
    pub fn nearest_gold(&self, pos: GridPos) -> Option<GridPos> {
        self.golds.iter()
            .min_by_key(|&&g| chebyshev(pos, g))
            .copied()
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Index of the fixed region that contains `pos` on a `grid_w × grid_h` map.
/// Regions are laid out row-major: k = ry * REGION_COLS + rx.
#[inline]
pub fn region_of(pos: GridPos, grid_w: i32, grid_h: i32) -> usize {
    let rx = (pos.x * REGION_COLS / grid_w.max(1)).clamp(0, REGION_COLS - 1);
    let ry = (pos.y * REGION_ROWS / grid_h.max(1)).clamp(0, REGION_ROWS - 1);
    (ry * REGION_COLS + rx) as usize
}

/// Bin `gold` into the fixed REGION_COLS × REGION_ROWS grid. Slot k holds every
/// gold piece whose position falls in region k (stable across ticks). Empty
/// regions are `None`.
pub fn find_clusters(gold: &[GridPos], grid_w: i32, grid_h: i32) -> [Option<GoldCluster>; CLUSTER_K] {
    let mut result: [Option<GoldCluster>; CLUSTER_K] = std::array::from_fn(|_| None);
    for &g in gold {
        let k = region_of(g, grid_w, grid_h);
        result[k].get_or_insert_with(GoldCluster::default).golds.push(g);
    }
    result
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline]
pub fn chebyshev(a: GridPos, b: GridPos) -> i32 {
    (a.x - b.x).abs().max((a.y - b.y).abs())
}
