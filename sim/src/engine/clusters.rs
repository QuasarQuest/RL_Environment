// src/engine/clusters.rs
//
// Adaptive-radius union-find gold clustering.
//
// Algorithm: try radii [3, 5, 8, 12, 20, 50] in ascending order. At each radius,
// merge any two gold pieces within that Chebyshev distance into the same cluster.
//
// Cluster count is monotonically NON-INCREASING as the radius grows (a larger
// radius can only merge more pairs, never split them). We therefore stop at the
// first radius that collapses the gold into AT MOST CLUSTER_K clusters — the
// tightest grouping in which every piece still fits into one of the K slots, so
// no gold is ever dropped from the observation. The resulting clusters are then
// sorted by distance from the agent (nearest first).
//
// NOTE on the previous bug: the stop condition used to be `groups.len() >= K`
// combined with `truncate(K)` *before* sorting. Because count is non-increasing
// in radius, `>= K` always returned at radius 3 (or fell through to the
// fallback), making radii 5..50 dead code; and `truncate` then kept an arbitrary
// K clusters in HashMap iteration order (randomly seeded per-instance), so the
// retained set — and thus cluster IDs — reshuffled every tick. The cluster
// feature was effectively noise. `<= K` fixes both: the radius adapts, and since
// we keep ≤ K clusters and sort all of them, the result is deterministic and the
// IDs map to stable spatial regions.
//
// Sorting by distance-to-agent makes Cluster 0 always the nearest cluster.
// The policy can learn the simple rule "Nav Cluster 0 = go to nearest gold"
// without needing to understand the spatial meaning of slots 1-3. Slots 1-3
// serve as ordered fallbacks when the nearest cluster depletes.
//
// Because radius 50 >= the maximum Chebyshev distance on a 50x50 grid (49), the
// loop always succeeds for any non-empty gold set; the single-cluster fallback
// is retained only as defensive code.

use std::collections::HashMap;

use crate::world::coords::GridPos;

/// Number of gold cluster slots exposed to the RL agent.
/// Must match CLUSTER_K in rl/action.rs and rl/obs.rs.
pub const CLUSTER_K: usize = 4;

const RADII: &[i32] = &[3, 5, 8, 12, 20, 50];

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

    /// Mean position of all gold pieces in the cluster.
    pub fn centroid(&self) -> (f32, f32) {
        let n = self.golds.len() as f32;
        let cx = self.golds.iter().map(|g| g.x as f32).sum::<f32>() / n;
        let cy = self.golds.iter().map(|g| g.y as f32).sum::<f32>() / n;
        (cx, cy)
    }
}

// ── Union-Find ────────────────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank:   Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self { parent: (0..n).collect(), rank: vec![0; n] }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry { return; }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less    => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal   => { self.parent[ry] = rx; self.rank[rx] += 1; }
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Returns up to CLUSTER_K clusters sorted by distance from `agent_pos`
/// (nearest cluster first). Distance is measured as the Chebyshev distance
/// from `agent_pos` to the cluster's nearest gold piece. Ties are broken by
/// gold count (more gold first), then centroid X for full determinism.
/// Empty slots are `None`.
pub fn find_clusters(gold: &[GridPos], agent_pos: GridPos) -> [Option<GoldCluster>; CLUSTER_K] {
    let mut result: [Option<GoldCluster>; CLUSTER_K] = std::array::from_fn(|_| None);
    if gold.is_empty() { return result; }

    let n = gold.len();

    for &radius in RADII {
        let mut uf = UnionFind::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                if chebyshev(gold[i], gold[j]) <= radius {
                    uf.union(i, j);
                }
            }
        }

        // Group gold indices by their cluster root.
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            groups.entry(uf.find(i)).or_default().push(i);
        }

        // First (tightest) radius that fits every piece into <= CLUSTER_K slots.
        if groups.len() <= CLUSTER_K {
            let mut clusters: Vec<(i32, usize, f32, GoldCluster)> = groups
                .into_values()
                .map(|indices| {
                    let c = GoldCluster {
                        golds: indices.iter().map(|&i| gold[i]).collect(),
                    };
                    let dist = c.nearest_gold(agent_pos)
                        .map(|g| chebyshev(agent_pos, g))
                        .unwrap_or(i32::MAX);
                    let count = c.count();
                    let (cx, _) = c.centroid();
                    (dist, count, cx, c)
                })
                .collect();

            // Sort: nearest first, then most gold, then leftmost centroid for
            // full determinism regardless of HashMap iteration order.
            clusters.sort_by(|a, b| {
                a.0.cmp(&b.0)
                    .then(b.1.cmp(&a.1))
                    .then(a.2.total_cmp(&b.2))
            });

            for (slot, (_, _, _, c)) in clusters.into_iter().enumerate() {
                result[slot] = Some(c);
            }
            return result;
        }
    }

    // Unreachable for any non-empty gold set (radius 50 >= max grid distance),
    // kept as defensive fallback so the agent always has one valid target.
    result[0] = Some(GoldCluster { golds: gold.to_vec() });
    result
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline]
pub fn chebyshev(a: GridPos, b: GridPos) -> i32 {
    (a.x - b.x).abs().max((a.y - b.y).abs())
}