// src/engine/clusters.rs
//
// Adaptive-radius union-find gold clustering.
//
// Algorithm: try radii [3, 5, 8, 12, 20, 50] in ascending order.
// At each radius, merge any two gold pieces within that Chebyshev distance
// into the same cluster.  Stop at the first radius that produces at least
// CLUSTER_K distinct clusters, then return the top-K by gold count.
//
// This ensures that on a sparse map (gold spread out), small radii keep
// clusters separate; on a dense map (gold clumped), larger radii group them.
// Fallback (all gold in one cluster at slot 0) fires when even radius 50
// doesn't produce enough distinct clusters.

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

/// Returns up to CLUSTER_K clusters sorted by gold count (largest first).
/// Empty slots are `None`.
pub fn find_clusters(gold: &[GridPos]) -> [Option<GoldCluster>; CLUSTER_K] {
    // Workaround: [None; CLUSTER_K] requires Copy, use explicit init instead.
    let mut result: [Option<GoldCluster>; CLUSTER_K] = [None, None, None, None];
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

        if groups.len() >= CLUSTER_K {
            let mut clusters: Vec<GoldCluster> = groups.values()
                .map(|indices| GoldCluster {
                    golds: indices.iter().map(|&i| gold[i]).collect(),
                })
                .collect();
            clusters.sort_by(|a, b| b.count().cmp(&a.count()));
            clusters.truncate(CLUSTER_K);
            for (i, c) in clusters.into_iter().enumerate() {
                result[i] = Some(c);
            }
            return result;
        }
    }

    // Fallback: not enough distinct clusters even at max radius — put everything
    // in slot 0 so the agent always has at least one valid navigation target.
    result[0] = Some(GoldCluster { golds: gold.to_vec() });
    result
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline]
pub fn chebyshev(a: GridPos, b: GridPos) -> i32 {
    (a.x - b.x).abs().max((a.y - b.y).abs())
}
