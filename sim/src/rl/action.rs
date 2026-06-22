// src/rl/action.rs
//
// High-level action space for the RL agent — single-agent gold rush.
// The agent picks a navigation goal; A* (engine/nav.rs) handles the movement.
// There is no combat. Besides gold regions and the base, the policy can also
// detour to the nearest speed boost or score multiplier.
//
// Layout:
//   0..K     NavigateToCluster(0..K)  — fixed 3×3 map regions (stable spatial slots)
//   K        NavigateToBase           — return to base and deposit
//   K+1      NavigateToSpeed          — nearest speed-boost item
//   K+2      NavigateToMultiplier     — nearest score-multiplier item
//   K+3      Wait
//
// CLUSTER_K must match engine/clusters.rs and rl/obs.rs.
// ACTION_SIZE = CLUSTER_K + 4.
//
// There is intentionally no "nearest gold" macro-action: the agent must reach gold
// through the fixed-region slots (reading the per-region obs features), so
// nearest-gold-seeking is an emergent, learned behaviour rather than a hardcoded
// greedy primitive. With an event-only reward (every gold worth the same), a single
// "go to closest gold" action is reward-optimal and strictly dominates every region
// slot, collapsing the policy onto it — removing it forces real spatial decisions.

pub const CLUSTER_K:   usize = 9;
pub const ACTION_SIZE: usize = CLUSTER_K + 4; // 13

// Index helpers (kept relative to CLUSTER_K so the layout stays correct if the
// region grid size changes).
pub const ACTION_BASE:    u32 = CLUSTER_K as u32;       // 9
pub const ACTION_SPEED:   u32 = CLUSTER_K as u32 + 1;   // 10
pub const ACTION_MULT:    u32 = CLUSTER_K as u32 + 2;   // 11
pub const ACTION_WAIT:    u32 = CLUSTER_K as u32 + 3;   // 12

/// High-level action the RL policy selects each decision point.
/// Navigation goals are resolved to a concrete GridPos target and executed via A*
/// (run to completion as a temporally-extended option — see SimCore::step_option).
/// `Wait` is applied immediately for a single tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RlAction {
    NavigateToCluster(u8),
    NavigateToBase,
    NavigateToSpeed,
    NavigateToMultiplier,
    Wait,
}

impl RlAction {
    /// True for the navigation goals executed via A* as temporally-extended
    /// options (see `SimCore::step_option`). False for `Wait` (single tick).
    pub fn is_navigation(self) -> bool {
        !matches!(self, Self::Wait)
    }
}

impl std::fmt::Display for RlAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NavigateToCluster(k)  => write!(f, "Nav Cluster {k}"),
            Self::NavigateToBase        => write!(f, "Nav Base"),
            Self::NavigateToSpeed       => write!(f, "Nav Speed"),
            Self::NavigateToMultiplier  => write!(f, "Nav Multiplier"),
            Self::Wait                  => write!(f, "Wait"),
        }
    }
}

/// Convert a neural-net output integer to an RlAction.
/// Panics on out-of-range — Python side must clamp to 0..ACTION_SIZE.
pub fn int_to_rl_action(action: u32) -> RlAction {
    match action {
        a if a < CLUSTER_K as u32 => RlAction::NavigateToCluster(a as u8),
        a if a == ACTION_BASE     => RlAction::NavigateToBase,
        a if a == ACTION_SPEED    => RlAction::NavigateToSpeed,
        a if a == ACTION_MULT     => RlAction::NavigateToMultiplier,
        a if a == ACTION_WAIT     => RlAction::Wait,
        _ => panic!("Invalid RL action index: {action} (must be 0..{ACTION_SIZE})"),
    }
}
