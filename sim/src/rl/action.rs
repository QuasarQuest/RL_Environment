// src/rl/action.rs
//
// High-level action space for the RL agent — single-agent gold rush only.
// The agent picks a navigation goal; A* handles the movement. There is no
// combat: enemies/ammo/health were removed from the agent's world (see
// reward.rs / obs.rs). Combat lives in a separate future game mode.
//
// Layout:
//   0..8   NavigateToCluster(0..8)  — fixed 3×3 map regions (stable spatial slots)
//   9      NavigateToBase           — return to base and deposit
//   10     Wait
//
// CLUSTER_K must match engine/clusters.rs and rl/obs.rs.
// ACTION_SIZE = CLUSTER_K + 2 (Base, Wait).

pub const CLUSTER_K:   usize = 9;
pub const ACTION_SIZE: usize = CLUSTER_K + 2; // 11
pub const ACTION_WAIT: u32   = (ACTION_SIZE - 1) as u32; // 10

/// High-level action the RL policy selects each decision point.
/// Navigation goals are resolved to a concrete GridPos target and executed via A*
/// (run to completion as a temporally-extended option — see SimCore::step_option).
/// `Wait` is applied immediately for a single tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RlAction {
    NavigateToCluster(u8),
    NavigateToBase,
    Wait,
}

impl RlAction {
    /// True for the navigation goals (cluster/base) that are executed via A* and
    /// run as temporally-extended options (see `SimCore::step_option`). False for
    /// `Wait`, which is a single-tick action.
    pub fn is_navigation(self) -> bool {
        matches!(self, Self::NavigateToCluster(_) | Self::NavigateToBase)
    }
}

impl std::fmt::Display for RlAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NavigateToCluster(k) => write!(f, "Nav Cluster {k}"),
            Self::NavigateToBase       => write!(f, "Nav Base"),
            Self::Wait                 => write!(f, "Wait"),
        }
    }
}

/// Convert a neural-net output integer to an RlAction.
/// Panics on out-of-range — Python side must clamp to 0..ACTION_SIZE.
///
/// Indices 0..CLUSTER_K are the fixed region-navigation slots; the remaining two
/// are NavigateToBase and Wait, defined relative to CLUSTER_K so the layout stays
/// correct if the region grid size changes.
pub fn int_to_rl_action(action: u32) -> RlAction {
    let k = CLUSTER_K as u32;
    match action {
        a if a < k      => RlAction::NavigateToCluster(a as u8),
        a if a == k     => RlAction::NavigateToBase,
        a if a == k + 1 => RlAction::Wait,
        _ => panic!("Invalid RL action index: {action} (must be 0..{ACTION_SIZE})"),
    }
}
