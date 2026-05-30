// src/rl/action.rs
//
// High-level action space for the RL agent.
// The agent picks a navigation goal or a direct action; A* handles movement.
//
// Layout:
//   0..8   NavigateToCluster(0..8)  — fixed 3×3 map regions (stable spatial slots)
//   9      NavigateToBase           — return to base and deposit
//   10     NavigateToHealth         — nearest health pickup
//   11     NavigateToAmmo           — nearest ammo pickup
//   12     NavigateToEnemy          — nearest enemy (for hunting / interception)
//   13     MeleeAttack              — auto-target lowest-HP enemy in melee range
//   14     RangedAttack             — auto-target lowest-HP enemy in ranged range
//   15     Wait
//
// CLUSTER_K must match engine/clusters.rs and rl/obs.rs.
// ACTION_SIZE = CLUSTER_K + 7 (Base, Health, Ammo, Enemy, Melee, Ranged, Wait).

pub const CLUSTER_K:   usize = 9;
pub const ACTION_SIZE: usize = CLUSTER_K + 7; // 16
pub const ACTION_WAIT: u32   = (ACTION_SIZE - 1) as u32; // 15

/// High-level action the RL policy selects each tick.
/// Navigation goals are resolved to a concrete GridPos target and executed via A*.
/// Direct actions (attack, wait) are applied immediately without pathfinding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RlAction {
    NavigateToCluster(u8),
    NavigateToBase,
    NavigateToHealth,
    NavigateToAmmo,
    NavigateToEnemy,
    MeleeAttack,
    RangedAttack,
    Wait,
}

impl std::fmt::Display for RlAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NavigateToCluster(k) => write!(f, "Nav Cluster {k}"),
            Self::NavigateToBase       => write!(f, "Nav Base"),
            Self::NavigateToHealth     => write!(f, "Nav Health"),
            Self::NavigateToAmmo       => write!(f, "Nav Ammo"),
            Self::NavigateToEnemy      => write!(f, "Nav Enemy"),
            Self::MeleeAttack          => write!(f, "Melee Atk"),
            Self::RangedAttack         => write!(f, "Ranged Atk"),
            Self::Wait                 => write!(f, "Wait"),
        }
    }
}

/// Convert a neural-net output integer to an RlAction.
/// Panics on out-of-range — Python side must clamp to 0..ACTION_SIZE.
///
/// Indices 0..CLUSTER_K are the fixed region-navigation slots; the remaining
/// seven are the direct/nav actions, defined relative to CLUSTER_K so the layout
/// stays correct if the region grid size changes.
pub fn int_to_rl_action(action: u32) -> RlAction {
    let k = CLUSTER_K as u32;
    match action {
        a if a < k       => RlAction::NavigateToCluster(a as u8),
        a if a == k      => RlAction::NavigateToBase,
        a if a == k + 1  => RlAction::NavigateToHealth,
        a if a == k + 2  => RlAction::NavigateToAmmo,
        a if a == k + 3  => RlAction::NavigateToEnemy,
        a if a == k + 4  => RlAction::MeleeAttack,
        a if a == k + 5  => RlAction::RangedAttack,
        a if a == k + 6  => RlAction::Wait,
        _ => panic!("Invalid RL action index: {action} (must be 0..{ACTION_SIZE})"),
    }
}
