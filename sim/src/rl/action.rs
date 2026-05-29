// src/rl/action.rs
//
// High-level action space for the RL agent.
// The agent picks a navigation goal or a direct action; A* handles movement.
//
// Layout:
//   0..3   NavigateToCluster(0..3)  — top-4 gold clusters by density, largest first
//   4      NavigateToBase           — return to base and deposit
//   5      NavigateToHealth         — nearest health pickup
//   6      NavigateToAmmo           — nearest ammo pickup
//   7      NavigateToEnemy          — nearest enemy (for hunting / interception)
//   8      MeleeAttack              — auto-target lowest-HP enemy in melee range
//   9      RangedAttack             — auto-target lowest-HP enemy in ranged range
//   10     Wait

pub const ACTION_SIZE: usize = 11;
pub const ACTION_WAIT: u32   = 10;
pub const CLUSTER_K:   usize = 4;

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
pub fn int_to_rl_action(action: u32) -> RlAction {
    match action {
        0  => RlAction::NavigateToCluster(0),
        1  => RlAction::NavigateToCluster(1),
        2  => RlAction::NavigateToCluster(2),
        3  => RlAction::NavigateToCluster(3),
        4  => RlAction::NavigateToBase,
        5  => RlAction::NavigateToHealth,
        6  => RlAction::NavigateToAmmo,
        7  => RlAction::NavigateToEnemy,
        8  => RlAction::MeleeAttack,
        9  => RlAction::RangedAttack,
        10 => RlAction::Wait,
        _  => panic!("Invalid RL action index: {action} (must be 0..{ACTION_SIZE})"),
    }
}
