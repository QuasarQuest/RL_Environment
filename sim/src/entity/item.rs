use crate::world::coords::GridPos;

/// Pickups that appear on the map. Single-agent gold rush only — no combat
/// items (health/ammo). Gold is the objective; the rest are temporary buffs the
/// policy can choose to detour for, plus one hazard the navigator routes around.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ItemKind {
    /// The objective — carry to base to score.
    Gold,
    /// Speed boost, three rarity tiers (rarer = longer 2× window). All give the
    /// same 2× move speed; only the buff duration differs (see config buff consts).
    Speed1,
    Speed2,
    Speed3,
    /// Score multiplier — picking one up holds a single charge that doubles the
    /// next deposit (consumed on that deposit). See engine/pickup + physics.
    Multiplier,
    /// Trap hazard — stepping on it immobilises the agent (0 move speed) for a long
    /// window (TRAP_TICKS). Spawned in 3-tile clusters; the navigator treats trap
    /// tiles as impassable, so the agent routes around them.
    Trap,
}

impl ItemKind {
    /// True for the beneficial speed buffs the policy may deliberately navigate toward.
    pub fn is_speed(self) -> bool {
        matches!(self, ItemKind::Speed1 | ItemKind::Speed2 | ItemKind::Speed3)
    }
}

/// How many of a given item kind are allowed on the map simultaneously.
#[derive(Debug, Clone, Copy)]
pub struct ItemSpawnConfig {
    pub kind:       ItemKind,
    pub max_on_map: usize,
}

#[derive(Clone)]
pub struct ItemState {
    pub pos:  GridPos,
    pub kind: ItemKind,
}
