// src/item/mod.rs

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ItemKind {
    Gold,
    Health,
    Ammo,
    SpeedBoost,
}

/// How many of a given item kind are allowed on the map simultaneously.
#[derive(Debug, Clone, Copy)]
pub struct ItemSpawnConfig {
    pub kind:       ItemKind,
    pub max_on_map: usize,
}
