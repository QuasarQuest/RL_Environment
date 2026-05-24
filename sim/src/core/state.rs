// src/sim_core/state.rs
//
// Plain data types for agents and items. No logic here.

use crate::item::ItemKind;
use crate::world::coords::GridPos;

#[derive(Clone)]
pub struct ItemState {
    pub pos:  GridPos,
    pub kind: ItemKind,
}

#[derive(Clone)]
pub struct AgentState {
    pub pos:          GridPos,
    pub team:         u8,
    pub gold_carried: u8,
    pub score:        u32,
    pub hearts:       u8,
    pub ammo:         u8,
    pub speed_buff:   u8,
    pub spawn_pos:    GridPos,
    pub base_pos:     GridPos,
}