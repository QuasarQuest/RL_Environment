// src/engine/pickup.rs
//
// Item pickup logic — applied after movement each tick.

use crate::config::{AGENT_MAX_GOLD, MULT_CHARGE_MAX, SPEED_TICKS};
use crate::entity::item::ItemKind;
use crate::entity::{AgentState, ItemState};

pub fn pickup(agents: &mut [AgentState], items: &mut Vec<ItemState>) {
    for a in agents.iter_mut() {
        let mut item_idx = 0;
        while item_idx < items.len() {
            if items[item_idx].pos != a.pos { item_idx += 1; continue; }
            let picked = match items[item_idx].kind {
                ItemKind::Gold => {
                    if a.gold_carried < AGENT_MAX_GOLD {
                        a.gold_carried += 1; true
                    } else { false }
                }
                // Buffs extend rather than reset: never cut short a window already
                // running longer than the one just picked up.
                ItemKind::Speed => { a.speed_buff = a.speed_buff.max(SPEED_TICKS); true }
                // Hold a single multiplier charge — consumed on the next deposit.
                // Leave a second one on the map (don't pick up) when already at
                // capacity, so it isn't wasted.
                ItemKind::Multiplier => {
                    if a.mult_charge < MULT_CHARGE_MAX {
                        a.mult_charge = MULT_CHARGE_MAX;
                        true
                    } else {
                        false
                    }
                }
            };
            if picked { items.swap_remove(item_idx); } else { item_idx += 1; }
        }
    }
}
