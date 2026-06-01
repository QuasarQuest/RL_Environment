// src/engine/pickup.rs
//
// Item pickup logic — applied after movement each tick.

use crate::config::{AGENT_MAX_GOLD, MULT_TICKS, SLOW_TICKS, SPEED1_TICKS, SPEED2_TICKS, SPEED3_TICKS};
use crate::entity::item::ItemKind;
use crate::entity::{AgentState, ItemState};

pub fn pickup(agents: &mut Vec<AgentState>, items: &mut Vec<ItemState>) {
    for agent_idx in 0..agents.len() {
        let pos = agents[agent_idx].pos;
        let mut item_idx = 0;
        while item_idx < items.len() {
            if items[item_idx].pos != pos { item_idx += 1; continue; }
            let kind = items[item_idx].kind;
            let picked = {
                let a = &mut agents[agent_idx];
                match kind {
                    ItemKind::Gold => {
                        if a.gold_carried < AGENT_MAX_GOLD {
                            a.gold_carried += 1; true
                        } else { false }
                    }
                    // Buffs extend rather than reset: never cut short a window already
                    // running longer than the one just picked up.
                    ItemKind::Speed1 => { a.speed_buff = a.speed_buff.max(SPEED1_TICKS); true }
                    ItemKind::Speed2 => { a.speed_buff = a.speed_buff.max(SPEED2_TICKS); true }
                    ItemKind::Speed3 => { a.speed_buff = a.speed_buff.max(SPEED3_TICKS); true }
                    ItemKind::Slow       => { a.slow_buff = a.slow_buff.max(SLOW_TICKS); true }
                    ItemKind::Multiplier => { a.mult_buff = a.mult_buff.max(MULT_TICKS); true }
                }
            };
            if picked { items.swap_remove(item_idx); } else { item_idx += 1; }
        }
    }
}
