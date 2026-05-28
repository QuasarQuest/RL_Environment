// src/engine/pickup.rs
//
// Item pickup logic — applied after movement each tick.

use crate::config;
use crate::entity::item::ItemKind;
use crate::entity::{AgentState, ItemState};

pub fn pickup(agents: &mut Vec<AgentState>, items: &mut Vec<ItemState>) {
    for agent_idx in 0..agents.len() {
        if agents[agent_idx].respawn_timer > 0 { continue; }
        let pos = agents[agent_idx].pos;
        let mut item_idx = 0;
        while item_idx < items.len() {
            if items[item_idx].pos != pos { item_idx += 1; continue; }
            let kind = items[item_idx].kind;
            let picked = {
                let a = &mut agents[agent_idx];
                match kind {
                    ItemKind::Gold => {
                        if a.gold_carried < config::AGENT_MAX_GOLD {
                            a.gold_carried += 1; true
                        } else { false }
                    }
                    ItemKind::Health => {
                        if a.hearts < config::AGENT_MAX_HEARTS {
                            a.hearts += 1; true
                        } else { false }
                    }
                    ItemKind::Ammo => {
                        if a.ammo < config::AGENT_MAX_AMMO {
                            a.ammo = (a.ammo + config::AMMO_PER_PICKUP).min(config::AGENT_MAX_AMMO);
                            true
                        } else { false }
                    }
                    ItemKind::SpeedBoost => {
                        // Extend, not reset: never cut short a buff already running longer.
                        a.speed_buff = a.speed_buff.max(config::SPEED_BUFF_TICKS);
                        true
                    }
                }
            };
            if picked { items.swap_remove(item_idx); } else { item_idx += 1; }
        }
    }
}
