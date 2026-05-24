// src/sim_core/items.rs
//
// Item pickup. Gold respawns mid-episode via SimCore::respawn_gold() when the
// count drops below gold_respawn_min — see sim_core/mod.rs.

use crate::config;
use crate::item::ItemKind;
use super::state::{AgentState, ItemState};

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
                        a.speed_buff = config::SPEED_BUFF_TICKS;
                        true
                    }
                }
            };
            if picked { items.swap_remove(item_idx); } else { item_idx += 1; }
        }
    }
}
