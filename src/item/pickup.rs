// src/item/pickup.rs
//
// Pickup system — runs on OnSimTick after agent movement.
// Agents standing on an item entity pick it up automatically.
// Consumption logic (what the item does) lives here per ItemKind.

use bevy::prelude::*;
use crate::world::coords::GridPos;
use crate::agent::components::{GoldCarried, Score};
use crate::team::{Team, TeamScore};
use super::{Item, ItemKind};

/// Marker for items that have been claimed this tick — prevents double pickup.
#[derive(Component)]
pub struct Claimed;

pub fn pickup_items(
    mut commands:  Commands,
    mut agents:    Query<(&GridPos, &mut GoldCarried, &mut Score, &Team)>,
    items:         Query<(Entity, &GridPos, &Item), Without<Claimed>>,
    mut team_score: ResMut<TeamScore>,
) {
    for (item_entity, item_pos, item) in items.iter() {
        // Find an agent standing on this tile.
        for (agent_pos, mut gold, mut score, team) in agents.iter_mut() {
            if agent_pos != item_pos { continue; }

            match item.kind {
                ItemKind::Gold => {
                    if !gold.is_full() {
                        gold.0 += 1;
                        commands.entity(item_entity).insert(Claimed);
                        info!("Team {} picked up Gold ({} carried)", team.name(), gold.0);
                    }
                }
                // Future items handled here:
                // ItemKind::Health     => { health.0 = (health.0 + 30).min(MAX_HEALTH); ... }
                // ItemKind::SpeedBoost => { commands.entity(agent_entity).insert(SpeedBuff); ... }
            }
            break; // one agent per item per tick
        }
    }
}

/// Despawn all claimed items at end of tick.
pub fn despawn_claimed(
    mut commands: Commands,
    query:        Query<Entity, With<Claimed>>,
) {
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }
}

/// Sync item entity transforms when GridPos changes (e.g. after spawn).
pub fn sync_item_transforms(
    offset: Res<crate::viz::grid_offset::GridOffset>,
    mut query: Query<(&GridPos, &mut Transform), With<Item>>,
) {
    for (pos, mut tf) in query.iter_mut() {
        let wp = offset.world_pos(pos.x, pos.y);
        tf.translation = Vec3::new(wp.x, wp.y, super::ItemKind::Gold.z_layer());
    }
}