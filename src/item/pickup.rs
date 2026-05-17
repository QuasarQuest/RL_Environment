// src/item/pickup.rs

use bevy::prelude::*;
use crate::world::coords::GridPos;
use crate::world::grid::Grid;
use crate::agent::components::{Ammo, GoldCarried, Hearts, RespawnIn, Score, SpeedBuff};
use crate::team::{Team, TeamScore};
use crate::config;
use super::{Item, ItemKind};

#[derive(Component)]
pub struct Claimed;

pub fn pickup_items(
    mut commands: Commands,
    // Dead agents (RespawnIn) can't pick up items.
    mut agents:   Query<(
        Entity, &GridPos, &mut GoldCarried, &mut Hearts, &mut Ammo, &Team,
    ), Without<RespawnIn>>,
    items:        Query<(Entity, &GridPos, &Item), Without<Claimed>>,
) {
    for (item_entity, item_pos, item) in items.iter() {
        for (agent_entity, agent_pos, mut gold, mut hearts, mut ammo, team)
        in agents.iter_mut()
        {
            if agent_pos != item_pos { continue; }

            match item.kind {
                ItemKind::Gold => {
                    if !gold.is_full() {
                        gold.0 += 1;
                        commands.entity(item_entity).insert(Claimed);
                        info!("Team {} picked up Gold ({} carried)", team.name(), gold.0);
                    }
                }
                ItemKind::Health => {
                    if !hearts.is_full() {
                        hearts.gain_one();
                        commands.entity(item_entity).insert(Claimed);
                        info!("Team {} picked up Health ({} hearts)", team.name(), hearts.0);
                    }
                }
                ItemKind::Ammo => {
                    if !ammo.is_full() {
                        ammo.add(config::AMMO_PER_PICKUP);
                        commands.entity(item_entity).insert(Claimed);
                        info!("Team {} picked up Ammo ({} total)", team.name(), ammo.0);
                    }
                }
                ItemKind::SpeedBoost => {
                    // Always pick up — refreshes or extends existing buff.
                    commands.entity(agent_entity)
                        .insert(SpeedBuff(config::SPEED_BUFF_TICKS));
                    commands.entity(item_entity).insert(Claimed);
                    info!("Team {} picked up SpeedBoost", team.name());
                }
            }
            break; // one agent per item per tick
        }
    }
}

/// Agent standing on their own base tile with gold → deposit all carried gold.
/// Each deposited gold unit scores 1 point for the agent and their team.
pub fn deposit_gold(
    grid:           Res<Grid>,
    mut team_score: ResMut<TeamScore>,
    mut agents:     Query<(&GridPos, &mut GoldCarried, &mut Score, &Team), Without<RespawnIn>>,
) {
    for (pos, mut gold, mut score, team) in agents.iter_mut() {
        if gold.is_empty() { continue; }
        if grid.get(pos.x, pos.y) != Some(crate::world::tile::Tile::Base(team.0)) { continue; }

        let deposited = gold.0 as u32;
        gold.0        = 0;
        score.0      += deposited;
        team_score.add(*team, deposited);
        info!("Team {} deposited {} gold (agent score: {}, team score: {})",
            team.name(), deposited, score.0, team_score.get(*team));
    }
}

pub fn despawn_claimed(
    mut commands: Commands,
    query:        Query<Entity, With<Claimed>>,
) {
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }
}

pub fn sync_item_transforms(
    offset:    Res<crate::viz::grid_offset::GridOffset>,
    mut query: Query<(&GridPos, &mut Transform), With<Item>>,
) {
    for (pos, mut tf) in query.iter_mut() {
        let wp = offset.world_pos(pos.x, pos.y);
        tf.translation = Vec3::new(wp.x, wp.y, 0.5);
    }
}