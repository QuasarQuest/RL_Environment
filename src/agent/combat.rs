// src/agent/combat.rs

use bevy::prelude::*;
use crate::world::Grid;
use crate::world::tile::Tile;
use crate::world::coords::GridPos;
use crate::item::{ItemBundle, ItemKind};
use crate::viz::grid_offset::GridOffset;
use crate::config;
use super::action::Action;
use super::components::{GoldCarried, Health};
use super::systems::PendingAction;

pub const ATTACK_DAMAGE: u32 = 10;

#[derive(Component)]
pub struct Dead;

pub fn resolve_combat(
    mut commands:  Commands,
    grid:          Res<Grid>,
    offset:        Res<GridOffset>,
    attackers:     Query<(Entity, &GridPos, &PendingAction)>,
    mut defenders: Query<(Entity, &GridPos, &mut Health, &mut GoldCarried), Without<Dead>>,
) {
    let defender_map: std::collections::HashMap<GridPos, Entity> = defenders
        .iter()
        .map(|(e, pos, _, _)| (*pos, e))
        .collect();

    for (_attacker, pos, pending) in attackers.iter() {
        let Some(Action::Attack(dir)) = pending.0 else { continue };
        let target_pos = pos.apply_delta(dir.delta().0, dir.delta().1);

        if let Some(&defender_entity) = defender_map.get(&target_pos) {
            if let Ok((entity, def_pos, mut health, mut gold)) =
                defenders.get_mut(defender_entity)
            {
                if health.0 <= ATTACK_DAMAGE {
                    // Drop carried gold as item entities on the tile.
                    if gold.0 > 0 && grid.get(def_pos.x, def_pos.y) != Some(Tile::Obstacle) {
                        for _ in 0..gold.0 {
                            let world_pos = offset.world_pos(def_pos.x, def_pos.y);
                            commands.spawn(ItemBundle::new(
                                ItemKind::Gold, *def_pos, config::TILE_SIZE, world_pos,
                            ));
                        }
                        gold.0 = 0;
                    }
                    commands.entity(entity).insert(Dead);
                    info!("Agent {:?} killed at {:?}", entity, def_pos);
                } else {
                    health.0 -= ATTACK_DAMAGE;
                    info!("Agent {:?} hit for {ATTACK_DAMAGE} — {} hp remaining",
                        entity, health.0);
                }
            }
        }
    }
}

pub fn despawn_dead(
    mut commands: Commands,
    query:        Query<Entity, With<Dead>>,
) {
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }
}