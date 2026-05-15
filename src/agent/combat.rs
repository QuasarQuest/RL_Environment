// src/agent/combat.rs

use bevy::prelude::*;
use crate::world::Grid;
use crate::world::coords::GridPos;
use crate::item::{ItemBundle, ItemKind};
use crate::viz::grid_offset::GridOffset;
use crate::config;
use super::action::Action;
use super::components::{Ammo, GoldCarried, Hearts, RespawnIn, SpawnPoint};
use super::systems::PendingAction;

// ── Resolve combat ────────────────────────────────────────────────────────────

pub fn resolve_combat(
    mut commands: Commands,
    grid:         Res<Grid>,
    offset:       Res<GridOffset>,
    attackers:    Query<(Entity, &GridPos, &PendingAction, &mut Ammo)>,
    mut targets:  Query<(Entity, &GridPos, &mut Hearts, &mut GoldCarried, &SpawnPoint)>,
) {
    // Build target map: pos → entity.
    let target_map: std::collections::HashMap<GridPos, Entity> = targets
        .iter()
        .map(|(e, pos, _, _, _)| (*pos, e))
        .collect();

    for (attacker_entity, attacker_pos, pending, mut ammo) in
    // Safety: attacker and target are always different entities.
    // We iterate attackers, then look up targets by position.
        unsafe { attackers.iter_unsafe() }
    {
        let action = match pending.0 {
            Some(Action::Attack(dir))       => Some((dir, false)),
            Some(Action::RangedAttack(dir)) => Some((dir, true)),
            _                               => None,
        };
        let Some((dir, is_ranged)) = action else { continue };

        // Ranged requires ammo — consume or skip.
        if is_ranged && !ammo.consume() { continue; }

        let (dx, dy) = dir.delta();

        let range = if is_ranged { config::RANGED_RANGE } else { config::MELEE_RANGE };

        // Walk along the direction up to `range` tiles, stop at first hit.
        let mut hit_entity: Option<Entity> = None;
        for dist in 1..=range {
            let check = GridPos::new(
                attacker_pos.x + dx * dist,
                attacker_pos.y + dy * dist,
            );
            // Ranged shots are blocked by obstacles.
            if is_ranged {
                use crate::world::tile::Tile;
                if grid.get(check.x, check.y) == Some(Tile::Obstacle) { break; }
            }
            if let Some(&entity) = target_map.get(&check) {
                // Don't hit self.
                if entity != attacker_entity { hit_entity = Some(entity); break; }
            }
        }

        let Some(target_entity) = hit_entity else { continue };
        let Ok((_, def_pos, mut hearts, mut gold, spawn)) =
            targets.get_mut(target_entity) else { continue };

        hearts.lose_one();
        info!(
            "{} hit {:?} at {:?} — {} heart(s) remaining",
            if is_ranged { "Ranged" } else { "Melee" },
            target_entity, def_pos, hearts.0
        );

        if hearts.is_dead() {
            // Drop carried gold on death tile.
            if gold.0 > 0 {
                let world_pos = offset.world_pos(def_pos.x, def_pos.y);
                for _ in 0..gold.0 {
                    commands.spawn(ItemBundle::new(
                        ItemKind::Gold, *def_pos, config::TILE_SIZE, world_pos,
                    ));
                }
                gold.0 = 0;
            }
            // Queue respawn instead of despawning — permanent death collapses episodes.
            commands.entity(target_entity).insert(RespawnIn(config::AGENT_RESPAWN_TICKS));
            info!("Agent {:?} died — respawning at {:?} in {} ticks",
                target_entity, spawn.0, config::AGENT_RESPAWN_TICKS);
        }
    }
}

// ── Respawn ───────────────────────────────────────────────────────────────────

pub fn tick_respawn(
    mut commands: Commands,
    mut query: Query<(
        Entity, &mut RespawnIn, &mut GridPos, &mut Hearts,
        &mut Ammo, &SpawnPoint,
    )>,
) {
    for (entity, mut timer, mut pos, mut hearts, mut ammo, spawn) in query.iter_mut() {
        if timer.0 > 0 {
            timer.0 -= 1;
            continue;
        }
        // Timer expired — revive.
        *pos   = spawn.0;
        *hearts = Hearts::default();
        *ammo   = Ammo::default();
        commands.entity(entity)
            .remove::<RespawnIn>()
            .remove::<super::components::SpeedBuff>();
        info!("Agent {:?} respawned at {:?}", entity, spawn.0);
    }
}