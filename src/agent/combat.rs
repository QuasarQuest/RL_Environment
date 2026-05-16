// src/agent/combat.rs

use bevy::prelude::*;
use crate::world::Grid;
use crate::world::coords::GridPos;
use crate::item::{ItemBundle, ItemKind};
use crate::viz::grid_offset::GridOffset;
use crate::config;
use super::action::Action;
use super::brain::AgentBrain;
use super::components::{Ammo, GoldCarried, Hearts, RespawnIn, SpawnPoint};
use super::systems::PendingAction;

// ── Resolve combat ────────────────────────────────────────────────────────────
//
// Split into two passes to avoid query aliasing:
//   Pass 1 — read-only scan: collect all pending attacks with attacker pos.
//   Pass 2 — mutation: apply damage and ammo consumption per attack.
//
// Bevy allows two separate queries over different component sets on the same
// world without unsafe, as long as they don't mutably alias the same component
// on the same entity in the same system.

pub fn resolve_combat(
    mut commands:   Commands,
    grid:           Res<Grid>,
    offset:         Res<GridOffset>,
    mut attackers:  Query<(Entity, &GridPos, &PendingAction, &mut Ammo)>,
    mut targets:    Query<(Entity, &GridPos, &mut Hearts, &mut GoldCarried, &SpawnPoint)>,
) {
    // Build GridPos → target Entity lookup (read-only pass on targets).
    let target_map: std::collections::HashMap<GridPos, Entity> = targets
        .iter()
        .map(|(e, pos, _, _, _)| (*pos, e))
        .collect();

    // Collect all attacks before mutating anything.
    struct PendingAttack {
        attacker_entity: Entity,
        attacker_pos:    GridPos,
        dir:             super::action::Dir,
        is_ranged:       bool,
    }

    let mut pending_attacks: Vec<PendingAttack> = Vec::new();

    for (entity, pos, pending, _ammo) in attackers.iter() {
        match pending.0 {
            Some(Action::Attack(dir)) => {
                pending_attacks.push(PendingAttack {
                    attacker_entity: entity,
                    attacker_pos:    *pos,
                    dir,
                    is_ranged: false,
                });
            }
            Some(Action::RangedAttack(dir)) => {
                pending_attacks.push(PendingAttack {
                    attacker_entity: entity,
                    attacker_pos:    *pos,
                    dir,
                    is_ranged: true,
                });
            }
            _ => {}
        }
    }

    // Apply attacks.
    for attack in pending_attacks {
        // Ranged: consume ammo first; skip if empty.
        if attack.is_ranged {
            let Ok((_, _, _, mut ammo)) = attackers.get_mut(attack.attacker_entity) else { continue };
            if !ammo.consume() { continue; }
        }

        let (dx, dy) = attack.dir.delta();
        let range    = if attack.is_ranged { config::RANGED_RANGE } else { config::MELEE_RANGE };

        // Walk ray, stop at first hit.
        let mut hit_entity: Option<Entity> = None;
        for dist in 1..=range {
            let check = GridPos::new(
                attack.attacker_pos.x + dx * dist,
                attack.attacker_pos.y + dy * dist,
            );
            if attack.is_ranged {
                use crate::world::tile::Tile;
                if grid.get(check.x, check.y) == Some(Tile::Obstacle) { break; }
            }
            if let Some(&entity) = target_map.get(&check) {
                if entity != attack.attacker_entity { hit_entity = Some(entity); break; }
            }
        }

        let Some(target_entity) = hit_entity else { continue };
        let Ok((_, def_pos, mut hearts, mut gold, spawn)) =
            targets.get_mut(target_entity) else { continue };

        hearts.lose_one();
        info!(
            "{} hit {:?} at {:?} — {} heart(s) remaining",
            if attack.is_ranged { "Ranged" } else { "Melee" },
            target_entity, def_pos, hearts.0
        );

        if hearts.is_dead() {
            if gold.0 > 0 {
                let world_pos = offset.world_pos(def_pos.x, def_pos.y);
                for _ in 0..gold.0 {
                    commands.spawn(ItemBundle::new(
                        ItemKind::Gold, *def_pos, config::TILE_SIZE, world_pos,
                    ));
                }
                gold.0 = 0;
            }
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
        &mut Ammo, &SpawnPoint, &mut AgentBrain,
    )>,
) {
    for (entity, mut timer, mut pos, mut hearts, mut ammo, spawn, mut brain) in query.iter_mut() {
        if timer.0 > 0 {
            timer.0 -= 1;
            continue;
        }
        *pos    = spawn.0;
        *hearts = Hearts::default();
        *ammo   = Ammo::default();
        // Reset brain state — clears planner path, FSM/BT nav state, etc.
        // Calling brain.0.reset() here also makes AgentBehavior::reset()
        // reachable from ECS, resolving the dead_code warning.
        brain.0.reset();
        commands.entity(entity)
            .remove::<RespawnIn>()
            .remove::<super::components::SpeedBuff>();
        info!("Agent {:?} respawned at {:?}", entity, spawn.0);
    }
}