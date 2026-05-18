// src/agent/combat.rs

use bevy::prelude::*;
use crate::world::Grid;
use crate::world::coords::GridPos;
use crate::world::config::WorldConfig;
use crate::team::{Team, TeamScore};
use super::action::Action;
use super::brain::AgentBrain;
use super::components::{
    Ammo, AttackCooldown, DeathCount, GoldCarried, Hearts,
    KillCount, RespawnIn, Score, SpawnPoint,
};
use super::systems::PendingAction;

#[cfg(not(feature = "headless"))]
use crate::item::{ItemBundle, ItemKind};
#[cfg(not(feature = "headless"))]
use crate::config;
#[cfg(not(feature = "headless"))]
use crate::viz::grid_offset::GridOffset;

// ── Tick attack cooldown ──────────────────────────────────────────────────────

pub fn tick_attack_cooldown(
    mut commands: Commands,
    mut query:    Query<(Entity, &mut AttackCooldown)>,
) {
    for (entity, mut cd) in query.iter_mut() {
        if cd.0 > 1 {
            cd.0 -= 1;
        } else {
            commands.entity(entity).remove::<AttackCooldown>();
        }
    }
}

// ── Resolve combat ────────────────────────────────────────────────────────────

pub fn resolve_combat(
    mut commands:   Commands,
    grid:           Res<Grid>,
    #[cfg(not(feature = "headless"))]
    offset:         Res<GridOffset>,
    map:            Res<WorldConfig>,
    mut team_score: ResMut<TeamScore>,
    mut attackers:  Query<(
        Entity, &GridPos, &Team, &PendingAction,
        &mut Ammo, &mut KillCount, &mut Score,
        Option<&AttackCooldown>,
    )>,
    mut targets:    Query<(
        Entity, &GridPos, &mut Hearts, &mut GoldCarried,
        &SpawnPoint, &mut DeathCount,
    )>,
) {
    let target_map: std::collections::HashMap<GridPos, Entity> = targets
        .iter()
        .map(|(e, pos, _, _, _, _)| (*pos, e))
        .collect();

    struct PendingAttack {
        attacker_entity: Entity,
        attacker_pos:    GridPos,
        attacker_team:   Team,
        dir:             super::action::Dir,
        is_ranged:       bool,
    }

    let mut pending_attacks: Vec<PendingAttack> = Vec::new();

    for (entity, pos, team, pending, _ammo, _kills, _score, cooldown) in attackers.iter() {
        if cooldown.is_some() { continue; }
        match pending.0 {
            Some(Action::Attack(dir)) => pending_attacks.push(PendingAttack {
                attacker_entity: entity,
                attacker_pos:    *pos,
                attacker_team:   *team,
                dir,
                is_ranged: false,
            }),
            Some(Action::RangedAttack(dir)) => pending_attacks.push(PendingAttack {
                attacker_entity: entity,
                attacker_pos:    *pos,
                attacker_team:   *team,
                dir,
                is_ranged: true,
            }),
            _ => {}
        }
    }

    for attack in pending_attacks {
        if attack.is_ranged {
            let Ok((_, _, _, _, mut ammo, _, _, _)) =
                attackers.get_mut(attack.attacker_entity) else { continue };
            if !ammo.consume() { continue; }
        }

        let cd = if attack.is_ranged {
            map.ranged_cooldown_ticks
        } else {
            map.melee_cooldown_ticks
        };
        commands.entity(attack.attacker_entity).insert(AttackCooldown(cd));

        let (dx, dy) = attack.dir.delta();
        let range    = if attack.is_ranged { map.ranged_range } else { map.melee_range };
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
        let Ok((_, def_pos, mut hearts, mut gold, _spawn, mut deaths)) =
            targets.get_mut(target_entity) else { continue };

        let tile_team = grid.get(def_pos.x, def_pos.y).and_then(|t| t.team_id());
        if tile_team.is_some() && tile_team != Some(attack.attacker_team.0) {
            info!("Attack blocked — target in safe zone at {:?}", def_pos);
            continue;
        }

        let damage = if attack.is_ranged { map.ranged_damage } else { map.melee_damage };
        for _ in 0..damage { hearts.lose_one(); }

        info!(
            "{} hit {:?} at {:?} — {} heart(s) remaining",
            if attack.is_ranged { "Ranged" } else { "Melee" },
            target_entity, def_pos, hearts.0
        );

        if hearts.is_dead() {
            if let Ok((_, _, _, _, _, mut kills, mut score, _)) =
                attackers.get_mut(attack.attacker_entity)
            {
                kills.0 += 1;
                score.0 += map.kill_reward as u32;
                team_score.add(attack.attacker_team, map.kill_reward as u32);
                info!("Kill +{} → team {}", map.kill_reward, attack.attacker_team.name());
            }
            deaths.0 += 1;

            #[cfg(not(feature = "headless"))]
            if gold.0 > 0 {
                let world_pos = offset.world_pos(def_pos.x, def_pos.y);
                for _ in 0..gold.0 {
                    commands.spawn(ItemBundle::new(
                        ItemKind::Gold, *def_pos, config::TILE_SIZE, world_pos,
                    ));
                }
            }
            gold.0 = 0;

            commands.entity(target_entity).insert(RespawnIn(map.respawn_ticks));
            info!("Agent {:?} died — respawning in {} ticks",
                target_entity, map.respawn_ticks);
        }
    }
}

// ── Respawn ───────────────────────────────────────────────────────────────────

pub fn tick_respawn(
    mut commands: Commands,
    mut query:    Query<(
        Entity, &mut RespawnIn, &mut GridPos, &mut Hearts,
        &mut Ammo, &SpawnPoint, &mut AgentBrain,
    )>,
) {
    for (entity, mut timer, mut pos, mut hearts, mut ammo, spawn, mut brain)
    in query.iter_mut()
    {
        if timer.0 > 0 { timer.0 -= 1; continue; }

        *pos    = spawn.0;
        *hearts = Hearts::default();
        *ammo   = Ammo::default();
        brain.0.reset();

        commands.entity(entity)
            .remove::<RespawnIn>()
            .remove::<AttackCooldown>()
            .remove::<super::components::SpeedBuff>();

        info!("Agent {:?} respawned at {:?}", entity, spawn.0);
    }
}