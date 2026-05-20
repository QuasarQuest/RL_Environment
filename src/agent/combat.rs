// src/agent/combat.rs

use bevy::prelude::*;
use std::collections::HashMap;
use crate::world::Grid;
use crate::world::coords::GridPos;
use crate::world::config::WorldConfig;
use crate::team::{Team, TeamScore};
use super::action::{Action, Dir};
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

// ── Pending attack (internal collection type) ────────────────────────────────

struct PendingAttack {
    attacker_entity: Entity,
    attacker_pos:    GridPos,
    attacker_team:   Team,
    dir:             Dir,
    is_ranged:       bool,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Ray-march from `origin` along `dir` up to `range` tiles, returning the
/// first agent entity hit. Ranged attacks stop at obstacles; melee does not.
/// Self is never returned.
fn raycast_target(
    origin:     GridPos,
    dir:        Dir,
    range:      i32,
    is_ranged:  bool,
    grid:       &Grid,
    target_map: &HashMap<GridPos, Entity>,
    self_e:     Entity,
) -> Option<Entity> {
    use crate::world::tile::Tile;
    let (dx, dy) = dir.delta();
    for dist in 1..=range {
        let check = GridPos::new(origin.x + dx * dist, origin.y + dy * dist);
        if is_ranged && grid.get(check.x, check.y) == Some(Tile::Obstacle) {
            break;
        }
        if let Some(&e) = target_map.get(&check) {
            if e != self_e { return Some(e); }
        }
    }
    None
}

/// Returns true if the tile under `pos` is a safe zone owned by a team
/// other than `attacker_team`.
fn is_in_enemy_safe_zone(grid: &Grid, pos: GridPos, attacker_team: Team) -> bool {
    grid.get(pos.x, pos.y)
        .and_then(|t| t.team_id())
        .map(|t| t != attacker_team.0)
        .unwrap_or(false)
}

/// Apply `damage` hearts of damage. Returns true if the target died on
/// this hit.
fn apply_damage(hearts: &mut Hearts, damage: u8) -> bool {
    for _ in 0..damage { hearts.lose_one(); }
    hearts.is_dead()
}

/// All bookkeeping for a kill: attacker stats, team score, defender
/// death count, gold drop, and respawn timer.
fn handle_death(
    commands:       &mut Commands,
    attackers:      &mut Query<(
        Entity, &GridPos, &Team, &PendingAction,
        &mut Ammo, &mut KillCount, &mut Score,
        Option<&AttackCooldown>,
    )>,
    team_score:     &mut TeamScore,
    attacker_e:     Entity,
    attacker_team:  Team,
    target_e:       Entity,
    target_pos:     GridPos,
    target_gold:    &mut GoldCarried,
    target_deaths:  &mut DeathCount,
    map:            &WorldConfig,
    #[cfg(not(feature = "headless"))]
    offset:         &GridOffset,
) {
    if let Ok((_, _, _, _, _, mut kills, mut score, _)) =
        attackers.get_mut(attacker_e)
    {
        kills.0 += 1;
        score.0 += map.kill_reward as u32;
        team_score.add(attacker_team, map.kill_reward as u32);
        info!("Kill +{} → team {}", map.kill_reward, attacker_team.name());
    }
    target_deaths.0 += 1;

    #[cfg(not(feature = "headless"))]
    if target_gold.0 > 0 {
        let world_pos = offset.world_pos(target_pos.x, target_pos.y);
        for _ in 0..target_gold.0 {
            commands.spawn(ItemBundle::new(
                ItemKind::Gold, target_pos, config::TILE_SIZE, world_pos,
            ));
        }
    }
    target_gold.0 = 0;

    commands.entity(target_e).insert(RespawnIn(map.respawn_ticks));
    info!("Agent {:?} died — respawning in {} ticks",
        target_e, map.respawn_ticks);
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
    // Build a tile → entity map of all valid targets.
    let target_map: HashMap<GridPos, Entity> = targets
        .iter()
        .map(|(e, pos, _, _, _, _)| (*pos, e))
        .collect();

    // Phase 1: collect pending attacks (releases the attackers borrow).
    let pending_attacks: Vec<PendingAttack> = attackers
        .iter()
        .filter(|(_, _, _, _, _, _, _, cd)| cd.is_none())
        .filter_map(|(e, pos, team, pending, _, _, _, _)| {
            match pending.0 {
                Some(Action::Attack(dir))       => Some(PendingAttack {
                    attacker_entity: e, attacker_pos: *pos, attacker_team: *team,
                    dir, is_ranged: false,
                }),
                Some(Action::RangedAttack(dir)) => Some(PendingAttack {
                    attacker_entity: e, attacker_pos: *pos, attacker_team: *team,
                    dir, is_ranged: true,
                }),
                _ => None,
            }
        })
        .collect();

    // Phase 2: resolve each attack.
    for attack in pending_attacks {
        // Ranged costs ammo — consume it up front, bail if empty.
        if attack.is_ranged {
            let Ok((_, _, _, _, mut ammo, _, _, _)) =
                attackers.get_mut(attack.attacker_entity) else { continue };
            if !ammo.consume() { continue; }
        }

        // Apply cooldown regardless of hit/miss.
        let cd = if attack.is_ranged {
            map.ranged_cooldown_ticks
        } else {
            map.melee_cooldown_ticks
        };
        commands.entity(attack.attacker_entity).insert(AttackCooldown(cd));

        // Find target via raycast.
        let range = if attack.is_ranged { map.ranged_range as i32 } else { map.melee_range as i32 };
        let Some(target_e) = raycast_target(
            attack.attacker_pos, attack.dir, range, attack.is_ranged,
            &grid, &target_map, attack.attacker_entity,
        ) else { continue };

        // Resolve hit on target.
        let Ok((_, def_pos, mut hearts, mut gold, _spawn, mut deaths)) =
            targets.get_mut(target_e) else { continue };

        if is_in_enemy_safe_zone(&grid, *def_pos, attack.attacker_team) {
            info!("Attack blocked — target in safe zone at {:?}", def_pos);
            continue;
        }

        let damage = if attack.is_ranged { map.ranged_damage } else { map.melee_damage };
        let died   = apply_damage(&mut hearts, damage);

        info!(
            "{} hit {:?} at {:?} — {} heart(s) remaining",
            if attack.is_ranged { "Ranged" } else { "Melee" },
            target_e, def_pos, hearts.0
        );

        if died {
            let target_pos = *def_pos;
            handle_death(
                &mut commands, &mut attackers, &mut team_score,
                attack.attacker_entity, attack.attacker_team,
                target_e, target_pos, &mut gold, &mut deaths,
                &map,
                #[cfg(not(feature = "headless"))]
                &offset,
            );
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