// src/agent/systems.rs

use bevy::prelude::*;
use std::collections::HashSet;
use std::sync::Arc;
use crate::world::{Grid, tile::Tile};
use crate::world::coords::GridPos;
use crate::sim::config::SimConfig;
use crate::team::{Team, TeamScore};
use crate::item::Item;
use super::action::Action;
use super::brain::AgentBrain;
use super::components::{Ammo, GoldCarried, Hearts, RespawnIn, Score, SpeedBuff};
use super::observation::{Observation, VisibleAgent, VisibleItem, WorldSnapshot};

#[derive(Component, Default)]
pub struct PendingAction(pub Option<Action>);

// ── Tick agents ───────────────────────────────────────────────────────────────

pub fn tick_agents(
    grid:      Res<Grid>,
    sim_cfg:   Res<SimConfig>,
    items:     Query<(&GridPos, &Item)>,
    mut query: Query<(
        &GridPos, &GoldCarried, &Hearts, &Ammo, &Score, &Team,
        &mut AgentBrain, &mut PendingAction,
    ), Without<RespawnIn>>,
) {
    let tick = sim_cfg.tick;

    let occupied: HashSet<GridPos> = query.iter().map(|(pos, ..)| *pos).collect();

    let all_agents: Vec<VisibleAgent> = query.iter()
        .map(|(pos, gold, hearts, ammo, _, team, _, _)| VisibleAgent {
            pos:          *pos,
            team:         *team,
            hearts:       *hearts,
            ammo:         *ammo,
            gold_carried: *gold,
        })
        .collect();

    let all_items: Vec<VisibleItem> = items.iter()
        .map(|(pos, item)| VisibleItem { pos: *pos, kind: item.kind })
        .collect();

    let snapshot = WorldSnapshot::new(
        Arc::new((*grid).clone()),
        occupied,
        all_agents,
        all_items,
    );

    for (pos, gold, hearts, ammo, score, team, mut brain, mut pending) in query.iter_mut() {
        let obs = Observation::new(
            *pos, *gold, *hearts, *ammo, *score, *team,
            tick, 0.0,
            snapshot.clone(),
        );
        pending.0 = Some(brain.act(&obs));
    }
}

// ── Tick speed buff ───────────────────────────────────────────────────────────

pub fn tick_speed_buff(
    mut commands: Commands,
    mut query:    Query<(Entity, &mut SpeedBuff)>,
) {
    for (entity, mut buff) in query.iter_mut() {
        if buff.0 > 0 {
            buff.0 -= 1;
        } else {
            commands.entity(entity).remove::<SpeedBuff>();
        }
    }
}

// ── Apply actions ─────────────────────────────────────────────────────────────
//
// Handles Move and Drop only.
// Attack and RangedAttack are left in PendingAction for resolve_combat.
// Wait is consumed here as a no-op.

pub fn apply_actions(
    grid:           Res<Grid>,
    mut team_score: ResMut<TeamScore>,
    mut query: Query<(
        Entity, &mut GridPos, &mut GoldCarried,
        &mut Score, &Team, &mut PendingAction,
        Option<&SpeedBuff>,
    ), Without<RespawnIn>>,
) {
    let mut claimed: HashSet<GridPos> = HashSet::new();

    for (_, mut pos, mut gold, mut score, team, mut pending, speed) in query.iter_mut() {
        // Peek at the action — only consume it if it's not a combat action.
        let is_combat = matches!(pending.0, Some(Action::Attack(_)) | Some(Action::RangedAttack(_)));
        if is_combat { continue; }

        let Some(action) = pending.0.take() else { continue };

        let moves = if speed.is_some() { 2 } else { 1 };

        match action {
            Action::Move(dir) => {
                let (dx, dy) = dir.delta();
                for _ in 0..moves {
                    let next = pos.apply_delta(dx, dy);
                    if grid.is_walkable(next.x, next.y) && claimed.insert(next) {
                        *pos = next;
                    } else {
                        break;
                    }
                }
            }
            Action::Drop => {
                let on_own_base = matches!(
                    grid.get(pos.x, pos.y),
                    Some(Tile::Base(t)) if t == team.0
                );
                if on_own_base && !gold.is_empty() {
                    let delivered = gold.0;
                    score.0      += delivered;
                    team_score.add(*team, delivered);
                    gold.0        = 0;
                    info!("Team {} delivered {} gold — total: {}",
                        team.name(), delivered, team_score.get(*team));
                }
            }
            Action::Wait => {} // consumed, nothing to do
            // Combat actions are handled above — this arm is unreachable.
            Action::Attack(_) | Action::RangedAttack(_) => unreachable!(),
        }
    }
}