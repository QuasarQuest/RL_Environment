// src/agent/systems.rs

use bevy::prelude::*;
use std::collections::{HashMap, HashSet};
use crate::world::{Grid, tile::Tile};
use crate::world::coords::GridPos;
use crate::sim::config::SimConfig;
use crate::team::{Team, TeamScore};
use crate::item::{Item, ItemKind};
use super::action::Action;
use super::brain::Brain;
use super::components::{GoldCarried, Health, Score};
use super::observation::{Observation, VisibleAgent, VisibleItem};

#[derive(Component, Default)]
pub struct PendingAction(pub Option<Action>);

pub fn tick_agents(
    grid:      Res<Grid>,
    sim_cfg:   Res<SimConfig>,
    items:     Query<(&GridPos, &Item)>,
    mut query: Query<(
        &GridPos, &GoldCarried, &Health, &Score, &Team,
        &mut Brain, &mut PendingAction,
    )>,
) {
    let tick = sim_cfg.tick;

    let occupied: HashSet<GridPos> = query.iter().map(|(pos, ..)| *pos).collect();

    let all_agents: Vec<VisibleAgent> = query.iter()
        .map(|(pos, gold, health, _, team, _, _)| VisibleAgent {
            pos: *pos, team: *team, health: *health, gold_carried: *gold,
        })
        .collect();

    let all_items: Vec<VisibleItem> = items.iter()
        .map(|(pos, item)| VisibleItem { pos: *pos, kind: item.kind })
        .collect();

    for (pos, gold, health, score, team, mut brain, mut pending) in query.iter_mut() {
        let others: Vec<VisibleAgent> = all_agents.iter()
            .filter(|a| a.pos != *pos)
            .copied()
            .collect();

        let obs = Observation::new(
            *pos, *gold, *health, *score, *team,
            &grid, &occupied, &others, &all_items, tick, 0.0,
        );
        pending.0 = Some(brain.act(&obs));
    }
}

pub fn apply_actions(
    mut grid:       ResMut<Grid>,
    mut team_score: ResMut<TeamScore>,
    mut query: Query<(
        Entity, &mut GridPos, &mut GoldCarried,
        &mut Score, &Team, &mut PendingAction,
    )>,
) {
    let mut requested:  HashMap<GridPos, u32>        = HashMap::new();

    for (_, pos, _, _, _, pending) in query.iter() {
        if let Some(Action::Move(dir)) = &pending.0 {
            let (dx, dy) = dir.delta();
            *requested.entry(pos.apply_delta(dx, dy)).or_insert(0) += 1;
        }
    }

    for (_, mut pos, mut gold, mut score, team, mut pending) in query.iter_mut() {
        let Some(action) = pending.0.take() else { continue };

        match action {
            Action::Move(dir) => {
                let (dx, dy) = dir.delta();
                let next     = pos.apply_delta(dx, dy);
                if grid.is_walkable(next.x, next.y)
                    && requested.get(&next).copied().unwrap_or(0) == 1
                {
                    *pos = next;
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
            // Pickup is now handled by item::pickup::pickup_items — no tile mutation needed.
            Action::Pickup | Action::Attack(_) | Action::Wait => {}
        }
    }
}