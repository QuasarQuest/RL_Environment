// src/agent/systems.rs
//
// Changes from original:
//   1. Observation is now owned — built from WorldSnapshot (Arc internals).
//      No borrowed references escape into Brain::act.
//   2. apply_actions: count-map replaced with claimed HashSet.
//   3. AgentBrain replaces Brain as the ECS component name.

use bevy::prelude::*;
use std::collections::HashSet;
use std::sync::Arc;
use crate::world::{Grid, tile::Tile};
use crate::world::coords::GridPos;
use crate::sim::config::SimConfig;
use crate::team::{Team, TeamScore};
use crate::item::Item; // Removed ItemKind
use super::action::Action;
use super::brain::AgentBrain;
use super::components::{GoldCarried, Health, Score};
use super::observation::{Observation, VisibleAgent, VisibleItem, WorldSnapshot};

#[derive(Component, Default)]
pub struct PendingAction(pub Option<Action>);

pub fn tick_agents(
    grid:      Res<Grid>,
    sim_cfg:   Res<SimConfig>,
    items:     Query<(&GridPos, &Item)>,
    mut query: Query<(
        &GridPos, &GoldCarried, &Health, &Score, &Team,
        &mut AgentBrain, &mut PendingAction,
    )>,
) {
    let tick = sim_cfg.tick;

    // Build shared snapshot once — O(n) work shared across all agents.
    let occupied: HashSet<GridPos> = query.iter().map(|(pos, ..)| *pos).collect();

    let all_agents: Vec<VisibleAgent> = query.iter()
        .map(|(pos, gold, health, _, team, _, _)| VisibleAgent {
            pos: *pos, team: *team, health: *health, gold_carried: *gold,
        })
        .collect();

    let all_items: Vec<VisibleItem> = items.iter()
        .map(|(pos, item)| VisibleItem { pos: *pos, kind: item.kind })
        .collect();

    let snapshot = WorldSnapshot::new(
        Arc::new((*grid).clone()), // Dereference the Res wrapper, then clone the Grid
        occupied,
        all_agents,
        all_items,
    );

    for (pos, gold, health, score, team, mut brain, mut pending) in query.iter_mut() {
        let obs = Observation::new(
            *pos, *gold, *health, *score, *team,
            tick, 0.0,
            snapshot.clone(), // Arc clone — O(1)
        );
        pending.0 = Some(brain.act(&obs));
    }
}

pub fn apply_actions(
    grid:           Res<Grid>, // Changed from mut grid: ResMut<Grid>
    mut team_score: ResMut<TeamScore>,
    mut query: Query<(
        Entity, &mut GridPos, &mut GoldCarried,
        &mut Score, &Team, &mut PendingAction,
    )>,
) {
    // Claimed set: first agent to request a cell wins.
    // Losers wait; stuck_ticks in strategy triggers replan after 3 ticks.
    let mut claimed: HashSet<GridPos> = HashSet::new();

    for (_, mut pos, mut gold, mut score, team, mut pending) in query.iter_mut() {
        let Some(action) = pending.0.take() else { continue };

        match action {
            Action::Move(dir) => {
                let (dx, dy) = dir.delta();
                let next     = pos.apply_delta(dx, dy);
                if grid.is_walkable(next.x, next.y) && claimed.insert(next) {
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
            Action::Pickup | Action::Attack(_) | Action::Wait => {}
        }
    }
}