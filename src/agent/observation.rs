// src/agent/observation.rs

use std::collections::HashSet;
use crate::world::Grid;
use crate::world::coords::GridPos;
use crate::world::tile::Tile;
use crate::team::Team;
use crate::item::{Item, ItemKind};
use super::components::{GoldCarried, Health, Score};

#[derive(Clone, Copy, Debug)]
pub struct VisibleAgent {
    pub pos:          GridPos,
    pub team:         Team,
    pub health:       Health,
    pub gold_carried: GoldCarried,
}

impl VisibleAgent {
    pub fn is_enemy(&self, my_team: Team) -> bool { self.team != my_team }
    pub fn is_ally(&self, my_team: Team)  -> bool { self.team == my_team }
}

#[derive(Clone, Copy, Debug)]
pub struct VisibleItem {
    pub pos:  GridPos,
    pub kind: ItemKind,
}

#[derive(Clone)]
pub struct Observation<'a> {
    pub pos:          GridPos,
    pub gold_carried: GoldCarried,
    pub health:       Health,
    pub score:        Score,
    pub team:         Team,
    pub tick:         u64,
    pub reward:       f32,

    grid:             &'a Grid,
    occupied:         &'a HashSet<GridPos>,
    pub other_agents: &'a [VisibleAgent],
    pub visible_items: &'a [VisibleItem],
}

impl<'a> Observation<'a> {
    pub fn new(
        pos: GridPos, gold_carried: GoldCarried, health: Health,
        score: Score, team: Team, grid: &'a Grid,
        occupied: &'a HashSet<GridPos>, other_agents: &'a [VisibleAgent],
        visible_items: &'a [VisibleItem],
        tick: u64, reward: f32,
    ) -> Self {
        Self { pos, gold_carried, health, score, team, grid, occupied,
            other_agents, visible_items, tick, reward }
    }

    pub fn is_tile(&self, pos: GridPos, tile: Tile) -> bool {
        self.grid.get(pos.x, pos.y) == Some(tile)
    }

    /// Nearest item of a given kind.
    pub fn nearest_item(&self, kind: ItemKind) -> Option<GridPos> {
        self.visible_items.iter()
            .filter(|i| i.kind == kind)
            .min_by_key(|i| i.pos.dist_sq(self.pos))
            .map(|i| i.pos)
    }

    /// Nearest base tile owned by this agent's team.
    pub fn nearest_own_base(&self) -> Option<GridPos> {
        let team_id = self.team.0;
        self.grid.iter()
            .filter(|(_, _, tile)| matches!(tile, Tile::Base(t) if *t == team_id))
            .min_by_key(|(x, y, _)| GridPos::new(*x as i32, *y as i32).dist_sq(self.pos))
            .map(|(x, y, _)| GridPos::new(x as i32, y as i32))
    }

    pub fn is_walkable(&self, pos: GridPos) -> bool {
        self.grid.is_walkable(pos.x, pos.y)
            && (!self.occupied.contains(&pos) || pos == self.pos)
    }

    pub fn grid_tile(&self, pos: GridPos) -> Option<Tile> {
        self.grid.get(pos.x, pos.y)
    }

    pub fn grid_is_walkable_terrain(&self, pos: GridPos) -> bool {
        self.grid.is_walkable(pos.x, pos.y)
    }

    pub fn visible_agents(&self) -> &[VisibleAgent] { self.other_agents }

    pub fn nearest_enemy(&self) -> Option<&VisibleAgent> {
        self.other_agents.iter()
            .filter(|a| a.is_enemy(self.team))
            .min_by_key(|a| a.pos.dist_sq(self.pos))
    }

    pub fn nearest_ally(&self) -> Option<&VisibleAgent> {
        self.other_agents.iter()
            .filter(|a| a.is_ally(self.team) && a.pos != self.pos)
            .min_by_key(|a| a.pos.dist_sq(self.pos))
    }

    pub fn nearest_agent(&self) -> Option<&VisibleAgent> {
        self.other_agents.iter()
            .filter(|a| a.pos != self.pos)
            .min_by_key(|a| a.pos.dist_sq(self.pos))
    }
}