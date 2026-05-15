// src/agent/observation.rs

use std::sync::Arc;
use std::collections::HashSet;
use crate::world::Grid;
use crate::world::coords::GridPos;
use crate::world::tile::Tile;
use crate::team::Team;
use crate::item::ItemKind;
use super::components::{Ammo, GoldCarried, Hearts, Score};

// ── Visible types ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct VisibleAgent {
    pub pos:          GridPos,
    pub team:         Team,
    pub hearts:       Hearts,
    pub ammo:         Ammo,
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

// ── WorldSnapshot ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct WorldSnapshot {
    pub grid:     Arc<Grid>,
    pub occupied: Arc<HashSet<GridPos>>,
    pub agents:   Arc<Vec<VisibleAgent>>,
    pub items:    Arc<Vec<VisibleItem>>,
}

impl WorldSnapshot {
    pub fn new(
        grid:     Arc<Grid>,
        occupied: HashSet<GridPos>,
        agents:   Vec<VisibleAgent>,
        items:    Vec<VisibleItem>,
    ) -> Self {
        Self {
            grid,
            occupied: Arc::new(occupied),
            agents:   Arc::new(agents),
            items:    Arc::new(items),
        }
    }

    pub fn is_terrain_walkable(&self, pos: GridPos) -> bool {
        self.grid.is_walkable(pos.x, pos.y)
    }

    pub fn is_walkable(&self, pos: GridPos, self_pos: GridPos) -> bool {
        self.grid.is_walkable(pos.x, pos.y)
            && (!self.occupied.contains(&pos) || pos == self_pos)
    }
}

// ── Observation ───────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Observation {
    // Agent state
    pub pos:          GridPos,
    pub gold_carried: GoldCarried,
    pub hearts:       Hearts,
    pub ammo:         Ammo,
    pub score:        Score,
    pub team:         Team,
    pub tick:         u64,
    pub reward:       f32,

    world: WorldSnapshot,
}

impl Observation {
    pub fn new(
        pos:          GridPos,
        gold_carried: GoldCarried,
        hearts:       Hearts,
        ammo:         Ammo,
        score:        Score,
        team:         Team,
        tick:         u64,
        reward:       f32,
        world:        WorldSnapshot,
    ) -> Self {
        Self { pos, gold_carried, hearts, ammo, score, team, tick, reward, world }
    }

    // ── Terrain ───────────────────────────────────────────────────────────────

    pub fn is_walkable(&self, pos: GridPos) -> bool {
        self.world.is_walkable(pos, self.pos)
    }

    pub fn is_terrain_walkable(&self, pos: GridPos) -> bool {
        self.world.is_terrain_walkable(pos)
    }

    pub fn walkability_fn(&self) -> impl Fn(GridPos) -> bool + '_ {
        let grid = Arc::clone(&self.world.grid);
        move |pos| grid.is_walkable(pos.x, pos.y)
    }

    pub fn grid_tile(&self, pos: GridPos) -> Option<Tile> {
        self.world.grid.get(pos.x, pos.y)
    }

    // ── Items ─────────────────────────────────────────────────────────────────

    pub fn nearest_item(&self, kind: ItemKind) -> Option<GridPos> {
        self.world.items.iter()
            .filter(|i| i.kind == kind)
            .min_by_key(|i| i.pos.dist_sq(self.pos))
            .map(|i| i.pos)
    }

    pub fn items_of_kind(&self, kind: ItemKind) -> impl Iterator<Item = &VisibleItem> {
        self.world.items.iter().filter(move |i| i.kind == kind)
    }

    // ── Base ──────────────────────────────────────────────────────────────────

    pub fn nearest_own_base(&self) -> Option<GridPos> {
        let team_id = self.team.0;
        self.world.grid.iter()
            .filter(|(_, _, tile)| matches!(tile, Tile::Base(t) if *t == team_id))
            .min_by_key(|(x, y, _)| GridPos::new(*x as i32, *y as i32).dist_sq(self.pos))
            .map(|(x, y, _)| GridPos::new(x as i32, y as i32))
    }

    // ── Agents ────────────────────────────────────────────────────────────────

    pub fn visible_agents(&self) -> &[VisibleAgent] { &self.world.agents }

    pub fn nearest_enemy(&self) -> Option<&VisibleAgent> {
        self.world.agents.iter()
            .filter(|a| a.is_enemy(self.team) && a.pos != self.pos)
            .min_by_key(|a| a.pos.dist_sq(self.pos))
    }

    pub fn nearest_ally(&self) -> Option<&VisibleAgent> {
        self.world.agents.iter()
            .filter(|a| a.is_ally(self.team) && a.pos != self.pos)
            .min_by_key(|a| a.pos.dist_sq(self.pos))
    }

    // ── Convenience predicates ────────────────────────────────────────────────

    pub fn needs_health(&self) -> bool { !self.hearts.is_full() }
    pub fn needs_ammo(&self)   -> bool { !self.ammo.is_full() }
    pub fn has_ammo(&self)     -> bool { !self.ammo.is_empty() }
}