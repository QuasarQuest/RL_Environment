// src/agent/planning/random_agent.rs

use crate::agent::action::{Action, Dir};
use crate::agent::brain::Agent;
use crate::agent::observation::Observation;
use crate::item::ItemKind;
use crate::world::tile::Tile;

pub struct RandomAgent;

impl Agent for RandomAgent {
    fn name(&self) -> &str { "Random Walker" }

    fn act(&mut self, obs: &Observation<'_>) -> Action {
        // Drop on own base if carrying gold.
        let on_own_base = matches!(
            obs.grid_tile(obs.pos),
            Some(Tile::Base(t)) if t == obs.team.0
        );
        if on_own_base && !obs.gold_carried.is_empty() {
            return Action::Drop;
        }
        // Pickup is handled automatically by the item system — just move randomly.
        let dirs = Dir::all();
        Action::Move(dirs[rand::random_range(0..dirs.len())])
    }
}