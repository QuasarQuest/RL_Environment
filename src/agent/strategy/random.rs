// src/agent/strategy/random.rs

use crate::agent::action::{Action, Dir};
use crate::agent::observation::Observation;
use crate::agent::planner::PathPlanner;
use crate::world::tile::Tile;
use super::DecisionStrategy;

pub struct RandomStrategy;

impl DecisionStrategy for RandomStrategy {
    fn name(&self) -> &'static str { "Random" }

    fn decide(&mut self, obs: &Observation, _planner: &mut impl PathPlanner) -> Action {
        let on_base = matches!(obs.grid_tile(obs.pos), Some(Tile::Base(t)) if t == obs.team.0);
        if on_base && !obs.gold_carried.is_empty() { return Action::Drop; }
        let dirs = Dir::all();
        Action::Move(dirs[rand::random_range(0..dirs.len())])
    }
}