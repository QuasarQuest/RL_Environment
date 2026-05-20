// src/agent/brain.rs
//
// AgentBrain is the Bevy ECS component holding one boxed DecisionStrategy.
// The DecisionStrategy trait itself lives in strategy/mod.rs and now
// carries everything an agent needs: name, decide, debug_draw, reset.
//
// There is no separate AgentBehavior trait or Brain<S,P> wrapper — every
// strategy owns its own planner (if it needs one), so the strategy *is*
// the behaviour.

use bevy::prelude::Component;
use crate::agent::action::Action;
use crate::agent::debug::DebugDraw;
use crate::agent::observation::Observation;
use crate::agent::strategy::DecisionStrategy;

#[derive(Component)]
pub struct AgentBrain(pub Box<dyn DecisionStrategy>);

impl AgentBrain {
    pub fn act(&mut self, obs: &Observation) -> Action     { self.0.decide(obs) }
    pub fn debug_draw(&self) -> Option<Box<dyn DebugDraw>> { self.0.debug_draw() }
}