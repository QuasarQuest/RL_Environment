// src/agent/brain.rs
//
// Two things only:
//   AgentBehavior  — the trait every agent must implement
//   AgentBrain     — the Bevy ECS Component (one Box<dyn AgentBehavior>)
//
// Composition logic (Brain<S,P>) lives in composition.rs.

use bevy::prelude::Component;
use crate::agent::action::Action;
use crate::agent::debug::DebugDraw;
use crate::agent::observation::Observation;

pub trait AgentBehavior: Send + Sync {
    fn name(&self) -> &str;
    fn act(&mut self, obs: &Observation) -> Action;
    fn debug_draw(&self) -> Option<Box<dyn DebugDraw>> { None }
    fn reset(&mut self) {}
}

#[derive(Component)]
pub struct AgentBrain(pub Box<dyn AgentBehavior>);

impl AgentBrain {
    pub fn act(&mut self, obs: &Observation) -> Action     { self.0.act(obs) }
    pub fn debug_draw(&self) -> Option<Box<dyn DebugDraw>> { self.0.debug_draw() }
}