// src/agent.rs
//
// Module declarations for the agent subsystem.
// composition.rs was removed when Brain<S,P> was collapsed into
// Box<dyn DecisionStrategy> — each strategy now owns its own planner.

pub mod action;
pub mod brain;
pub mod combat;
pub mod components;
pub mod debug;
pub mod observation;
pub mod planner;
pub mod plugin;
pub mod registry;
pub mod spawn;
pub mod strategy;
pub mod systems;

pub use plugin::AgentPlugin;