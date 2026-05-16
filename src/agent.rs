// src/agent.rs

pub mod action;
pub mod brain;
pub mod combat;
pub mod components;
pub mod composition;
pub mod debug;
pub mod observation;
pub mod planning;
pub mod plugin;
pub mod registry;
pub mod spawn;
pub mod systems;

pub use plugin::AgentPlugin;