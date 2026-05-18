// src/factory/mod.rs

use bevy::prelude::*;

#[cfg(not(feature = "headless"))]
mod display;

#[cfg(not(feature = "headless"))]
pub use display::assign_display_components;

/// Carries the index into `MapConfig::agents` for the entity spawned from
/// that slot. Set at spawn time; read by the factory layer to look up
/// display metadata without relying on entity-ID ordering.
#[derive(Component)]
pub struct AgentConfigIndex(pub usize);