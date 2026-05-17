// src/factory/mod.rs

mod display;

pub use display::assign_display_components;

use bevy::prelude::*;

/// Carries the index into `MapConfig::agents` for the entity spawned from
/// that slot. Set at spawn time; read by the factory layer to look up
/// display metadata without relying on entity-ID ordering.
#[derive(Component)]
pub struct AgentConfigIndex(pub usize);