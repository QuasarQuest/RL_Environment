// src/rl/marker.rs
//
// RlAgent is a zero-size ECS marker component.
// Exactly one entity carries it — the agent being trained (team 1, Blue).
// All RL systems (obs, reward, action injection) filter on this marker
// instead of searching by team id or entity index.

use bevy::prelude::*;

#[derive(Component)]
pub struct RlAgent;