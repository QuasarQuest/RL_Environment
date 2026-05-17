// src/agent/debug.rs

use bevy::prelude::Color;
use crate::agent::components::GridPos;

#[derive(Clone, Debug)]
pub struct DebugLine {
    pub start: GridPos,
    pub end:   GridPos,
    pub color: Color,
}

/// Agents that want to visualize their path implement this trait
/// to return line primitives in raw grid coordinates.
pub trait DebugDraw: Send + Sync {
    fn draw_lines(&self, agent_pos: GridPos) -> Vec<DebugLine>;
}