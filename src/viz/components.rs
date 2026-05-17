// src/viz/components.rs
//
// Display-only ECS components. Attached to agent entities by the factory layer
// after spawn, consumed exclusively by viz systems. The agent simulation module
// has no knowledge of these.

use bevy::prelude::*;

// ── Agent display label ───────────────────────────────────────────────────────

#[derive(Component, Clone, Debug)]
pub struct AgentLabel(pub String);

impl AgentLabel {
    pub fn new(name: impl Into<String>) -> Self { Self(name.into()) }
}

// ── Agent algorithm info ──────────────────────────────────────────────────────

#[derive(Component, Clone, Debug)]
pub struct AgentInfo {
    pub strategy: &'static str,
    pub planner:  &'static str,
}

// ── Debug viz toggles ─────────────────────────────────────────────────────────
//
// Each marker: present = hidden, absent = visible.
// Inserted at spawn (all hidden by default).
//
// HideRangeViz — gates melee + ranged combat range rings.
// HidePathViz  — gates path polyline + destination marker from debug_draw().
//
// The old HideViz is kept as an alias that controls both, used by the
// tooltip left-click toggle for convenience.

#[derive(Component)] pub struct HideRangeViz;
#[derive(Component)] pub struct HidePathViz;

/// Convenience marker inserted/removed by tooltip left-click.
/// When present, both range and path overlays are hidden.
/// The scoreboard buttons operate HideRangeViz / HidePathViz independently.
#[derive(Component)] pub struct HideViz;