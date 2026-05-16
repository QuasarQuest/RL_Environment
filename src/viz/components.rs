// src/viz/components.rs
//
// Display-only ECS components. These are attached to agent entities by the
// factory layer after spawn, and consumed exclusively by viz systems.
// The agent simulation module has no knowledge of these.

use bevy::prelude::*;

// ── Agent display label ───────────────────────────────────────────────────────
// Human-readable name shown in scoreboard and tooltip.
// Format: "Red #1", "Blue #2", etc.

#[derive(Component, Clone, Debug)]
pub struct AgentLabel(pub String);

impl AgentLabel {
    pub fn new(name: impl Into<String>) -> Self { Self(name.into()) }
}

// ── Agent algorithm info ──────────────────────────────────────────────────────
// Strategy and planner names shown as subtext in the scoreboard.

#[derive(Component, Clone, Debug)]
pub struct AgentInfo {
    pub strategy: &'static str,
    pub planner:  &'static str,
}

// ── Debug viz toggle ──────────────────────────────────────────────────────────
// Marker component: present = hide debug overlay, absent = show it.
// Inserted at spawn (hidden by default), toggled by scoreboard VIZ button
// and tooltip left-click.

#[derive(Component)]
pub struct HideViz;