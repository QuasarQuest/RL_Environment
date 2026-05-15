// src/viz/hud/components.rs

use bevy::prelude::*;

// ── HUD label markers ─────────────────────────────────────────────────────────
#[derive(Component)] pub struct TickLabelMarker;
#[derive(Component)] pub struct TimeLabelMarker;
#[derive(Component)] pub struct TeamScoreMarker(pub u8);

// ── Tab scoreboard ────────────────────────────────────────────────────────────
#[derive(Component)] pub struct TabScoreboard;
#[derive(Component)] pub struct TabScoreboardContent;

// ── Speed / pause controls ────────────────────────────────────────────────────
#[derive(Component)] pub struct SpeedDecreaseButton;
#[derive(Component)] pub struct SpeedIncreaseButton;
#[derive(Component)] pub struct SpeedResetButton;
#[derive(Component)] pub struct CurrentSpeedLabel;
#[derive(Component)] pub struct PauseButtonMarker;
/// Marks the text child of the pause button so it can be updated independently.
#[derive(Component)] pub struct PauseButtonText;

// ── Debug viz toggle ──────────────────────────────────────────────────────────
#[derive(Component)] pub struct HideViz;