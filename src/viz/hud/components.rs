// src/viz/hud/components.rs
//
// HUD toolbar markers only — scoreboard markers live in panels/components.rs.

use bevy::prelude::*;

// ── HUD label markers ─────────────────────────────────────────────────────────
#[derive(Component)] pub struct TickLabelMarker;
#[derive(Component)] pub struct TimeLabelMarker;
#[derive(Component)] pub struct TeamScoreMarker(pub u8);

// ── Speed / pause controls ────────────────────────────────────────────────────
#[derive(Component)] pub struct SpeedDecreaseButton;
#[derive(Component)] pub struct SpeedIncreaseButton;
#[derive(Component)] pub struct SpeedResetButton;
#[derive(Component)] pub struct CurrentSpeedLabel;
#[derive(Component)] pub struct PauseButtonMarker;
#[derive(Component)] pub struct PauseButtonText;