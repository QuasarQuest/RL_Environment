// src/viz/hud/components.rs

use bevy::prelude::*;

// ── HUD label markers ─────────────────────────────────────────────────────────
#[derive(Component)] pub struct TickLabelMarker;
#[derive(Component)] pub struct TimeLabelMarker;
#[derive(Component)] pub struct TeamScoreMarker(pub u8);

// ── Tab scoreboard ────────────────────────────────────────────────────────────
#[derive(Component)] pub struct TabScoreboard;
#[derive(Component)] pub struct TabScoreboardContent;

/// Marks a scoreboard row, carrying the agent Entity it represents.
#[allow(dead_code)]
#[derive(Component)] pub struct ScoreboardRow(pub Entity);

/// Per-cell markers for in-place text updates.
#[derive(Component)] pub struct ScoreboardRowHp(pub Entity);
#[derive(Component)] pub struct ScoreboardRowAmmo(pub Entity);
#[derive(Component)] pub struct ScoreboardRowGold(pub Entity);
#[derive(Component)] pub struct ScoreboardRowScore(pub Entity);

/// Text child of the RANGE viz button.
#[derive(Component)] pub struct ScoreboardRowRangeLabel(pub Entity);
/// Text child of the PATH viz button.
#[derive(Component)] pub struct ScoreboardRowPathLabel(pub Entity);

// ── Speed / pause controls ────────────────────────────────────────────────────
#[derive(Component)] pub struct SpeedDecreaseButton;
#[derive(Component)] pub struct SpeedIncreaseButton;
#[derive(Component)] pub struct SpeedResetButton;
#[derive(Component)] pub struct CurrentSpeedLabel;
#[derive(Component)] pub struct PauseButtonMarker;
#[derive(Component)] pub struct PauseButtonText;