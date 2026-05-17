// src/viz/panels/components.rs
//
// Scoreboard ECS marker components.
// Imported by panels/scoreboard.rs via super::components.

use bevy::prelude::*;

// ── Tab scoreboard ────────────────────────────────────────────────────────────
#[derive(Component)] pub struct TabScoreboard;
#[derive(Component)] pub struct TabScoreboardContent;

/// Marks a scoreboard agent row.
#[derive(Component)] pub struct ScoreboardRow;

/// Marks the live team score in the team header. u8 = team_id.
#[derive(Component)] pub struct ScoreboardTeamScore(pub u8);

/// Marks avg stat cells embedded in the team header. u8 = team_id.
#[derive(Component)] pub struct ScoreboardAvgScore(pub u8);
#[derive(Component)] pub struct ScoreboardAvgKills(pub u8);
#[derive(Component)] pub struct ScoreboardAvgDeaths(pub u8);
#[derive(Component)] pub struct ScoreboardAvgKd(pub u8);

/// Per-cell markers for agent row in-place text updates.
#[derive(Component)] pub struct ScoreboardRowHp(pub Entity);
#[derive(Component)] pub struct ScoreboardRowAmmo(pub Entity);
#[derive(Component)] pub struct ScoreboardRowGold(pub Entity);
#[derive(Component)] pub struct ScoreboardRowScore(pub Entity);
#[derive(Component)] pub struct ScoreboardRowKills(pub Entity);
#[derive(Component)] pub struct ScoreboardRowDeaths(pub Entity);
#[derive(Component)] pub struct ScoreboardRowKd(pub Entity);

/// Text child of the RANGE / PATH viz buttons.
#[derive(Component)] pub struct ScoreboardRowRangeLabel(pub Entity);
#[derive(Component)] pub struct ScoreboardRowPathLabel(pub Entity);