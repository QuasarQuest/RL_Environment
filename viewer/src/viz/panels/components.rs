use bevy::prelude::*;

#[derive(Component)] pub struct TabScoreboard;
#[derive(Component)] pub struct TabScoreboardContent;
#[derive(Component)] pub struct ScoreboardRow;
#[derive(Component)] pub struct ScoreboardTeamScore(pub u8);

// Per-row live-update markers — keyed by agent index (usize).
#[derive(Component)] pub struct ScoreboardRowSpeed(pub usize);
#[derive(Component)] pub struct ScoreboardRowSlow(pub usize);
#[derive(Component)] pub struct ScoreboardRowMult(pub usize);
#[derive(Component)] pub struct ScoreboardRowGold(pub usize);
#[derive(Component)] pub struct ScoreboardRowScore(pub usize);
