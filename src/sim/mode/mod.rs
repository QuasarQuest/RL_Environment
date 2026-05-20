// src/sim/mode/mod.rs
use bevy::prelude::*;
use serde::Deserialize;

pub mod gold_rush;
pub use gold_rush::GoldRush;

// ── GameModeKind ──────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default, Resource)]
pub enum GameModeKind {
    #[default]
    GoldRush,
    TeamDeathmatch,
    CaptureTheFlag,
    Conquest,
    FreeForAll,
}

// ── PrevModeState ─────────────────────────────────────────────────────────────

/// Per-tick snapshot carried between steps so reward functions can compute
/// deltas. Each mode reads whatever fields it needs.
#[derive(Default, Clone, Copy)]
pub struct PrevModeState {
    pub score:    u32,
    pub was_dead: bool,
    pub gold:     u8,
}

// ── GameMode trait ────────────────────────────────────────────────────────────

/// Implemented by every game mode. Stored as a Bevy resource.
/// The RL env and sim systems call these — they never know which mode is active.
pub trait GameMode: Send + Sync + 'static {
    /// Reward signal for the tick that just ran.
    fn reward(&self, world: &mut World, prev: &PrevModeState) -> f32;

    /// True if the episode should end before the tick limit.
    fn is_terminal(&self, world: &mut World) -> bool;

    /// Called once on every episode reset for mode-specific setup.
    fn on_reset(&self, _world: &mut World) {}

    /// Snapshot of whatever state the reward function needs this tick.
    fn snapshot(&self, world: &mut World) -> PrevModeState;
}

// ── ActiveGameMode resource ───────────────────────────────────────────────────

/// Type-erased game mode stored as a Bevy resource.
/// Built once at startup from GameModeKind, replaced on config change.
#[derive(Resource)]
pub struct ActiveGameMode(pub Box<dyn GameMode>);

impl ActiveGameMode {
    pub fn from_kind(kind: GameModeKind) -> Self {
        let mode: Box<dyn GameMode> = match kind {
            GameModeKind::GoldRush       => Box::new(GoldRush::default()),
            GameModeKind::TeamDeathmatch => Box::new(GoldRush::default()),
            GameModeKind::CaptureTheFlag => Box::new(GoldRush::default()),
            GameModeKind::Conquest       => Box::new(GoldRush::default()),
            GameModeKind::FreeForAll     => Box::new(GoldRush::default()),
        };
        Self(mode)
    }
}