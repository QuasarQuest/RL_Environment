// src/agent/components.rs
//
// Pure simulation components. No display, no viz, no factory concerns.
// AgentLabel, AgentInfo, HideViz live in src/viz/components.rs.

use bevy::prelude::*;
use crate::config;

pub use crate::world::coords::GridPos;

// ── Gold ──────────────────────────────────────────────────────────────────────

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct GoldCarried(pub u32);

impl GoldCarried {
    pub fn is_full(self)  -> bool { self.0 >= config::AGENT_MAX_GOLD }
    pub fn is_empty(self) -> bool { self.0 == 0 }
}

// ── Score ─────────────────────────────────────────────────────────────────────

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Score(pub u32);

// ── Hearts ────────────────────────────────────────────────────────────────────

#[derive(Component, Clone, Copy, Debug)]
pub struct Hearts(pub u8);

impl Default for Hearts {
    fn default() -> Self { Self(config::AGENT_MAX_HEARTS) }
}

impl Hearts {
    pub fn is_dead(self)  -> bool { self.0 == 0 }
    pub fn is_full(self)  -> bool { self.0 >= config::AGENT_MAX_HEARTS }
    pub fn lose_one(&mut self) { self.0 = self.0.saturating_sub(1); }
    pub fn gain_one(&mut self) { self.0 = (self.0 + 1).min(config::AGENT_MAX_HEARTS); }
}

// ── Ammo ──────────────────────────────────────────────────────────────────────

#[derive(Component, Clone, Copy, Debug)]
pub struct Ammo(pub u8);

impl Default for Ammo {
    fn default() -> Self { Self(config::AGENT_START_AMMO) }
}

impl Ammo {
    pub fn is_empty(self) -> bool { self.0 == 0 }
    pub fn is_full(self)  -> bool { self.0 >= config::AGENT_MAX_AMMO }
    pub fn consume(&mut self) -> bool {
        if self.0 > 0 { self.0 -= 1; true } else { false }
    }
    pub fn add(&mut self, n: u8) {
        self.0 = (self.0 + n).min(config::AGENT_MAX_AMMO);
    }
}

// ── Kill / Death counters ─────────────────────────────────────────────────────

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct KillCount(pub u32);

#[derive(Component, Clone, Copy, Debug, Default)]
pub struct DeathCount(pub u32);

// ── SpeedBuff ─────────────────────────────────────────────────────────────────

#[derive(Component, Clone, Copy, Debug)]
pub struct SpeedBuff(pub u8);

// ── RespawnIn ─────────────────────────────────────────────────────────────────

#[derive(Component, Clone, Copy, Debug)]
pub struct RespawnIn(pub u8);

// ── SpawnPoint ────────────────────────────────────────────────────────────────

#[derive(Component, Clone, Copy, Debug)]
pub struct SpawnPoint(pub GridPos);