// src/world/config.rs

use bevy::prelude::*;
use serde::Deserialize;
use crate::item::spawner::ItemSpawnConfig;
use crate::item::ItemKind;
use crate::agent::strategy::StrategyKind;
use crate::agent::planner::PlannerKind;
use crate::config;

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum TileKind { Free, Obstacle, Base, BaseRed, BaseBlue }

#[derive(Debug, Deserialize, Clone)]
pub struct FixedTile {
    pub x:    usize,
    pub y:    usize,
    pub tile: TileKind,
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub struct AgentConfig {
    pub x:        i32,
    pub y:        i32,
    pub strategy: StrategyKind,
    pub planner:  PlannerKind,
    #[serde(default)]
    pub team:     Option<u32>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum ObstacleKind { Block, Wall, Scatter }

#[derive(Debug, Deserialize, Clone)]
pub struct ObstacleCluster {
    pub kind:  ObstacleKind,
    pub count: usize,
    pub size:  (usize, usize),
}

#[derive(Debug, Deserialize, Clone)]
pub struct ItemSpawnerRon {
    pub kind:       String,
    pub max_on_map: usize,
    #[serde(default)]
    pub initial:    usize,
}

impl ItemSpawnerRon {
    pub fn to_config(&self) -> Option<ItemSpawnConfig> {
        let kind = match self.kind.as_str() {
            "Gold"       => ItemKind::Gold,
            "Health"     => ItemKind::Health,
            "Ammo"       => ItemKind::Ammo,
            "SpeedBoost" => ItemKind::SpeedBoost,
            _            => return None,
        };
        Some(ItemSpawnConfig { kind, max_on_map: self.max_on_map })
    }
}

// ── Serde defaults ────────────────────────────────────────────────────────────

fn default_sim_speed()             -> f32 { config::DEFAULT_TICKS_PER_SECOND }
fn default_sim_speeds()            -> Vec<f32> {
    vec![1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0,
         500.0, 1000.0, 2500.0, 5000.0, 10000.0]
}
fn default_melee_range()           -> i32 { config::MELEE_RANGE }
fn default_ranged_range()          -> i32 { config::RANGED_RANGE }
fn default_kill_reward()           -> u8  { config::KILL_REWARD }
fn default_melee_damage()          -> u8  { config::MELEE_DAMAGE }
fn default_ranged_damage()         -> u8  { config::RANGED_DAMAGE }
fn default_melee_cooldown_ticks()  -> u8  { config::MELEE_COOLDOWN_TICKS }
fn default_ranged_cooldown_ticks() -> u8  { config::RANGED_COOLDOWN_TICKS }
fn default_respawn_ticks()         -> u8  { config::AGENT_RESPAWN_TICKS }
fn default_gold_carry_speed()      -> f32 { config::GOLD_CARRY_SPEED }
fn default_base_safe_radius()      -> u8  { config::BASE_SAFE_RADIUS }

// ── WorldConfig ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Clone, Resource)]
pub struct WorldConfig {
    pub width:  usize,
    pub height: usize,

    pub match_duration_ticks: u64,

    #[serde(default = "default_sim_speed")]
    pub sim_speed:  f32,
    #[serde(default = "default_sim_speeds")]
    pub sim_speeds: Vec<f32>,

    #[serde(default = "default_respawn_ticks")]
    pub respawn_ticks:    u8,
    #[serde(default = "default_gold_carry_speed")]
    pub gold_carry_speed: f32,
    #[serde(default = "default_base_safe_radius")]
    pub base_safe_radius: u8,

    #[serde(default = "default_melee_range")]
    pub melee_range:  i32,
    #[serde(default = "default_ranged_range")]
    pub ranged_range: i32,
    #[serde(default = "default_kill_reward")]
    pub kill_reward:  u8,

    #[serde(default = "default_melee_damage")]
    pub melee_damage:  u8,
    #[serde(default = "default_ranged_damage")]
    pub ranged_damage: u8,

    #[serde(default = "default_melee_cooldown_ticks")]
    pub melee_cooldown_ticks:  u8,
    #[serde(default = "default_ranged_cooldown_ticks")]
    pub ranged_cooldown_ticks: u8,

    pub item_spawners:     Vec<ItemSpawnerRon>,
    pub fixed:             Vec<FixedTile>,
    pub agents:            Vec<AgentConfig>,
    pub obstacle_clusters: Vec<ObstacleCluster>,
}

impl WorldConfig {
    pub fn load(path: &str) -> Self {
        let resolved = Self::resolve_path(path);
        let text = std::fs::read_to_string(&resolved)
            .unwrap_or_else(|_| panic!("Cannot read map config: {resolved}"));
        ron::from_str(&text)
            .unwrap_or_else(|e| panic!("Cannot parse map config {resolved}: {e}"))
    }

    /// Walk up from the current working directory until we find `path`.
    /// This allows the sim to be invoked from any subdirectory of the
    /// project (e.g. `rl/src/` when running from PyCharm).
    fn resolve_path(path: &str) -> String {
        let mut dir = std::env::current_dir()
            .expect("Cannot determine current directory");

        loop {
            let candidate = dir.join(path);
            if candidate.exists() {
                return candidate.to_string_lossy().into_owned();
            }
            if !dir.pop() {
                // Reached filesystem root without finding the file.
                // Fall back to the original path so the error message
                // names the file the user expects.
                return path.to_owned();
            }
        }
    }
}