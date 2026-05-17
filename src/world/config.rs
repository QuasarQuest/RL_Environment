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

// ── Default helpers for serde ─────────────────────────────────────────────────

fn default_melee_range()  -> i32 { config::MELEE_RANGE }
fn default_ranged_range() -> i32 { config::RANGED_RANGE }
fn default_kill_reward()  -> u32 { config::KILL_REWARD }
fn default_sim_speed()    -> f32       { config::DEFAULT_TICKS_PER_SECOND }
fn default_sim_speeds()   -> Vec<f32>  {
    vec![1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0]
}

// ── WorldConfig ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Clone, Resource)]
pub struct WorldConfig {
    pub width:  usize,
    pub height: usize,

    pub match_duration_ticks: u64,

    /// Starting sim speed in ticks/second.
    #[serde(default = "default_sim_speed")]
    pub sim_speed: f32,

    /// Available speed steps for the HUD +/- buttons and F/S keys.
    #[serde(default = "default_sim_speeds")]
    pub sim_speeds: Vec<f32>,

    #[serde(default = "default_melee_range")]
    pub melee_range:  i32,

    #[serde(default = "default_ranged_range")]
    pub ranged_range: i32,

    #[serde(default = "default_kill_reward")]
    pub kill_reward: u32,

    pub item_spawners:     Vec<ItemSpawnerRon>,
    pub fixed:             Vec<FixedTile>,
    pub agents:            Vec<AgentConfig>,
    pub obstacle_clusters: Vec<ObstacleCluster>,
}

impl WorldConfig {
    pub fn load(path: &str) -> Self {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|_| panic!("Cannot read map config: {path}"));
        ron::from_str(&text)
            .unwrap_or_else(|e| panic!("Cannot parse map config {path}: {e}"))
    }
}