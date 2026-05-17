// src/world/config

use bevy::prelude::*;
use serde::Deserialize;
use crate::item::spawner::ItemSpawnConfig;
use crate::item::ItemKind;
use crate::agent::strategy::StrategyKind;
use crate::agent::planner::PlannerKind;

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

#[derive(Debug, Deserialize, Clone, Resource)]
pub struct MapConfig {
    pub width:  usize,
    pub height: usize,

    /// Episode length in sim-ticks — loaded from the map asset.
    pub match_duration_ticks: u64,

    pub item_spawners:     Vec<ItemSpawnerRon>,
    pub fixed:             Vec<FixedTile>,
    pub agents:            Vec<AgentConfig>,
    pub obstacle_clusters: Vec<ObstacleCluster>,
}

impl MapConfig {
    pub fn load(path: &str) -> Self {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|_| panic!("Cannot read map config: {path}"));
        ron::from_str(&text)
            .unwrap_or_else(|e| panic!("Cannot parse map config {path}: {e}"))
    }
}