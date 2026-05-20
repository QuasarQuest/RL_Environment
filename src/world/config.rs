// src/world/config.rs
//
// WorldConfig — declarative map description.
//
// Design philosophy:
//   Nothing in this file contains absolute pixel or tile coordinates.
//   Everything is expressed as fractions of the grid dimensions so that
//   a single config.ron works correctly at any map size.
//
//   Layout resolution (fraction → tile coordinate) lives entirely in
//   world/layout.rs. This file is pure data.
//
// Coordinate convention:
//   (0,0) = bottom-left. x grows right, y grows up.
//   "TopRight corner" means high x, high y.

use bevy::prelude::*;
use serde::Deserialize;
use std::collections::HashSet;
use crate::agent::strategy::StrategyKind;
use crate::agent::planner::PlannerKind;
use crate::sim::mode::GameModeKind;
use crate::item::ItemKind;
use crate::item::spawner::ItemSpawnConfig;
use crate::config as global;

// ── Defaults ──────────────────────────────────────────────────────────────────
// All magic numbers live here. Serde default fns below are thin wrappers only.

const DEFAULT_BASE_MARGIN:        f32 = 0.08;
const DEFAULT_SPAWN_OFFSET:       f32 = 0.05;
const DEFAULT_SAFE_ZONE_FRACTION: f32 = 0.06;
const DEFAULT_OBSTACLE_DENSITY:   f32 = 0.08;
const DEFAULT_MAX_BLOCK_FRACTION: f32 = 0.12;
const DEFAULT_MAX_WALL_FRACTION:  f32 = 0.14;
const DEFAULT_GOLD_DENSITY:       f32 = 0.8;

// ── Corner ────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum Corner {
    BottomLeft,
    BottomRight,
    TopLeft,
    TopRight,
}

impl Corner {
    /// Resolve to a tile coordinate given grid dimensions and an inset margin.
    /// `margin` is a fraction of the shorter grid dimension.
    pub fn resolve(self, width: usize, height: usize, margin: f32) -> (i32, i32) {
        let inset = ((width.min(height) as f32) * margin).round() as i32;
        let inset = inset.max(1);
        let max_x = width  as i32 - 1;
        let max_y = height as i32 - 1;
        match self {
            Corner::BottomLeft  => (inset,         inset),
            Corner::BottomRight => (max_x - inset, inset),
            Corner::TopLeft     => (inset,         max_y - inset),
            Corner::TopRight    => (max_x - inset, max_y - inset),
        }
    }
}

// ── Base placement ────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Clone)]
pub struct BaseConfig {
    pub team:   u8,
    pub corner: Corner,
    /// Fraction of the shorter grid dimension used as inset from the corner.
    #[serde(default = "default_base_margin")]
    pub margin: f32,
}

// ── Agent spawn intent ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum SpawnIntent {
    NearBase,
    Centre,
}

impl Default for SpawnIntent {
    fn default() -> Self { SpawnIntent::NearBase }
}

#[derive(Debug, Deserialize, Clone)]
pub struct AgentConfig {
    pub team:         u8,
    pub strategy:     StrategyKind,
    pub planner:      PlannerKind,
    #[serde(default)]
    pub spawn:        SpawnIntent,
    /// Distance from base as fraction of the shorter grid dimension.
    #[serde(default = "default_spawn_offset")]
    pub spawn_offset: f32,
}

// ── Obstacle generation ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum ObstacleProfile {
    Blocks,
    Walls,
    Scatter,
    Mixed,
    None,
}

impl Default for ObstacleProfile {
    fn default() -> Self { ObstacleProfile::Mixed }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ObstacleConfig {
    /// Fraction of total grid tiles that become obstacles. Clamped to [0, 0.4].
    #[serde(default = "default_obstacle_density")]
    pub density: f32,
    #[serde(default)]
    pub profile: ObstacleProfile,
    /// Max block size as fraction of shorter grid dimension.
    #[serde(default = "default_max_block_fraction")]
    pub max_block_fraction: f32,
    /// Max wall length as fraction of shorter grid dimension.
    #[serde(default = "default_max_wall_fraction")]
    pub max_wall_fraction: f32,
}

impl Default for ObstacleConfig {
    fn default() -> Self {
        Self {
            density:            DEFAULT_OBSTACLE_DENSITY,
            profile:            ObstacleProfile::Mixed,
            max_block_fraction: DEFAULT_MAX_BLOCK_FRACTION,
            max_wall_fraction:  DEFAULT_MAX_WALL_FRACTION,
        }
    }
}

// ── Item density ──────────────────────────────────────────────────────────────

/// Item spawn densities expressed as items per 100 free tiles.
/// "0.5 gold per 100 tiles" is immediately meaningful without knowing map size.
/// Conversion to absolute counts happens in `to_spawn_configs`.
#[derive(Debug, Deserialize, Clone)]
pub struct ItemDensityConfig {
    /// Gold items per 100 free tiles.
    #[serde(default = "default_gold_density")]
    pub gold: f32,
    #[serde(default)]
    pub health: f32,
    #[serde(default)]
    pub ammo: f32,
    #[serde(default)]
    pub speed_boost: f32,
}

impl Default for ItemDensityConfig {
    fn default() -> Self {
        Self {
            gold:        DEFAULT_GOLD_DENSITY,
            health:      0.0,
            ammo:        0.0,
            speed_boost: 0.0,
        }
    }
}

impl ItemDensityConfig {
    pub fn to_spawn_configs(&self, free_tiles: usize) -> Vec<ItemSpawnConfig> {
        let mut out = Vec::new();
        let mut push = |kind: ItemKind, per_hundred: f32| {
            let initial = ((free_tiles as f32) * per_hundred / 100.0).round() as usize;
            if initial > 0 {
                out.push(ItemSpawnConfig { kind, max_on_map: (initial * 2).max(1) });
            }
        };
        push(ItemKind::Gold,       self.gold);
        push(ItemKind::Health,     self.health);
        push(ItemKind::Ammo,       self.ammo);
        push(ItemKind::SpeedBoost, self.speed_boost);
        out
    }
}

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

    pub bases:  Vec<BaseConfig>,
    pub agents: Vec<AgentConfig>,

    #[serde(default = "default_safe_zone_fraction")]
    pub safe_zone_fraction: f32,

    #[serde(default)]
    pub obstacles: ObstacleConfig,

    #[serde(default)]
    pub item_density: ItemDensityConfig,

    /// Which game mode to run. Determines reward, terminal, and win conditions.
    #[serde(default)]
    pub game_mode: GameModeKind,

    #[serde(default = "default_respawn_ticks")]
    pub respawn_ticks: u8,
    #[serde(default = "default_gold_carry_speed")]
    pub gold_carry_speed: f32,

    #[serde(default = "default_melee_range")]
    pub melee_range: u8,
    #[serde(default = "default_ranged_range")]
    pub ranged_range: u8,
    #[serde(default = "default_kill_reward")]
    pub kill_reward: u8,
    #[serde(default = "default_melee_damage")]
    pub melee_damage: u8,
    #[serde(default = "default_ranged_damage")]
    pub ranged_damage: u8,
    #[serde(default = "default_melee_cooldown_ticks")]
    pub melee_cooldown_ticks: u8,
    #[serde(default = "default_ranged_cooldown_ticks")]
    pub ranged_cooldown_ticks: u8,
}

// ── Serde default fns — thin wrappers around constants and global config ──────
// Serde requires fn() -> T, so we cannot use constants directly.

fn default_base_margin()           -> f32      { DEFAULT_BASE_MARGIN }
fn default_spawn_offset()          -> f32      { DEFAULT_SPAWN_OFFSET }
fn default_safe_zone_fraction()    -> f32      { DEFAULT_SAFE_ZONE_FRACTION }
fn default_obstacle_density()      -> f32      { DEFAULT_OBSTACLE_DENSITY }
fn default_max_block_fraction()    -> f32      { DEFAULT_MAX_BLOCK_FRACTION }
fn default_max_wall_fraction()     -> f32      { DEFAULT_MAX_WALL_FRACTION }
fn default_gold_density()          -> f32      { DEFAULT_GOLD_DENSITY }
fn default_sim_speed()             -> f32      { global::DEFAULT_TICKS_PER_SECOND }
fn default_sim_speeds()            -> Vec<f32> {
    vec![1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0,
         500.0, 1000.0, 2500.0, 5000.0, 10000.0]
}
fn default_respawn_ticks()         -> u8  { global::AGENT_RESPAWN_TICKS }
fn default_gold_carry_speed()      -> f32 { global::GOLD_CARRY_SPEED }
fn default_melee_range()           -> u8  { global::MELEE_RANGE }
fn default_ranged_range()          -> u8  { global::RANGED_RANGE }
fn default_kill_reward()           -> u8  { global::KILL_REWARD }
fn default_melee_damage()          -> u8  { global::MELEE_DAMAGE }
fn default_ranged_damage()         -> u8  { global::RANGED_DAMAGE }
fn default_melee_cooldown_ticks()  -> u8  { global::MELEE_COOLDOWN_TICKS }
fn default_ranged_cooldown_ticks() -> u8  { global::RANGED_COOLDOWN_TICKS }

// ── WorldConfig methods ───────────────────────────────────────────────────────

impl WorldConfig {
    pub fn load(path: &str) -> Self {
        let resolved = Self::resolve_path(path);
        let text = std::fs::read_to_string(&resolved)
            .unwrap_or_else(|e| panic!("Cannot read config '{resolved}': {e}"));
        let cfg: Self = ron::from_str(&text)
            .unwrap_or_else(|e| panic!("Cannot parse config '{resolved}': {e}"));
        cfg.validate();
        cfg
    }

    fn validate(&self) {
        let mut errors: Vec<String> = Vec::new();

        // ── Grid ──────────────────────────────────────────────────────────────
        if self.width < 4 {
            errors.push(format!("width must be ≥ 4 (got {})", self.width));
        }
        if self.height < 4 {
            errors.push(format!("height must be ≥ 4 (got {})", self.height));
        }

        // ── Bases ─────────────────────────────────────────────────────────────
        if self.bases.is_empty() {
            errors.push("at least one base is required".into());
        }
        for base in &self.bases {
            if base.margin <= 0.0 || base.margin >= 0.5 {
                errors.push(format!(
                    "base team {} margin {:.3} must be in (0.0, 0.5)",
                    base.team, base.margin
                ));
            }
        }

        // ── Agents ────────────────────────────────────────────────────────────
        if self.agents.is_empty() {
            errors.push("at least one agent is required".into());
        }
        let base_teams: HashSet<u8> = self.bases.iter().map(|b| b.team).collect();
        for agent in &self.agents {
            if !base_teams.contains(&agent.team) {
                errors.push(format!(
                    "agent on team {} has no matching base (bases defined for teams: {:?})",
                    agent.team,
                    { let mut v: Vec<u8> = base_teams.iter().copied().collect(); v.sort(); v }
                ));
            }
        }

        // ── Obstacles ─────────────────────────────────────────────────────────
        if self.obstacles.density > 0.4 {
            errors.push(format!(
                "obstacle density {:.3} exceeds maximum 0.4 — reduce it",
                self.obstacles.density
            ));
        }

        // ── Sim speeds ────────────────────────────────────────────────────────
        if self.sim_speeds.is_empty() {
            errors.push("sim_speeds must not be empty".into());
        }
        if !self.sim_speeds.contains(&self.sim_speed) {
            errors.push(format!(
                "sim_speed {:.1} is not in sim_speeds {:?}",
                self.sim_speed, self.sim_speeds
            ));
        }

        // ── Game mode constraints ─────────────────────────────────────────────
        // Each mode validates that the config supplies what it needs.
        self.validate_for_game_mode(&mut errors);

        // ── Report ────────────────────────────────────────────────────────────
        if !errors.is_empty() {
            eprintln!("\n╔══ Config validation failed ({} error(s)) ══", errors.len());
            for (i, e) in errors.iter().enumerate() {
                eprintln!("║  [{}] {}", i + 1, e);
            }
            eprintln!("╚══ Fix the errors above in config.ron\n");
            panic!("Invalid config — see errors above.");
        }
    }

    /// Mode-specific constraints. Add a match arm here when adding a new mode.
    fn validate_for_game_mode(&self, errors: &mut Vec<String>) {
        let agent_teams: HashSet<u8> = self.agents.iter().map(|a| a.team).collect();

        match self.game_mode {
            GameModeKind::GoldRush => {
                if self.item_density.gold <= 0.0 {
                    errors.push(
                        "GoldRush requires item_density.gold > 0.0".into()
                    );
                }
            }
            GameModeKind::TeamDeathmatch => {
                if agent_teams.len() < 2 {
                    errors.push(
                        "TeamDeathmatch requires agents on at least 2 different teams".into()
                    );
                }
            }
            GameModeKind::CaptureTheFlag => {
                if agent_teams.len() < 2 {
                    errors.push(
                        "CaptureTheFlag requires agents on at least 2 different teams".into()
                    );
                }
            }
            GameModeKind::Conquest => {
                if agent_teams.len() < 2 {
                    errors.push(
                        "Conquest requires agents on at least 2 different teams".into()
                    );
                }
            }
            GameModeKind::FreeForAll => {
                if self.agents.len() < 2 {
                    errors.push(
                        "FreeForAll requires at least 2 agents".into()
                    );
                }
            }
        }
    }

    fn resolve_path(path: &str) -> String {
        let mut dir = std::env::current_dir()
            .expect("Cannot determine current directory");
        loop {
            let candidate = dir.join(path);
            if candidate.exists() {
                return candidate.to_string_lossy().into_owned();
            }
            if !dir.pop() {
                return path.to_owned();
            }
        }
    }

    pub fn diagonal(&self) -> f32 {
        let w = self.width  as f32;
        let h = self.height as f32;
        (w * w + h * h).sqrt()
    }

    pub fn short_side(&self) -> usize {
        self.width.min(self.height)
    }

    pub fn safe_zone_radius(&self) -> i32 {
        ((self.diagonal() * self.safe_zone_fraction).round() as i32).max(1)
    }
}