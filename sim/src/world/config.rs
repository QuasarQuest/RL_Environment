// src/world/config.rs
//
// WorldConfig — declarative map description loaded from config.ron.
// All coordinates are fractions of grid dimensions; layout.rs resolves them
// to tile coords. Unknown fields in the RON file are silently ignored.

use serde::Deserialize;
use std::collections::HashSet;
use crate::entity::item::{ItemKind, ItemSpawnConfig};
use crate::config as global;

// ── Defaults ──────────────────────────────────────────────────────────────────

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
    pub team: u8,
    #[serde(default)]
    pub spawn: SpawnIntent,
    #[serde(default = "default_spawn_offset")]
    pub spawn_offset: f32,
    /// For non-RL agents (team != 0): scripted behaviour used in stages 4+.
    /// Ignored for team 0 (always RL-controlled).
    #[serde(default)]
    pub enemy_kind: EnemyKind,
    // strategy and planner fields in config.ron are ignored — handled by SimCore
}

// ── Enemy kind ────────────────────────────────────────────────────────────────

/// Scripted behaviour for non-RL agents.  `None` means the agent stands still.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnemyKind {
    #[default]
    None,
    /// Greedy direction: moves one step toward the nearest gold (or base),
    /// no pathfinding — easy to trap against walls.
    SimpleChaser,
    /// A* pathfinding — finds optimal routes; acts as the hard enemy in s5+.
    BehaviorTree,
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

/// Explicit cluster-based obstacle descriptor (old-style config).
/// `size` is (width, height) for Block/Wall; ignored for Scatter.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum ObstacleKind { Block, Wall, Scatter }

#[derive(Debug, Deserialize, Clone)]
pub struct ObstacleCluster {
    pub kind:  ObstacleKind,
    pub count: usize,
    pub size:  (usize, usize),
}

#[derive(Debug, Deserialize, Clone)]
pub struct ObstacleConfig {
    #[serde(default = "default_obstacle_density")]
    pub density: f32,
    #[serde(default)]
    pub profile: ObstacleProfile,
    #[serde(default = "default_max_block_fraction")]
    pub max_block_fraction: f32,
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

#[derive(Debug, Deserialize, Clone)]
pub struct ItemDensityConfig {
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
        Self { gold: DEFAULT_GOLD_DENSITY, health: 0.0, ammo: 0.0, speed_boost: 0.0 }
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

// ── Reward config ─────────────────────────────────────────────────────────────

/// Per-stage reward weights.  Serde defaults match the stage-1 constants in
/// `rl/reward.rs` so omitting this section from a RON file is always safe.
#[derive(Debug, Deserialize, Clone)]
pub struct RewardConfig {
    #[serde(default = "default_reward_tick")]
    pub tick: f32,
    #[serde(default = "default_reward_pickup")]
    pub pickup: f32,
    #[serde(default = "default_reward_deposit")]
    pub deposit: f32,
    #[serde(default = "default_reward_approach")]
    pub approach: f32,
    /// Small penalty when a Move action is blocked by a wall (no position change).
    #[serde(default)]
    pub wall_hit: f32,
    /// Reward on killing an enemy agent.  0 in stages 1–5.
    #[serde(default)]
    pub kill: f32,
    /// Reward applied once on the tick the agent dies.  Added directly to the
    /// step reward, so it must be stored with its sign: use a NEGATIVE value to
    /// penalise death (e.g. -2.0).  0 in stages 1–5.
    #[serde(default)]
    pub death_penalty: f32,
    /// Discount used by the potential-based approach shaping (F = γΦ(s') − Φ(s)).
    /// Must match the PPO `gamma` so the shaping stays policy-invariant.
    #[serde(default = "default_shaping_gamma")]
    pub shaping_gamma: f32,
}

impl RewardConfig {
    pub const DEFAULT_TICK:          f32 = -0.0005;
    pub const DEFAULT_PICKUP:        f32 =  0.5;
    pub const DEFAULT_DEPOSIT:       f32 =  5.0;
    pub const DEFAULT_APPROACH:      f32 =  0.05;
    pub const DEFAULT_SHAPING_GAMMA: f32 =  0.99;
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            tick:          Self::DEFAULT_TICK,
            pickup:        Self::DEFAULT_PICKUP,
            deposit:       Self::DEFAULT_DEPOSIT,
            approach:      Self::DEFAULT_APPROACH,
            wall_hit:      0.0,
            kill:          0.0,
            death_penalty: 0.0,
            shaping_gamma: Self::DEFAULT_SHAPING_GAMMA,
        }
    }
}

// ── WorldConfig ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Clone)]
pub struct WorldConfig {
    pub width:  usize,
    pub height: usize,
    pub match_duration_ticks: u64,

    pub bases:  Vec<BaseConfig>,
    pub agents: Vec<AgentConfig>,

    #[serde(default = "default_safe_zone_fraction")]
    pub safe_zone_fraction: f32,
    #[serde(default)]
    pub obstacles: ObstacleConfig,
    /// If non-empty, cluster-based placement is used instead of `obstacles`.
    #[serde(default)]
    pub obstacle_clusters: Vec<ObstacleCluster>,
    #[serde(default)]
    pub item_density: ItemDensityConfig,
    #[serde(default = "default_gold_carry_speed")]
    pub gold_carry_speed: f32,
    #[serde(default)]
    pub reward: RewardConfig,

    // ── Combat ────────────────────────────────────────────────────────────────
    #[serde(default = "default_melee_range")]
    pub melee_range: u8,
    #[serde(default = "default_ranged_range")]
    pub ranged_range: u8,
    #[serde(default = "default_melee_damage")]
    pub melee_damage: u8,
    #[serde(default = "default_ranged_damage")]
    pub ranged_damage: u8,
    #[serde(default = "default_melee_cooldown_ticks")]
    pub melee_cooldown_ticks: u8,
    #[serde(default = "default_ranged_cooldown_ticks")]
    pub ranged_cooldown_ticks: u8,
    #[serde(default = "default_respawn_ticks")]
    pub respawn_ticks: u8,
}

// ── Serde default fns ────────────────────────────────────────────────────────

fn default_base_margin()        -> f32 { DEFAULT_BASE_MARGIN }
fn default_spawn_offset()       -> f32 { DEFAULT_SPAWN_OFFSET }
fn default_safe_zone_fraction() -> f32 { DEFAULT_SAFE_ZONE_FRACTION }
fn default_obstacle_density()   -> f32 { DEFAULT_OBSTACLE_DENSITY }
fn default_max_block_fraction() -> f32 { DEFAULT_MAX_BLOCK_FRACTION }
fn default_max_wall_fraction()  -> f32 { DEFAULT_MAX_WALL_FRACTION }
fn default_gold_density()       -> f32 { DEFAULT_GOLD_DENSITY }
fn default_gold_carry_speed()       -> f32 { global::GOLD_CARRY_SPEED }
fn default_reward_tick()            -> f32 { RewardConfig::DEFAULT_TICK }
fn default_reward_pickup()          -> f32 { RewardConfig::DEFAULT_PICKUP }
fn default_reward_deposit()         -> f32 { RewardConfig::DEFAULT_DEPOSIT }
fn default_reward_approach()        -> f32 { RewardConfig::DEFAULT_APPROACH }
fn default_shaping_gamma()          -> f32 { RewardConfig::DEFAULT_SHAPING_GAMMA }
fn default_melee_range()            -> u8  { global::MELEE_RANGE }
fn default_ranged_range()           -> u8  { global::RANGED_RANGE }
fn default_melee_damage()           -> u8  { global::MELEE_DAMAGE }
fn default_ranged_damage()          -> u8  { global::RANGED_DAMAGE }
fn default_melee_cooldown_ticks()   -> u8  { global::MELEE_COOLDOWN_TICKS }
fn default_ranged_cooldown_ticks()  -> u8  { global::RANGED_COOLDOWN_TICKS }
fn default_respawn_ticks()          -> u8  { global::AGENT_RESPAWN_TICKS }

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

        if self.width < 4  { errors.push(format!("width must be ≥ 4 (got {})",  self.width));  }
        if self.height < 4 { errors.push(format!("height must be ≥ 4 (got {})", self.height)); }

        if self.bases.is_empty()  { errors.push("at least one base is required".into()); }
        if self.agents.is_empty() { errors.push("at least one agent is required".into()); }

        let base_teams: HashSet<u8> = self.bases.iter().map(|b| b.team).collect();
        for agent in &self.agents {
            if !base_teams.contains(&agent.team) {
                errors.push(format!(
                    "agent on team {} has no matching base",
                    agent.team
                ));
            }
        }

        for base in &self.bases {
            if base.margin <= 0.0 || base.margin >= 0.5 {
                errors.push(format!(
                    "base team {} margin {:.3} must be in (0.0, 0.5)",
                    base.team, base.margin
                ));
            }
        }

        if self.obstacles.density > 0.4 {
            errors.push(format!(
                "obstacle density {:.3} exceeds maximum 0.4",
                self.obstacles.density
            ));
        }

        if !errors.is_empty() {
            eprintln!("\n╔══ Config validation failed ({} error(s)) ══", errors.len());
            for (i, e) in errors.iter().enumerate() {
                eprintln!("║  [{}] {}", i + 1, e);
            }
            eprintln!("╚══ Fix the errors above in config.ron\n");
            panic!("Invalid config — see errors above.");
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
