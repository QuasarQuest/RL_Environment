// src/world/config.rs
//
// WorldConfig — declarative map description loaded from config.ron.
// Single-agent gold rush: no teams, no enemies, no combat. The base position is
// randomised per episode inside a fixed spawn pocket (see engine/builder.rs), so
// the config no longer declares base corners or agents.
// Unknown fields in the RON file are silently ignored (serde default behaviour),
// so viewer-only keys like `game_mode` / `sim_speed` are harmless here.

use serde::Deserialize;
use crate::entity::item::{ItemKind, ItemSpawnConfig};

// ── Defaults ──────────────────────────────────────────────────────────────────

const DEFAULT_OBSTACLE_DENSITY:   f32 = 0.08;
const DEFAULT_MAX_BLOCK_FRACTION: f32 = 0.12;
const DEFAULT_MAX_WALL_FRACTION:  f32 = 0.14;
const DEFAULT_GOLD_DENSITY:       f32 = 0.8;

/// Half-width of the square spawn pocket carved around the (randomised) base.
/// 3 → a 7×7 pocket; kept ≥1 tile from every border (see builder.rs).
pub const SPAWN_POCKET_RADIUS: i32 = 3;

// ── Obstacle generation ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
pub enum ObstacleProfile {
    Blocks,
    Walls,
    Scatter,
    #[default]
    Mixed,
    None,
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
#[serde(default)]
pub struct ObstacleConfig {
    pub density: f32,
    pub profile: ObstacleProfile,
    pub max_block_fraction: f32,
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
//
// Per-kind spawn density, expressed as items per 100 free tiles. Gold is the
// objective; the speed boost and the multiplier are the gold-rush flavour items.

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct ItemDensityConfig {
    pub gold: f32,
    pub speed: f32,
    pub multiplier: f32,
    /// Omit (or `clustered_fraction: 0`) for legacy uniform scatter; otherwise gold
    /// forms small clumps plus sparse singletons. Total gold COUNT is unchanged.
    pub gold_clustering: GoldClusterConfig,
}

impl Default for ItemDensityConfig {
    fn default() -> Self {
        Self {
            gold: DEFAULT_GOLD_DENSITY,
            speed: 0.0,
            multiplier: 0.0,
            gold_clustering: GoldClusterConfig::default(),
        }
    }
}

/// Spatial grouping for gold (objective pickups). A `clustered_fraction` of the gold is
/// placed in clumps of `clump_size.0..=clump_size.1` tiles within Chebyshev `radius` of
/// a clump centre; the remainder is scattered as lone singletons. Runtime respawns reuse
/// the same radius to keep new gold near existing gold so clumps don't erode to uniform
/// as the agent eats them. `clustered_fraction == 0` ⇒ legacy uniform scatter.
#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct GoldClusterConfig {
    pub clustered_fraction: f32,
    pub clump_size: (usize, usize),
    pub radius: i32,
}

impl Default for GoldClusterConfig {
    fn default() -> Self {
        Self { clustered_fraction: 0.0, clump_size: (2, 3), radius: 1 }
    }
}

impl GoldClusterConfig {
    /// Whether clustered placement is active (false ⇒ legacy uniform scatter).
    pub fn enabled(&self) -> bool { self.clustered_fraction > 0.0 }
    /// Clamped, sane (min, max) clump size — min ≥ 1 and max ≥ min.
    pub fn clamped_clump(&self) -> (usize, usize) {
        let lo = self.clump_size.0.max(1);
        (lo, self.clump_size.1.max(lo))
    }
}

impl ItemDensityConfig {
    pub fn to_spawn_configs(&self, free_tiles: usize) -> Vec<ItemSpawnConfig> {
        [(ItemKind::Gold, self.gold), (ItemKind::Speed, self.speed), (ItemKind::Multiplier, self.multiplier)]
            .into_iter()
            .filter_map(|(kind, per_hundred)| {
                let initial = (free_tiles as f32 * per_hundred / 100.0).round() as usize;
                (initial > 0).then(|| ItemSpawnConfig { kind, max_on_map: (initial * 2).max(1) })
            })
            .collect()
    }
}

// ── Reward config ─────────────────────────────────────────────────────────────

/// Per-stage reward weights. Serde defaults match the stage-1 constants below so
/// omitting this section from a RON file is always safe.
#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct RewardConfig {
    pub tick: f32,
    pub pickup: f32,
    pub deposit: f32,
    pub wall_hit: f32,
    /// Per-tick discount applied WITHIN a navigation option: the option's return is
    /// Σ γᵗ rₜ over the ticks it ran (semi-MDP option return). This is what makes a
    /// nearer reward worth more than a farther one. Must match the PPO `gamma` so the
    /// intra-option discounting is consistent with the learner's cross-step discount.
    pub option_gamma: f32,
}

impl RewardConfig {
    pub const DEFAULT_TICK:           f32 = -0.0005;
    pub const DEFAULT_PICKUP:         f32 =  0.5;
    pub const DEFAULT_DEPOSIT:        f32 =  5.0;
    pub const DEFAULT_OPTION_GAMMA:   f32 =  0.99;
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            tick:           Self::DEFAULT_TICK,
            pickup:         Self::DEFAULT_PICKUP,
            deposit:        Self::DEFAULT_DEPOSIT,
            wall_hit:       0.0,
            option_gamma:   Self::DEFAULT_OPTION_GAMMA,
        }
    }
}

// ── WorldConfig ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Clone)]
pub struct WorldConfig {
    pub width:  usize,
    pub height: usize,
    pub match_duration_ticks: u64,

    #[serde(default)]
    pub obstacles: ObstacleConfig,
    /// If non-empty, cluster-based placement is used instead of `obstacles`.
    #[serde(default)]
    pub obstacle_clusters: Vec<ObstacleCluster>,
    #[serde(default)]
    pub item_density: ItemDensityConfig,
    #[serde(default)]
    pub reward: RewardConfig,
}

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

        // The base sits in a SPAWN_POCKET_RADIUS pocket kept ≥1 tile from the
        // border, so the map must be wide/tall enough to hold it with margin.
        let min_dim = (SPAWN_POCKET_RADIUS * 2 + 3) as usize; // pocket + border + slack
        if self.width  < min_dim { errors.push(format!("width must be ≥ {min_dim} (got {})",  self.width));  }
        if self.height < min_dim { errors.push(format!("height must be ≥ {min_dim} (got {})", self.height)); }

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

    pub fn short_side(&self) -> usize {
        self.width.min(self.height)
    }
}
