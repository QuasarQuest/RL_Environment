// src/engine/mod.rs
//
// Simulation core — one independent episode stream for the single RL agent
// (single-agent gold rush: no enemies, no combat).
//
// Per-tick step order (`tick_once`):
//   tick_buffs (speed/slow/multiplier decay)
//   → find_clusters (gold clustering for this tick)
//   → resolve RL action → A* navigate one step (or Wait)
//   → pickup → auto_deposit → spawner
//   → rebuild gold_positions
//   → compute reward
//
// Two public entry points wrap `tick_once`:
//   - `step`        : one tick + observation (viewer path — one tile/frame).
//   - `step_option` : the RL training/eval path. A navigation goal is committed
//                     and executed via A* until a reward-relevant event
//                     (pickup/deposit), arrival/blockage, episode end, or
//                     MAX_OPTION_TICKS — a semi-MDP over high-level decisions.
//
// The base is randomised per episode (engine/builder.rs); reset() regenerates the
// whole map (base + obstacles + items) from the env RNG, so episodes vary while
// staying seed-reproducible.

pub mod builder;
pub mod clusters;
pub mod nav;
pub mod obs;
pub mod physics;
pub mod pickup;
pub mod spawner;

pub use crate::entity::{AgentState, ItemState};

use rand::SeedableRng;
use rand::rngs::{SmallRng, SysRng};

use crate::config::AGENT_MAX_GOLD;
use crate::entity::agent::Action;
use crate::entity::item::ItemKind;
use crate::rl::action::{
    int_to_rl_action, RlAction, ACTION_BASE, ACTION_MULT, ACTION_SIZE, ACTION_SPEED, ACTION_WAIT,
    CLUSTER_K,
};
use crate::rl::obs::OBS_TOTAL;
use crate::rl::reward;
use crate::world::{config::WorldConfig, coords::GridPos, grid::Grid};
use self::clusters::{GoldCluster, find_clusters};
use self::nav::{navigate_action, NavCache};
use self::obs::build_obs_into;
use self::spawner::{SpawnBudget, DEFAULT_SPAWN_PROB, tick_spawns};

/// Maximum sim ticks a single navigation option runs before control returns to
/// the policy. Generous enough to cross the map with margin.
pub const MAX_OPTION_TICKS: u64 = 96;

pub struct SimCore {
    pub grid:    Grid,
    pub agents:  Vec<AgentState>,
    pub items:   Vec<ItemState>,
    pub tick:    u64,
    match_ticks: u64,
    world_cfg:   WorldConfig,
    prev_gold:   u8,
    prev_score:  u32,
    prev_pos:    GridPos,

    gold_positions: Vec<GridPos>,
    spawn_budgets:  Vec<SpawnBudget>,

    /// The RL agent's A* path cache.
    nav_cache: NavCache,

    rng: SmallRng,

    pub obs_buf: Vec<f32>,
}

impl SimCore {
    pub fn new(config_path: &str) -> Self {
        Self::new_with_seed(config_path, None)
    }

    pub fn new_with_seed(config_path: &str, seed: Option<u64>) -> Self {
        let world_cfg   = WorldConfig::load(config_path);
        let match_ticks = world_cfg.match_duration_ticks;

        let mut rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None    => SmallRng::try_from_rng(&mut SysRng).expect("SysRng failed"),
        };

        let snap = builder::build_episode(&world_cfg, &mut rng);

        let initial_pos    = snap.agents[0].pos;
        let gold_positions = gold_positions_of(&snap.items);
        let spawn_budgets  = build_spawn_budgets(&snap.items);

        let mut obs_buf = vec![0.0f32; OBS_TOTAL];
        let clusters    = find_clusters(
            &gold_positions, snap.grid.width as i32, snap.grid.height as i32,
        );
        build_obs_into(
            &mut obs_buf, &snap.agents[0],
            &gold_positions, &snap.items, &snap.grid, &clusters, 1.0,
        );

        Self {
            grid: snap.grid, agents: snap.agents, items: snap.items,
            tick: 0, match_ticks, world_cfg,
            prev_gold: 0, prev_score: 0, prev_pos: initial_pos,
            gold_positions, spawn_budgets,
            nav_cache: NavCache::new(),
            rng,
            obs_buf,
        }
    }

    pub fn reset(&mut self) {
        // Fresh map every episode: new randomised base, obstacles and items.
        let snap = builder::build_episode(&self.world_cfg, &mut self.rng);
        self.grid   = snap.grid;
        self.agents = snap.agents;
        self.items  = snap.items;

        self.gold_positions = gold_positions_of(&self.items);
        self.spawn_budgets  = build_spawn_budgets(&self.items);
        self.nav_cache = NavCache::new();

        self.tick       = 0;
        self.prev_gold  = 0;
        self.prev_score = 0;
        self.prev_pos   = self.agents[0].pos;

        self.build_observation();
    }

    fn tick_once(&mut self, rl_action: RlAction) -> (f32, bool) {
        self.prev_gold  = self.agents[0].gold_carried;
        self.prev_score = self.agents[0].score;
        self.prev_pos   = self.agents[0].pos;

        self.tick += 1;
        let done = self.tick >= self.match_ticks;

        physics::tick_buffs(&mut self.agents);

        let carry_speed = self.world_cfg.gold_carry_speed;

        // Gold regions for this tick — fixed 3×3 spatial grid (stable slots).
        let clusters = find_clusters(
            &self.gold_positions, self.grid.width as i32, self.grid.height as i32,
        );

        let wall_hit = self.apply_rl_action(rl_action, &clusters, carry_speed);

        pickup::pickup(&mut self.agents, &mut self.items);
        physics::auto_deposit(&mut self.agents, &self.grid);
        tick_spawns(&mut self.items, &self.agents, &self.grid, &self.spawn_budgets, &mut self.rng);

        self.gold_positions = gold_positions_of(&self.items);

        let rew = reward::compute(
            &self.world_cfg.reward,
            &self.agents[0],
            self.prev_gold,
            self.prev_score,
            wall_hit,
        );

        (rew, done)
    }

    /// Single-tick step (viewer path).
    pub fn step(&mut self, action: u32) -> (f32, bool) {
        let r = self.tick_once(int_to_rl_action(action));
        self.build_observation();
        r
    }

    /// Temporally-extended ("option") step used by the RL training/eval path.
    ///
    /// The option's return is the DISCOUNTED sum of its per-tick rewards,
    /// Σ γᵗ rₜ (semi-MDP option return, γ = `reward.option_gamma`). Discounting
    /// within the option is what makes a nearer reward worth more than a farther
    /// one: the same +pickup/+deposit earned after 50 ticks is scaled by ~γ⁵⁰,
    /// after 5 ticks by ~γ⁵. Without it every successful option looks equally good
    /// regardless of distance, so the policy has no incentive to prefer close gold.
    /// (Cross-option bootstrapping still uses the learner's per-step γ rather than a
    /// full γ^k jump — a further SMDP refinement — but the intra-option discount
    /// captures the dominant near-vs-far signal.)
    pub fn step_option(&mut self, action: u32) -> (f32, bool) {
        let rl_action = int_to_rl_action(action);
        let gamma = self.world_cfg.reward.option_gamma;
        let mut total_rew = 0.0f32;
        let mut discount = 1.0f32;
        let mut option_ticks = 0u64;
        let done = loop {
            let (r, d) = self.tick_once(rl_action);
            total_rew += discount * r;
            discount *= gamma;
            option_ticks += 1;

            if d { break true; }
            if !rl_action.is_navigation() { break false; }    // Wait: single tick
            if option_ticks >= MAX_OPTION_TICKS { break false; }
            // A reward-relevant event occurred this tick → return control to the policy.
            if self.agents[0].gold_carried != self.prev_gold
                || self.agents[0].score    != self.prev_score
            { break false; }
            // No movement this tick → arrived at the goal, blocked, or no valid target.
            if self.agents[0].pos == self.prev_pos { break false; }
        };
        self.build_observation();
        (total_rew, done)
    }

    /// Per-decision action mask: `valid[i] == true` iff action `i` is worth taking.
    /// `Wait` is never masked so there is always a legal action.
    ///
    ///   - cluster k : valid iff region k holds gold AND the agent can carry more
    ///   - base      : valid iff the agent is carrying gold
    ///   - speed     : valid iff a speed-boost item exists on the map
    ///   - multiplier: valid iff a multiplier item exists on the map
    ///   - wait      : always valid (fallback)
    pub fn action_mask(&self) -> [bool; ACTION_SIZE] {
        let mut mask = [false; ACTION_SIZE];
        mask[ACTION_WAIT as usize] = true;

        let agent = &self.agents[0];

        if agent.gold_carried < AGENT_MAX_GOLD {
            let clusters = find_clusters(
                &self.gold_positions, self.grid.width as i32, self.grid.height as i32,
            );
            for (k, slot) in mask.iter_mut().take(CLUSTER_K).enumerate() {
                if clusters[k].is_some() { *slot = true; }
            }
        }
        if agent.gold_carried > 0 {
            mask[ACTION_BASE as usize] = true;
        }
        if self.items.iter().any(|it| it.kind.is_speed()) {
            mask[ACTION_SPEED as usize] = true;
        }
        if self.items.iter().any(|it| it.kind == ItemKind::Multiplier) {
            mask[ACTION_MULT as usize] = true;
        }
        mask
    }

    /// Rebuild the observation buffer from the current world state.
    fn build_observation(&mut self) {
        let clusters = find_clusters(
            &self.gold_positions, self.grid.width as i32, self.grid.height as i32,
        );
        let time_remaining = 1.0 - (self.tick as f32 / self.match_ticks.max(1) as f32);
        build_obs_into(
            &mut self.obs_buf, &self.agents[0],
            &self.gold_positions, &self.items, &self.grid, &clusters, time_remaining,
        );
    }

    /// Resolve the RL action to a navigation goal and execute one A* step toward it.
    /// Returns `wall_hit` — true when a Move was attempted but the agent did not
    /// advance (blocked). `Wait` and goals with no valid target are no-ops.
    fn apply_rl_action(
        &mut self,
        rl_action:   RlAction,
        clusters:    &[Option<GoldCluster>; CLUSTER_K],
        carry_speed: f32,
    ) -> bool {
        match rl_action {
            RlAction::Wait => false,
            nav_action => {
                let goal = self.resolve_nav_goal(nav_action, clusters);
                if let Some(goal) = goal {
                    let prev_pos = self.agents[0].pos;
                    let act = navigate_action(
                        &self.agents[0], goal, &self.grid, &mut self.nav_cache,
                    );
                    physics::apply_action(
                        &mut self.agents, &self.grid, 0, act, carry_speed, self.tick, Some(goal),
                    );
                    matches!(act, Action::Move(_)) && self.agents[0].pos == prev_pos
                } else {
                    false
                }
            }
        }
    }

    /// Resolve a navigation RlAction to a concrete GridPos target.
    /// None if the target is unavailable (empty cluster, no such item on the map).
    fn resolve_nav_goal(
        &self,
        action:   RlAction,
        clusters: &[Option<GoldCluster>; CLUSTER_K],
    ) -> Option<GridPos> {
        let pos = self.agents[0].pos;
        match action {
            RlAction::NavigateToCluster(k) => clusters.get(k as usize)
                .and_then(|c| c.as_ref())
                .and_then(|c| c.nearest_gold(pos)),
            RlAction::NavigateToBase       => Some(self.agents[0].base_pos),
            RlAction::NavigateToSpeed      => self.nearest_item(pos, |k| k.is_speed()),
            RlAction::NavigateToMultiplier => self.nearest_item(pos, |k| k == ItemKind::Multiplier),
            RlAction::Wait                 => None,
        }
    }

    /// Nearest item (by Chebyshev distance) whose kind matches `pred`.
    fn nearest_item(&self, from: GridPos, pred: impl Fn(ItemKind) -> bool) -> Option<GridPos> {
        self.items.iter()
            .filter(|it| pred(it.kind))
            .min_by_key(|it| (it.pos.x - from.x).abs().max((it.pos.y - from.y).abs()))
            .map(|it| it.pos)
    }

    /// The RL agent's current A* path (remaining waypoints toward the last goal).
    pub fn agent_path(&self) -> &std::collections::VecDeque<GridPos> {
        self.nav_cache.path()
    }

    pub fn obs_as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.obs_buf.as_ptr() as *const u8,
                self.obs_buf.len() * std::mem::size_of::<f32>(),
            )
        }
    }

    pub fn match_ticks(&self) -> u64 { self.match_ticks }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn gold_positions_of(items: &[ItemState]) -> Vec<GridPos> {
    items.iter().filter(|it| it.kind == ItemKind::Gold).map(|it| it.pos).collect()
}

/// One spawn budget per item kind currently present, each refilled toward its
/// initial on-map count. Gold uses the default probability; flavour items refill
/// a little slower so the map doesn't saturate with buffs.
fn build_spawn_budgets(items: &[ItemState]) -> Vec<SpawnBudget> {
    use ItemKind::*;
    let mut budgets = Vec::new();
    for kind in [Gold, Speed1, Speed2, Speed3, Slow, Multiplier] {
        let target = items.iter().filter(|it| it.kind == kind).count();
        if target == 0 { continue; }
        let spawn_prob = if kind == Gold { DEFAULT_SPAWN_PROB } else { DEFAULT_SPAWN_PROB * 0.5 };
        budgets.push(SpawnBudget { kind, spawn_prob, target });
    }
    budgets
}
