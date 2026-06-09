// src/engine/mod.rs
//
// Simulation core — one independent episode stream for the single RL agent
// (single-agent gold rush: no enemies, no combat).
//
// Per-tick step order (`tick_once`):
//   tick_buffs (speed-buff decay)
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
use rustc_hash::FxHashSet;

use crate::config::{AGENT_MAX_GOLD, MOVE_ENERGY_BASE, MOVE_ENERGY_SPEED, MOVE_ENERGY_STEP};
use crate::entity::item::ItemKind;
use crate::rl::action::{
    int_to_rl_action, RlAction, ACTION_BASE, ACTION_MULT, ACTION_SIZE,
    ACTION_SPEED, ACTION_WAIT, CLUSTER_K,
};
use crate::rl::obs::OBS_TOTAL;
use crate::rl::reward;
use crate::world::{config::WorldConfig, coords::GridPos, grid::Grid};
use self::clusters::{GoldCluster, find_clusters, region_of};
use self::nav::{navigate_action, NavCache};
use self::obs::build_obs_into;
use self::spawner::{SpawnBudget, DEFAULT_SPAWN_PROB, tick_spawns};

/// Maximum sim ticks a single navigation option runs before control returns to
/// the policy. Generous enough to cross the map with margin — at the base movement
/// cadence (one tile every other tick) ~192 ticks covers ~96 tiles of travel, so a
/// far-corner goal on a 50×50 map is still reachable inside one option.
pub const MAX_OPTION_TICKS: u64 = 192;

/// Per-decision telemetry, captured at each option boundary (the moment the policy
/// commits to a high-level action). Used to answer "does the agent skip nearby gold
/// and range far?" — surfaced to Python and logged to TensorBoard during training.
#[derive(Clone, Copy)]
pub struct DecisionTelem {
    /// The chosen action was a `NavigateToCluster` (going to collect gold).
    pub is_cluster: bool,
    /// The agent's own region held collectible gold at the decision (and it could
    /// carry more) — i.e. there was local gold to grab.
    pub own_has_gold: bool,
    /// `own_has_gold` AND the agent chose a *different* gold region instead.
    pub skipped_own: bool,
    /// Chebyshev distance to the gold the agent committed to collect, or -1 when the
    /// decision was not a gold-collection action (base / buff / wait / empty region).
    pub chosen_dist: i32,
}

impl Default for DecisionTelem {
    fn default() -> Self {
        Self { is_cluster: false, own_has_gold: false, skipped_own: false, chosen_dist: -1 }
    }
}

/// One per-tick record captured inside `step_option` when tracing is enabled
/// (`set_trace(true)`). Lets the offline trace recorder reconstruct the agent's
/// path and the exact reward signal that fired each tick of an option, without
/// touching the zero-overhead batch-training path (tracing defaults off).
///
/// The map (gold/items/obstacles) is essentially static within an option — a
/// pickup/deposit ends the option — so full positions are snapshotted per
/// DECISION on the Python side; this record carries only what changes per tick.
#[derive(Clone, Copy, Debug)]
pub struct TickRecord {
    pub tick:         u64,
    pub ax:           i32,
    pub ay:           i32,
    pub gold_carried: u8,
    pub score:        u32,
    pub r_tick:       f32,
    pub r_pickup:     f32,
    pub r_deposit:    f32,
    pub r_wall:       f32,
    pub r_total:      f32,
    /// γᵗ applied to this tick within the option (running intra-option discount).
    pub discount:     f32,
    /// Gold tiles remaining on the map after this tick.
    pub gold_count:   u32,
}

/// Number of f32 columns when a `TickRecord` is flattened for FFI — keep in sync
/// with `TickRecord` field count (see env.rs `trace_flat` and pyo3.rs `get_trace`).
pub const TRACE_FIELDS: usize = 12;

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

    /// True while the agent is actively pursuing its nav goal — it either stepped or
    /// is resting between movement-cadence steps. The option loop keeps running while
    /// this holds and ends the option once it goes false (arrival, a wall block, or no
    /// valid target). Replaces the old "position unchanged ⇒ arrived" inference, which
    /// the half-cadence base movement would otherwise trip on every rest tick.
    en_route:    bool,

    /// Telemetry for the most recent high-level decision (option boundary).
    last_decision: DecisionTelem,
    /// Sim ticks the most recent option ran (for SMDP γ^k cross-option discounting).
    last_option_ticks: u64,

    gold_positions: Vec<GridPos>,
    spawn_budgets:  Vec<SpawnBudget>,
    /// Spawn pocket around the base — items never spawn here (kept in sync on reset).
    spawn_block:    FxHashSet<GridPos>,

    /// The RL agent's A* path cache.
    nav_cache: NavCache,

    rng: SmallRng,

    pub obs_buf: Vec<f32>,

    /// Offline trace capture (default off → zero overhead for training). When on,
    /// `step_option` records a `TickRecord` per tick into `trace` (cleared at each
    /// option boundary). Used only by the single-env trace recorder.
    trace_enabled: bool,
    trace:         Vec<TickRecord>,
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
        let spawn_block    = builder::spawn_pocket(
            initial_pos, snap.grid.width as i32, snap.grid.height as i32,
        ).into_iter().collect();

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
            en_route: false,
            last_decision: DecisionTelem::default(),
            last_option_ticks: 1,
            gold_positions, spawn_budgets, spawn_block,
            nav_cache: NavCache::new(),
            rng,
            obs_buf,
            trace_enabled: false,
            trace:         Vec::new(),
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
        self.spawn_block    = builder::spawn_pocket(
            self.agents[0].pos, self.grid.width as i32, self.grid.height as i32,
        ).into_iter().collect();
        self.nav_cache = NavCache::new();

        self.tick       = 0;
        self.prev_gold  = 0;
        self.prev_score = 0;
        self.prev_pos   = self.agents[0].pos;
        self.en_route   = false;
        self.last_decision = DecisionTelem::default();

        self.build_observation();
    }

    fn tick_once(&mut self, rl_action: RlAction) -> (reward::RewardBreakdown, bool) {
        self.prev_gold  = self.agents[0].gold_carried;
        self.prev_score = self.agents[0].score;
        self.prev_pos   = self.agents[0].pos;

        self.tick += 1;
        let done = self.tick >= self.match_ticks;

        physics::tick_buffs(&mut self.agents);

        // Gold regions for this tick — fixed 3×3 spatial grid (stable slots).
        let clusters = find_clusters(
            &self.gold_positions, self.grid.width as i32, self.grid.height as i32,
        );

        let wall_hit = self.apply_rl_action(rl_action, &clusters);

        pickup::pickup(&mut self.agents, &mut self.items);
        physics::auto_deposit(&mut self.agents, &self.grid);
        tick_spawns(&mut self.items, &self.agents, &self.grid, &self.spawn_budgets, &self.spawn_block, &mut self.rng);

        self.gold_positions = gold_positions_of(&self.items);

        let rew = reward::compute_components(
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
        let (rb, done) = self.tick_once(int_to_rl_action(action));
        self.build_observation();
        (rb.total(), done)
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
        // Capture decision telemetry at the option boundary (before stepping).
        self.last_decision = self.decision_telem(rl_action);
        if self.trace_enabled { self.trace.clear(); }
        let gamma = self.world_cfg.reward.option_gamma;
        let mut total_rew = 0.0f32;
        let mut discount = 1.0f32;
        let mut option_ticks = 0u64;
        let done = loop {
            let (rb, d) = self.tick_once(rl_action);
            let r = rb.total();
            total_rew += discount * r;
            if self.trace_enabled {
                let agent = &self.agents[0];
                self.trace.push(TickRecord {
                    tick:         self.tick,
                    ax:           agent.pos.x,
                    ay:           agent.pos.y,
                    gold_carried: agent.gold_carried,
                    score:        agent.score,
                    r_tick:       rb.tick,
                    r_pickup:     rb.pickup,
                    r_deposit:    rb.deposit,
                    r_wall:       rb.wall,
                    r_total:      r,
                    discount,
                    gold_count:   self.gold_positions.len() as u32,
                });
            }
            discount *= gamma;
            option_ticks += 1;

            if d { break true; }
            if !rl_action.is_navigation() { break false; }    // Wait: single tick
            if option_ticks >= MAX_OPTION_TICKS { break false; }
            // A reward-relevant event occurred this tick → return control to the policy.
            if self.agents[0].gold_carried != self.prev_gold
                || self.agents[0].score    != self.prev_score
            { break false; }
            // Option ends once the agent stops pursuing its goal — arrived, wall-blocked
            // or no valid target. A cadence "rest" tick (no step, but still en route)
            // is NOT an ending, so the option survives the base half-speed movement.
            if !self.en_route { break false; }
        };
        self.last_option_ticks = option_ticks;
        self.build_observation();
        (total_rew, done)
    }

    /// Sim ticks the most recent option ran (≥1). The SMDP cross-option discount is
    /// γ^k for an option of length k — see the Python SmdpRolloutBuffer.
    pub fn last_option_ticks(&self) -> u64 { self.last_option_ticks }

    /// Enable/disable per-tick trace capture (off by default — see `TickRecord`).
    pub fn set_trace(&mut self, on: bool) { self.trace_enabled = on; }

    /// Per-tick trace for the most recent option (empty unless tracing is enabled).
    pub fn trace(&self) -> &[TickRecord] { &self.trace }

    /// Reward weights in play: (tick, pickup, deposit, wall_hit, option_gamma).
    /// Surfaced so the trace recorder can record the exact shaping used.
    pub fn reward_weights(&self) -> (f32, f32, f32, f32, f32) {
        let r = &self.world_cfg.reward;
        (r.tick, r.pickup, r.deposit, r.wall_hit, r.option_gamma)
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

    /// Resolve the RL action to a navigation goal and, subject to the movement
    /// cadence, execute at most one A* step toward it. Returns `wall_hit` — true when
    /// a step was attempted but the agent walked straight into a wall. As a side effect
    /// sets `self.en_route` (see the field doc): true while the agent is still pursuing
    /// the goal (it stepped, or it is resting between cadence steps), false once it has
    /// arrived, is wall-blocked, or has no valid target.
    ///
    /// Movement cadence: each tick the agent gains move energy (base vs. speed-buffed)
    /// and only steps a tile once it reaches `MOVE_ENERGY_STEP`. Base steps every other
    /// tick; a speed buff steps every tick — capped at one tile/tick so a speed move
    /// never teleports two tiles.
    fn apply_rl_action(
        &mut self,
        rl_action: RlAction,
        clusters:  &[Option<GoldCluster>; CLUSTER_K],
    ) -> bool {
        let nav_action = match rl_action {
            RlAction::Wait => { self.en_route = false; return false; }
            nav => nav,
        };

        // Advance the movement-cadence accumulator; rest this tick if it hasn't yet
        // reached the step threshold (still en route — the option keeps running).
        let rate = if self.agents[0].speed_buff > 0 { MOVE_ENERGY_SPEED } else { MOVE_ENERGY_BASE };
        self.agents[0].move_energy = self.agents[0].move_energy.saturating_add(rate);
        if self.agents[0].move_energy < MOVE_ENERGY_STEP {
            self.en_route = true;
            return false;
        }
        self.agents[0].move_energy -= MOVE_ENERGY_STEP;

        let Some(goal) = self.resolve_nav_goal(nav_action, clusters) else {
            self.en_route = false; // no valid target
            return false;
        };

        let act = navigate_action(
            &self.agents[0], goal, &self.grid, &mut self.nav_cache,
        );
        let prev = self.agents[0].pos;
        // Returns true iff the agent stepped straight into a wall.
        let wall_hit = physics::apply_action(
            &mut self.agents, &self.grid, 0, act, self.tick, Some(goal),
        );
        // Still en route only if the step made progress; arrival/blockage ends the option.
        self.en_route = self.agents[0].pos != prev;
        wall_hit
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
            RlAction::NavigateToBase        => Some(self.agents[0].base_pos),
            RlAction::NavigateToSpeed       => self.nearest_item(pos, |k| k.is_speed()),
            RlAction::NavigateToMultiplier  => self.nearest_item(pos, |k| k == ItemKind::Multiplier),
            RlAction::Wait                  => None,
        }
    }

    /// Compute decision telemetry for a high-level action at the current state.
    /// "Own region" is the fixed 3×3 region the agent currently stands in; the agent
    /// "skips" local gold when that region holds collectible gold but it commits to a
    /// different region instead. `chosen_dist` is the Chebyshev distance to the gold
    /// it actually targets (−1 for non-gold actions).
    fn decision_telem(&self, action: RlAction) -> DecisionTelem {
        let pos = self.agents[0].pos;
        let (gw, gh) = (self.grid.width as i32, self.grid.height as i32);
        let clusters = find_clusters(&self.gold_positions, gw, gh);
        let own = region_of(pos, gw, gh);
        let own_has_gold =
            self.agents[0].gold_carried < AGENT_MAX_GOLD && clusters[own].is_some();

        // Which gold this action commits to (and the region it targets, so we can tell
        // whether it "skips" gold in the agent's own region). Only the region-nav
        // actions commit to gold; everything else is a non-gold goal.
        let target: Option<(GridPos, Option<usize>)> = match action {
            RlAction::NavigateToCluster(k) => {
                let kk = k as usize;
                clusters.get(kk).and_then(|c| c.as_ref())
                    .and_then(|c| c.nearest_gold(pos))
                    .map(|g| (g, Some(kk)))
            }
            _ => None,
        };

        match target {
            Some((g, region)) => {
                // Path-aware distance (around walls), not Chebyshev.
                let field = nav::dist_field(&self.grid, pos);
                let chosen_dist = nav::dist_at(&field, &self.grid, g).unwrap_or(-1);
                let skipped_own = own_has_gold && matches!(region, Some(k) if k != own);
                DecisionTelem { is_cluster: true, own_has_gold, skipped_own, chosen_dist }
            }
            None => DecisionTelem { is_cluster: false, own_has_gold, skipped_own: false, chosen_dist: -1 },
        }
    }

    /// Telemetry for the most recent high-level decision (set by `step_option`).
    pub fn last_decision(&self) -> DecisionTelem { self.last_decision }

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
    for kind in [Gold, Speed, Multiplier] {
        let target = items.iter().filter(|it| it.kind == kind).count();
        if target == 0 { continue; }
        let spawn_prob = if kind == Gold { DEFAULT_SPAWN_PROB } else { DEFAULT_SPAWN_PROB * 0.5 };
        budgets.push(SpawnBudget { kind, spawn_prob, target });
    }
    budgets
}
