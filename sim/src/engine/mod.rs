// src/engine/mod.rs
//
// Simulation core — one independent episode stream. agents[0] is always the RL
// agent (team 0).
//
// Per-tick step order (`tick_once`):
//   tick_speed_buffs
//   → find_clusters (gold clustering for this tick)
//   → resolve RL action → A* navigate OR direct combat/wait
//   → scripted enemy actions
//   → pickup → auto_deposit → spawner
//   → rebuild gold_positions
//   → compute reward
//
// Two public entry points wrap `tick_once`:
//   - `step`         : one tick + observation. Used by the viewer so rendering
//                      stays one-tile-per-frame and scripted baselines keep their
//                      per-tick decision cadence.
//   - `step_option`  : the RL training/eval path. Navigation actions are committed
//                      and executed via A* until a reward-relevant event (pickup /
//                      deposit / kill), arrival/blockage, death, episode end, or
//                      MAX_OPTION_TICKS — turning the per-tick MDP into a semi-MDP
//                      over high-level decisions (options framework). The agent only
//                      re-decides at option boundaries, which removes per-tick goal
//                      thrashing and shortens credit assignment to a few dozen
//                      decisions per episode. The observation is built once at the end.

pub mod builder;
pub mod clusters;
pub mod enemy;
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
use crate::rl::action::{RlAction, int_to_rl_action, ACTION_SIZE, ACTION_WAIT, CLUSTER_K};
use crate::rl::obs::OBS_TOTAL;
use crate::rl::reward;
use crate::world::{config::{EnemyKind, WorldConfig}, coords::GridPos, grid::Grid, tile::Tile};
use self::clusters::{GoldCluster, find_clusters};
use self::enemy::{compute_action, navigate_action, EnemyPathCache};
use self::obs::build_obs_into;
use self::spawner::{SpawnBudget, DEFAULT_SPAWN_PROB, tick_spawns};

/// Maximum sim ticks a single navigation option runs before control returns to
/// the policy (see `SimCore::step_option`). Generous enough to cross the map
/// (50×50, ~50-tile diagonal) with margin; caps wasted ticks on a stuck/empty goal.
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
    prev_kills:  u32,
    prev_pos:    GridPos,

    gold_positions: Vec<GridPos>,
    spawn_budgets:  Vec<SpawnBudget>,

    // Index 0: RL agent's A* path cache. Index 1+: scripted enemy caches.
    path_caches: Vec<EnemyPathCache>,
    enemy_kinds: Vec<EnemyKind>,

    rng: SmallRng,

    pub obs_buf: Vec<f32>,

    grid_snapshot:   Vec<Tile>,
    agents_snapshot: Vec<AgentState>,
    spawn_positions: Vec<GridPos>,
}

impl SimCore {
    pub fn new(config_path: &str) -> Self {
        Self::new_with_seed(config_path, None)
    }

    pub fn new_with_seed(config_path: &str, seed: Option<u64>) -> Self {
        let world_cfg   = WorldConfig::load(config_path);
        let match_ticks = world_cfg.match_duration_ticks;
        let snap        = builder::build(&world_cfg);

        let grid_snapshot   = snap.grid.clone_tiles();
        let agents_snapshot = snap.agents.clone();
        let spawn_positions = snap.agents.iter().map(|a| a.pos).collect();

        let initial_pos = snap.agents[0].pos;

        let gold_positions: Vec<GridPos> = snap.items.iter()
            .filter(|it| it.kind == ItemKind::Gold)
            .map(|it| it.pos)
            .collect();

        let spawn_budgets = vec![SpawnBudget {
            kind:       ItemKind::Gold,
            spawn_prob: DEFAULT_SPAWN_PROB,
            target:     gold_positions.len(),
        }];

        let n_agents = snap.agents.len();
        let enemy_kinds: Vec<EnemyKind> = snap.agents.iter()
            .map(|a| {
                world_cfg.agents.iter()
                    .find(|ac| ac.team == a.team)
                    .map(|ac| ac.enemy_kind)
                    .unwrap_or(EnemyKind::None)
            })
            .collect();
        let path_caches: Vec<EnemyPathCache> = (0..n_agents)
            .map(|_| EnemyPathCache::new())
            .collect();

        let mut obs_buf = vec![0.0f32; OBS_TOTAL];
        let clusters    = find_clusters(
            &gold_positions, snap.grid.width as i32, snap.grid.height as i32,
        );
        build_obs_into(
            &mut obs_buf, &snap.agents[0],
            &gold_positions, &snap.grid, &clusters,
            1.0, // full match remaining at construction
        );

        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None    => SmallRng::try_from_rng(&mut SysRng).expect("SysRng failed"),
        };

        Self {
            grid: snap.grid, agents: snap.agents, items: snap.items,
            tick: 0, match_ticks, world_cfg,
            prev_gold: 0, prev_score: 0, prev_kills: 0, prev_pos: initial_pos,
            gold_positions, spawn_budgets,
            path_caches, enemy_kinds,
            rng,
            obs_buf,
            grid_snapshot,
            agents_snapshot,
            spawn_positions,
        }
    }

    pub fn reset(&mut self) {
        self.grid.restore_tiles(&self.grid_snapshot);
        self.agents.clone_from(&self.agents_snapshot);
        self.items = builder::spawn_items(
            &self.world_cfg,
            &self.grid,
            &self.spawn_positions,
            &mut self.rng,
        );
        self.gold_positions.clear();
        self.gold_positions.extend(
            self.items.iter()
                .filter(|it| it.kind == ItemKind::Gold)
                .map(|it| it.pos),
        );
        if let Some(b) = self.spawn_budgets.iter_mut().find(|b| b.kind == ItemKind::Gold) {
            b.target = self.gold_positions.len();
        }
        for cache in &mut self.path_caches { *cache = EnemyPathCache::default(); }
        self.tick       = 0;
        self.prev_gold   = 0;
        self.prev_score  = 0;
        self.prev_kills  = 0;
        self.prev_pos    = self.agents[0].pos;

        self.build_observation();
    }

    /// Advance the simulation by exactly one tick for `rl_action` and return
    /// `(reward, done)`. Does NOT rebuild the observation — callers invoke
    /// `build_observation` once they finish advancing (after one tick for the
    /// viewer, or at the end of an option for the RL path).
    fn tick_once(&mut self, rl_action: RlAction) -> (f32, bool) {
        self.prev_gold   = self.agents[0].gold_carried;
        self.prev_score  = self.agents[0].score;
        self.prev_kills  = self.agents[0].kills;
        self.prev_pos    = self.agents[0].pos;

        self.tick += 1;
        let done = self.tick >= self.match_ticks;

        physics::tick_speed_buffs(&mut self.agents);
        physics::tick_cooldowns(&mut self.agents);
        let respawned = physics::tick_respawns(&mut self.agents);
        // Clear stale path caches for any agent that just respawned.
        for i in 0..self.path_caches.len() {
            if respawned & (1 << i) != 0 {
                self.path_caches[i] = EnemyPathCache::default();
            }
        }

        let carry_speed = self.world_cfg.gold_carry_speed;

        // Gold regions for this tick — fixed 3×3 spatial grid, so region k is a
        // stable slot. Used for action resolution and the obs cluster features.
        let clusters = find_clusters(
            &self.gold_positions, self.grid.width as i32, self.grid.height as i32,
        );

        let wall_hit = self.apply_rl_action(rl_action, &clusters, carry_speed);

        // Scripted enemies (agents 1+). Dormant in the gold game (no enemy agents);
        // kept intact for the future combat game mode.
        for idx in 1..self.agents.len() {
            let kind = self.enemy_kinds[idx];
            if kind == EnemyKind::None { continue; }
            let act = compute_action(
                kind, &self.agents[idx], &self.items, &self.grid,
                &mut self.path_caches[idx],
            );
            physics::apply_action(&mut self.agents, &self.grid, idx, act, carry_speed);
        }

        pickup::pickup(&mut self.agents, &mut self.items);
        physics::auto_deposit(&mut self.agents, &self.grid);
        tick_spawns(&mut self.items, &self.agents, &self.grid, &self.spawn_budgets, &mut self.rng);

        self.gold_positions.clear();
        self.gold_positions.extend(
            self.items.iter()
                .filter(|it| it.kind == ItemKind::Gold)
                .map(|it| it.pos),
        );

        // Event-based reward only — gold picked up + banked, time cost, wall_hit.
        let rew = reward::compute(
            &self.world_cfg.reward,
            &self.agents[0],
            self.prev_gold,
            self.prev_score,
            wall_hit,
        );

        (rew, done)
    }

    /// Single-tick step (decode + one tick + observation). Used by the viewer so
    /// rendering stays one-tile-per-frame and scripted baselines keep their
    /// per-tick decision cadence. The RL training/eval path uses `step_option`.
    pub fn step(&mut self, action: u32) -> (f32, bool) {
        let r = self.tick_once(int_to_rl_action(action));
        self.build_observation();
        r
    }

    /// Temporally-extended ("option") step used by the RL training/eval path.
    ///
    /// A navigation action is committed and executed via A* until a reward-relevant
    /// event (pickup / deposit), arrival or blockage (no movement this tick), the
    /// episode ending, or MAX_OPTION_TICKS — whichever comes first. `Wait` runs for
    /// a single tick. The accumulated reward over all inner ticks is returned, and
    /// the observation is built once at the end.
    ///
    /// This turns the per-tick MDP into a semi-MDP over high-level decisions: the
    /// policy only re-decides at option boundaries, which removes per-tick goal
    /// thrashing and shortens credit assignment from ~match_ticks steps to a few
    /// dozen decisions per episode (the options-framework / hierarchical-RL split,
    /// with A* as the low-level controller).
    pub fn step_option(&mut self, action: u32) -> (f32, bool) {
        let rl_action = int_to_rl_action(action);
        let mut total_rew = 0.0f32;
        let mut option_ticks = 0u64;
        let done = loop {
            let (r, d) = self.tick_once(rl_action);
            total_rew += r;
            option_ticks += 1;

            if d { break true; }
            if !rl_action.is_navigation() { break false; }    // Wait: single tick
            if option_ticks >= MAX_OPTION_TICKS { break false; }
            if self.agents[0].respawn_timer > 0 { break false; }  // died mid-option (combat only)
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

    /// Per-decision action mask: `valid[i] == true` iff action `i` is worth taking
    /// in the current state. Consumed by MaskablePPO. `Wait` is never masked so
    /// there is always a legal action.
    ///
    ///   - cluster k : valid iff region k holds gold AND the agent can carry more
    ///   - base      : valid iff the agent is carrying gold (something to deposit)
    ///   - wait      : always valid (fallback)
    pub fn action_mask(&self) -> [bool; ACTION_SIZE] {
        let mut mask = [false; ACTION_SIZE];
        mask[ACTION_WAIT as usize] = true;

        let agent = &self.agents[0];
        if agent.respawn_timer > 0 { return mask; }  // dormant: combat only

        if agent.gold_carried < AGENT_MAX_GOLD {
            let clusters = find_clusters(
                &self.gold_positions, self.grid.width as i32, self.grid.height as i32,
            );
            for (k, slot) in mask.iter_mut().take(CLUSTER_K).enumerate() {
                if clusters[k].is_some() { *slot = true; }
            }
        }
        if agent.gold_carried > 0 {
            mask[CLUSTER_K] = true;  // NavigateToBase
        }
        mask
    }

    /// Rebuild the observation buffer from the current world state. Cluster features
    /// are recomputed from the post-step gold positions so they stay consistent with
    /// the GOLD channel.
    fn build_observation(&mut self) {
        let clusters = find_clusters(
            &self.gold_positions, self.grid.width as i32, self.grid.height as i32,
        );
        let time_remaining = 1.0 - (self.tick as f32 / self.match_ticks.max(1) as f32);
        build_obs_into(
            &mut self.obs_buf, &self.agents[0],
            &self.gold_positions, &self.grid, &clusters,
            time_remaining,
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
        if self.agents[0].respawn_timer > 0 { return false; }
        match rl_action {
            RlAction::Wait => false,
            nav_action => {
                let goal = resolve_nav_goal(&self.agents, nav_action, clusters);
                if let Some(goal) = goal {
                    let prev_pos = self.agents[0].pos;
                    // navigate_action uses the cached path; recomputes only when goal changes.
                    let act = navigate_action(
                        &self.agents[0], goal, &self.grid, &mut self.path_caches[0],
                    );
                    physics::apply_action(&mut self.agents, &self.grid, 0, act, carry_speed);
                    matches!(act, Action::Move(_)) && self.agents[0].pos == prev_pos
                } else {
                    false
                }
            }
        }
    }

    /// The RL agent's current A* path (remaining waypoints toward the last navigation goal).
    /// Populated by apply_rl_action; empty for Wait.
    pub fn agent_path(&self) -> &std::collections::VecDeque<GridPos> {
        self.path_caches[0].path()
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

// ── Navigation goal resolution ────────────────────────────────────────────────

/// Resolve a navigation RlAction to a concrete GridPos target.
/// Returns None if the target is unavailable (e.g. an empty cluster region).
fn resolve_nav_goal(
    agents:   &[AgentState],
    action:   RlAction,
    clusters: &[Option<GoldCluster>; CLUSTER_K],
) -> Option<GridPos> {
    let agent = &agents[0];
    match action {
        RlAction::NavigateToCluster(k) => {
            clusters.get(k as usize)
                .and_then(|c| c.as_ref())
                .and_then(|c| c.nearest_gold(agent.pos))
        }
        RlAction::NavigateToBase => Some(agent.base_pos),
        // Wait is handled in apply_rl_action — never reaches here.
        RlAction::Wait => None,
    }
}
