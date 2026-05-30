// src/engine/mod.rs
//
// Simulation core — one independent episode stream. agents[0] is always the RL
// agent (team 0).
//
// Step order:
//   tick_speed_buffs
//   → find_clusters (gold clustering for this tick)
//   → resolve RL action → A* navigate OR direct combat/wait
//   → scripted enemy actions
//   → pickup → auto_deposit → spawner
//   → rebuild gold_positions
//   → compute reward
//   → build observation (includes enemy paths + cluster features)

pub mod builder;
pub mod clusters;
pub mod enemy;
pub mod obs;
pub mod physics;
pub mod pickup;
pub mod spawner;

pub use crate::entity::{AgentState, ItemState};

use std::collections::VecDeque;

use rand::SeedableRng;
use rand::rngs::{SmallRng, SysRng};

use crate::entity::agent::Action;
use crate::entity::item::ItemKind;
use crate::rl::action::{RlAction, int_to_rl_action};
use crate::rl::obs::OBS_TOTAL;
use crate::rl::reward;
use crate::world::{config::{EnemyKind, WorldConfig}, coords::GridPos, grid::Grid, tile::Tile};
use crate::rl::action::CLUSTER_K;
use self::clusters::{GoldCluster, find_clusters, chebyshev};
use self::enemy::{compute_action, navigate_action, EnemyPathCache};
use self::obs::build_obs_into;
use self::spawner::{SpawnBudget, DEFAULT_SPAWN_PROB, tick_spawns};

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
        let no_paths: Vec<&VecDeque<GridPos>> = vec![];
        build_obs_into(
            &mut obs_buf, &snap.agents[0], &snap.items, &snap.agents,
            &gold_positions, &snap.grid, &no_paths, &clusters,
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

        let clusters = find_clusters(
            &self.gold_positions, self.grid.width as i32, self.grid.height as i32,
        );
        let no_paths: Vec<&VecDeque<GridPos>> = vec![];
        build_obs_into(
            &mut self.obs_buf, &self.agents[0], &self.items, &self.agents,
            &self.gold_positions, &self.grid, &no_paths, &clusters,
            1.0, // full match remaining after reset
        );
    }

    pub fn step(&mut self, action: u32) -> (f32, bool) {
        self.prev_gold   = self.agents[0].gold_carried;
        self.prev_score  = self.agents[0].score;
        self.prev_kills  = self.agents[0].kills;
        self.prev_pos    = self.agents[0].pos;
        let prev_alive   = self.agents[0].respawn_timer == 0;

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
        // stable slot. Used for both obs features and action resolution.
        let clusters = find_clusters(
            &self.gold_positions, self.grid.width as i32, self.grid.height as i32,
        );

        // nav_goal (the action's chosen target) is no longer used for shaping —
        // shaping uses a state-defined objective instead (see below).
        let (wall_hit, _nav_goal) = self.apply_rl_action(int_to_rl_action(action), &clusters, carry_speed);

        // Scripted enemies (agents 1+).
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

        let just_died = prev_alive && self.agents[0].respawn_timer > 0;

        // Event-based reward only — no sim-side navigation objective. The agent
        // must learn where to go from the observation, not from a baked-in hint.
        let rew = reward::compute(
            &self.world_cfg.reward,
            &self.agents[0],
            self.prev_pos,
            self.prev_gold,
            self.prev_score,
            self.prev_kills,
            wall_hit,
            just_died,
        );

        // Collect path references for alive enemies only (path_caches[0] is the RL agent).
        let enemy_paths: Vec<&VecDeque<GridPos>> = (1..self.agents.len())
            .filter(|&i| self.agents[i].respawn_timer == 0)
            .map(|i| self.path_caches[i].path())
            .collect();

        // Fraction of the match still to play — fed to CH_TIME_REMAINING so the
        // value function is time-aware (keeps the truncation bootstrap unbiased).
        let time_remaining = 1.0 - (self.tick as f32 / self.match_ticks.max(1) as f32);
        build_obs_into(
            &mut self.obs_buf, &self.agents[0], &self.items, &self.agents,
            &self.gold_positions, &self.grid, &enemy_paths, &clusters,
            time_remaining,
        );

        (rew, done)
    }

    /// Resolve the RL action to a navigation goal or direct combat/wait.
    /// Returns `(wall_hit, nav_goal)` — nav_goal is None for Wait/Attack actions
    /// or when no valid target exists (e.g. no health pickups on the map).
    /// nav_goal is no longer used for reward shaping (shaping uses a state-defined
    /// objective resolved in `step`); it is retained for the agent's A* target.
    fn apply_rl_action(
        &mut self,
        rl_action:   RlAction,
        clusters:    &[Option<GoldCluster>; CLUSTER_K],
        carry_speed: f32,
    ) -> (bool, Option<GridPos>) {
        if self.agents[0].respawn_timer > 0 { return (false, None); }
        match rl_action {
            RlAction::MeleeAttack => {
                physics::try_melee_attack(
                    &mut self.agents, 0,
                    self.world_cfg.melee_range  as i32,
                    self.world_cfg.melee_damage,
                    self.world_cfg.melee_cooldown_ticks,
                    self.world_cfg.respawn_ticks,
                );
                (false, None)
            }
            RlAction::RangedAttack => {
                physics::try_ranged_attack(
                    &mut self.agents, 0,
                    self.world_cfg.ranged_range  as i32,
                    self.world_cfg.ranged_damage,
                    self.world_cfg.ranged_cooldown_ticks,
                    self.world_cfg.respawn_ticks,
                );
                (false, None)
            }
            RlAction::Wait => (false, None),

            nav_action => {
                // Resolve navigation goal from the action type.
                let goal = resolve_nav_goal(
                    &self.agents, &self.items, nav_action, clusters,
                );
                if let Some(goal) = goal {
                    let prev_pos = self.agents[0].pos;
                    // navigate_action uses the cached path; recomputes only when goal changes.
                    let act = navigate_action(
                        &self.agents[0], goal, &self.grid, &mut self.path_caches[0],
                    );
                    physics::apply_action(&mut self.agents, &self.grid, 0, act, carry_speed);
                    let wall_hit = matches!(act, Action::Move(_)) && self.agents[0].pos == prev_pos;
                    (wall_hit, Some(goal))
                } else {
                    (false, None)
                }
            }
        }
    }

    /// The RL agent's current A* path (remaining waypoints toward the last navigation goal).
    /// Populated by apply_rl_action every step; empty for Wait/Attack actions.
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
/// Returns None if the target type is unavailable (e.g., no health pickups on map).
fn resolve_nav_goal(
    agents:   &[AgentState],
    items:    &[ItemState],
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
        RlAction::NavigateToHealth => items.iter()
            .filter(|i| i.kind == ItemKind::Health)
            .min_by_key(|i| chebyshev(agent.pos, i.pos))
            .map(|i| i.pos),
        RlAction::NavigateToAmmo => items.iter()
            .filter(|i| i.kind == ItemKind::Ammo)
            .min_by_key(|i| chebyshev(agent.pos, i.pos))
            .map(|i| i.pos),
        RlAction::NavigateToEnemy => agents.iter()
            .skip(1)
            .filter(|a| a.team != agent.team)
            .min_by_key(|a| chebyshev(agent.pos, a.pos))
            .map(|a| a.pos),
        // Direct actions handled in apply_rl_action — never reach here.
        RlAction::MeleeAttack | RlAction::RangedAttack | RlAction::Wait => None,
    }
}
