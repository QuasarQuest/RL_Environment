// src/engine/mod.rs
//
// Simulation core — one independent episode stream. agents[0] is always the RL
// agent (team 0).
//
// Step order: tick_speed_buffs → apply_action → pickup → auto_deposit → spawner → reward → obs
//
// Performance
// -----------
// obs_buf        pre-allocated f32 slice, reused every step (no heap alloc)
// grid_snapshot  memcpy-fast tile reset instead of full Grid rebuild
// gold_positions rebuilt once per step after all item mutations (pickup + spawner)
// rng            SmallRng per SimCore for reproducible spawns when seeded

pub mod builder;
pub mod enemy;
pub mod obs;
pub mod physics;
pub mod pickup;
pub mod spawner;

// Re-export data types so callers can use atb::engine::{AgentState, ItemState}.
pub use crate::entity::{AgentState, ItemState};

use rand::SeedableRng;
use rand::rngs::{SmallRng, SysRng};

use crate::entity::agent::Action;
use crate::entity::item::ItemKind;
use crate::rl::action::int_to_action;
use crate::rl::obs::OBS_TOTAL;
use crate::rl::reward;
use crate::world::{config::{EnemyKind, WorldConfig}, coords::GridPos, grid::Grid, tile::Tile};
use self::enemy::{compute_action, EnemyPathCache};
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
    prev_pos:    GridPos,

    // Pre-filtered gold positions for reward shaping.
    // Rebuilt after every tick's item mutations (pickup + spawner).
    gold_positions: Vec<GridPos>,

    // Probabilistic spawner budgets — gold refills to initial map density.
    spawn_budgets: Vec<SpawnBudget>,

    // Per-enemy path caches indexed by agent slot (slot 0 is the RL agent).
    enemy_caches: Vec<EnemyPathCache>,
    // EnemyKind per agent slot; EnemyKind::None for the RL agent.
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
        // snapshot is sorted by team; look up enemy_kind by team to keep alignment.
        let enemy_kinds: Vec<EnemyKind> = snap.agents.iter()
            .map(|a| {
                world_cfg.agents.iter()
                    .find(|ac| ac.team == a.team)
                    .map(|ac| ac.enemy_kind)
                    .unwrap_or(EnemyKind::None)
            })
            .collect();
        let enemy_caches: Vec<EnemyPathCache> = (0..n_agents)
            .map(|_| EnemyPathCache::new())
            .collect();

        let mut obs_buf = vec![0.0f32; OBS_TOTAL];
        build_obs_into(&mut obs_buf, &snap.agents[0], &snap.items, &snap.agents, &gold_positions, &snap.grid);

        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None    => SmallRng::try_from_rng(&mut SysRng).expect("SysRng failed"),
        };

        Self {
            grid: snap.grid, agents: snap.agents, items: snap.items,
            tick: 0, match_ticks, world_cfg,
            prev_gold: 0, prev_score: 0, prev_pos: initial_pos,
            gold_positions, spawn_budgets,
            enemy_caches, enemy_kinds,
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
        // Sync gold budget target to fresh placement count.
        if let Some(b) = self.spawn_budgets.iter_mut().find(|b| b.kind == ItemKind::Gold) {
            b.target = self.gold_positions.len();
        }
        for cache in &mut self.enemy_caches { *cache = EnemyPathCache::default(); }
        self.tick       = 0;
        self.prev_gold  = 0;
        self.prev_score = 0;
        self.prev_pos   = self.agents[0].pos;
        build_obs_into(&mut self.obs_buf, &self.agents[0], &self.items, &self.agents, &self.gold_positions, &self.grid);
    }

    pub fn step(&mut self, action: u32) -> (f32, bool) {
        self.prev_gold  = self.agents[0].gold_carried;
        self.prev_score = self.agents[0].score;
        self.prev_pos   = self.agents[0].pos;

        self.tick += 1;
        let done = self.tick >= self.match_ticks;

        physics::tick_speed_buffs(&mut self.agents);
        let agent_action = int_to_action(action);
        physics::apply_action(&mut self.agents, &self.grid, 0, agent_action);
        let wall_hit = matches!(agent_action, Action::Move(_)) && self.agents[0].pos == self.prev_pos;

        // Run scripted enemies (agents 1+).
        for idx in 1..self.agents.len() {
            let kind = self.enemy_kinds[idx];
            if kind == EnemyKind::None { continue; }
            let act = compute_action(kind, &self.agents[idx], &self.items, &self.grid, &mut self.enemy_caches[idx]);
            physics::apply_action(&mut self.agents, &self.grid, idx, act);
        }

        pickup::pickup(&mut self.agents, &mut self.items);
        physics::auto_deposit(&mut self.agents, &self.grid);
        tick_spawns(&mut self.items, &self.agents, &self.grid, &self.spawn_budgets, &mut self.rng);

        // Rebuild gold_positions once after all item mutations.
        self.gold_positions.clear();
        self.gold_positions.extend(
            self.items.iter()
                .filter(|it| it.kind == ItemKind::Gold)
                .map(|it| it.pos),
        );

        let rew = reward::compute(
            &self.world_cfg.reward,
            &self.agents[0],
            self.prev_pos,
            self.prev_gold,
            self.prev_score,
            &self.gold_positions,
            wall_hit,
        );
        build_obs_into(&mut self.obs_buf, &self.agents[0], &self.items, &self.agents, &self.gold_positions, &self.grid);
        (rew, done)
    }

    pub fn obs_as_bytes(&self) -> &[u8] {
        // SAFETY: f32 has no invalid bit patterns; viewing as u8 is always valid.
        unsafe {
            std::slice::from_raw_parts(
                self.obs_buf.as_ptr() as *const u8,
                self.obs_buf.len() * std::mem::size_of::<f32>(),
            )
        }
    }

    pub fn match_ticks(&self) -> u64 { self.match_ticks }
}
