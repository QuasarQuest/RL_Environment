// src/sim_core/mod.rs
//
// Pure Rust simulation core — no Bevy ECS.
//
// One SimCore = one independent episode stream. Designed to be stepped from a
// Rayon thread pool (BatchEnv). agents[0] is always the RL agent (team 0).
//
// Step order:
//   tick_speed_buffs → apply_action → pickup_items → auto_deposit → reward
//
// Performance design
// ------------------
// obs_buf          pre-allocated f32 slice, reused every step (no heap alloc)
// grid_snapshot    memcpy-fast tile reset instead of full Grid rebuild
// agents_snapshot  tiny Vec clone on reset
// gold_positions   pre-filtered Vec<GridPos> updated only on pickup — the obs
//                  and reward hot paths iterate only gold, not all items
// rng              SmallRng per SimCore for reproducible item spawns when a
//                  seed is set; episodes are independent across envs
// prev_pos         GridPos copy before each action for approach shaping

pub mod agent;
pub mod items;
pub mod obs;
pub mod reward;
pub mod state;
pub mod world_builder;

pub use state::{AgentState, ItemState};

use rand::SeedableRng;
use rand::rngs::{SmallRng, SysRng};

use crate::item::ItemKind;
use crate::rl::action::int_to_action;
use crate::rl::obs::OBS_TOTAL;
use crate::world::{config::WorldConfig, coords::GridPos, grid::Grid, tile::Tile};
use obs::build_obs_into;

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

    // Pre-filtered gold positions — kept in sync with items on every pickup.
    // Passed directly to obs::build_obs_into and reward::compute so neither
    // hot path needs to iterate or filter the full items list.
    gold_positions: Vec<GridPos>,

    // Per-env RNG for reproducible item spawns.
    rng: SmallRng,

    // Pre-allocated observation buffer.
    pub obs_buf: Vec<f32>,

    // Cached initial state for fast reset.
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
        let snap        = world_builder::build(&world_cfg);

        let grid_snapshot   = snap.grid.clone_tiles();
        let agents_snapshot = snap.agents.clone();
        let spawn_positions = snap.agents.iter().map(|a| a.pos).collect();

        let initial_pos = snap.agents[0].pos;

        let gold_positions: Vec<GridPos> = snap.items.iter()
            .filter(|it| it.kind == ItemKind::Gold)
            .map(|it| it.pos)
            .collect();

        let mut obs_buf = vec![0.0f32; OBS_TOTAL];
        build_obs_into(&mut obs_buf, &snap.agents[0], &gold_positions, &snap.grid);

        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None    => SmallRng::try_from_rng(&mut SysRng).expect("SysRng failed"),
        };

        Self {
            grid: snap.grid, agents: snap.agents, items: snap.items,
            tick: 0, match_ticks, world_cfg,
            prev_gold: 0, prev_score: 0, prev_pos: initial_pos,
            gold_positions,
            rng,
            obs_buf,
            grid_snapshot,
            agents_snapshot,
            spawn_positions,
        }
    }

    /// Reset to a fresh episode.
    pub fn reset(&mut self) {
        self.grid.restore_tiles(&self.grid_snapshot);
        self.agents.clone_from(&self.agents_snapshot);
        self.items = world_builder::spawn_items(
            &self.world_cfg,
            &self.grid,
            &self.spawn_positions,
            &mut self.rng,
        );
        // Rebuild gold_positions from the new item list.
        self.gold_positions.clear();
        self.gold_positions.extend(
            self.items.iter()
                .filter(|it| it.kind == ItemKind::Gold)
                .map(|it| it.pos),
        );
        self.tick       = 0;
        self.prev_gold  = 0;
        self.prev_score = 0;
        self.prev_pos   = self.agents[0].pos;
        build_obs_into(&mut self.obs_buf, &self.agents[0], &self.gold_positions, &self.grid);
    }

    /// Step one tick. Returns (reward, done).
    pub fn step(&mut self, action: u32) -> (f32, bool) {
        self.prev_gold  = self.agents[0].gold_carried;
        self.prev_score = self.agents[0].score;
        self.prev_pos   = self.agents[0].pos;

        self.tick += 1;
        let done = self.tick >= self.match_ticks;

        agent::tick_speed_buffs(&mut self.agents);
        agent::apply_action(&mut self.agents, &self.grid, 0, int_to_action(action));

        // Mirror gold removals into gold_positions before obs/reward.
        // items::pickup uses swap_remove — we replicate that here by position.
        let gold_before = self.agents[0].gold_carried;
        items::pickup(&mut self.agents, &mut self.items);
        let gold_after = self.agents[0].gold_carried;

        // If the agent picked up gold, remove the matching position from
        // gold_positions. We know it must be at agent.pos since pickup only
        // removes items the agent is standing on.
        if gold_after > gold_before {
            let agent_pos = self.agents[0].pos;
            if let Some(idx) = self.gold_positions.iter().position(|&p| p == agent_pos) {
                self.gold_positions.swap_remove(idx);
            }
        }

        agent::auto_deposit(&mut self.agents, &self.grid);

        let rew = reward::compute(
            &self.agents[0],
            self.prev_pos,
            self.prev_gold,
            self.prev_score,
            &self.gold_positions,
        );
        build_obs_into(&mut self.obs_buf, &self.agents[0], &self.gold_positions, &self.grid);
        (rew, done)
    }

    /// View obs_buf as raw bytes for zero-copy transfer to Python.
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