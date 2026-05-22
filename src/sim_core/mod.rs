// src/sim_core/mod.rs
//
// Pure Rust simulation core — no Bevy ECS.
//
// One SimCore = one independent episode stream. Designed to be stepped from a
// Rayon thread pool (BatchEnv). agents[0] is always the RL agent (team 0);
// agents[1..] are opponent stubs (currently Wait) for future multi-agent use.
//
// Step order mirrors the old Bevy OnSimTick schedule:
//   tick_speed_buffs → apply_action → pickup_items → auto_deposit

pub mod agent;
pub mod items;
pub mod obs;
pub mod reward;
pub mod state;
pub mod world_builder;

pub use state::{AgentState, ItemState};

use crate::rl::action::int_to_action;
use crate::world::{config::WorldConfig, grid::Grid};

pub struct SimCore {
    pub grid:    Grid,
    pub agents:  Vec<AgentState>,
    pub items:   Vec<ItemState>,
    pub tick:    u64,
    match_ticks: u64,
    world_cfg:   WorldConfig,
    prev_gold:   u8,
    prev_score:  u32,
}

impl SimCore {
    pub fn new(config_path: &str) -> Self {
        let world_cfg   = WorldConfig::load(config_path);
        let match_ticks = world_cfg.match_duration_ticks;
        let snap        = world_builder::build(&world_cfg);
        Self {
            grid: snap.grid, agents: snap.agents, items: snap.items,
            tick: 0, match_ticks, world_cfg, prev_gold: 0, prev_score: 0,
        }
    }

    /// Reset to a fresh episode. Returns the initial observation.
    pub fn reset(&mut self) -> Vec<f32> {
        let snap        = world_builder::build(&self.world_cfg);
        self.grid       = snap.grid;
        self.agents     = snap.agents;
        self.items      = snap.items;
        self.tick       = 0;
        self.prev_gold  = 0;
        self.prev_score = 0;
        self.obs()
    }

    /// Step one tick. Returns (obs, reward, done).
    pub fn step(&mut self, action: u32) -> (Vec<f32>, f32, bool) {
        self.prev_gold  = self.agents[0].gold_carried;
        self.prev_score = self.agents[0].score;

        self.tick += 1;
        let done = self.tick >= self.match_ticks;

        agent::tick_speed_buffs(&mut self.agents);
        agent::apply_action(&mut self.agents, &self.grid, 0, int_to_action(action));
        // Future: call opponent AI for agents[1..]

        items::pickup(&mut self.agents, &mut self.items);
        agent::auto_deposit(&mut self.agents, &self.grid);

        let rew = reward::compute(&self.agents[0], self.prev_gold, self.prev_score);
        (self.obs(), rew, done)
    }

    /// CNN observation for the RL agent (agents[0]).
    pub fn obs(&self) -> Vec<f32> {
        obs::build_obs(&self.agents[0], &self.items, &self.grid)
    }

    pub fn match_ticks(&self) -> u64 { self.match_ticks }
}
