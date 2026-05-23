// src/rl/batch_env.rs
//
// Rayon-parallel batch environment.
//
// Performance design:
//   - obs_flat: pre-allocated contiguous buffer (n_envs × OBS_TOTAL f32s).
//     Rayon steps each env in parallel (each env writes into its own obs_buf),
//     then a serial gather copies them into obs_flat in one pass.
//   - No Vec<Vec<f32>> returned — callers read obs_flat directly or via bytes.

use rayon::prelude::*;
use crate::item::ItemKind;
use crate::rl::obs::OBS_TOTAL;
use crate::sim_core::SimCore;
use crate::world::tile::Tile;

pub struct BatchEnv {
    pub envs:     Vec<SimCore>,
    pub obs_flat: Vec<f32>,   // n_envs * OBS_TOTAL, pre-allocated
}

impl BatchEnv {
    pub fn new(n_envs: usize, config_path: String) -> Self {
        let envs     = (0..n_envs).map(|_| SimCore::new(&config_path)).collect();
        let obs_flat = vec![0.0f32; n_envs * OBS_TOTAL];
        Self { envs, obs_flat }
    }

    pub fn n_envs(&self) -> usize { self.envs.len() }

    // ── Reset ─────────────────────────────────────────────────────────────────

    pub fn reset_all(&mut self) {
        self.envs.par_iter_mut().for_each(|e| e.reset());
        self.gather_obs();
    }

    /// Reset a single env and update its slot in obs_flat.
    pub fn reset_env(&mut self, i: usize) {
        self.envs[i].reset();
        let start = i * OBS_TOTAL;
        self.obs_flat[start..start + OBS_TOTAL].copy_from_slice(&self.envs[i].obs_buf);
    }

    // ── Step ──────────────────────────────────────────────────────────────────

    /// Step all envs in parallel. Returns (rewards, dones).
    /// Observation data is written into self.obs_flat (read via obs_flat()).
    pub fn step_batch(&mut self, actions: &[u32]) -> (Vec<f32>, Vec<bool>) {
        let results: Vec<(f32, bool)> = self.envs
            .par_iter_mut()
            .zip(actions.par_iter())
            .map(|(env, &a)| env.step(a))
            .collect();

        self.gather_obs();

        let rews:  Vec<f32>  = results.iter().map(|(r, _)| *r).collect();
        let dones: Vec<bool> = results.iter().map(|(_, d)| *d).collect();
        (rews, dones)
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Copy each env's obs_buf into the corresponding slot in obs_flat.
    fn gather_obs(&mut self) {
        for (i, env) in self.envs.iter().enumerate() {
            let start = i * OBS_TOTAL;
            self.obs_flat[start..start + OBS_TOTAL].copy_from_slice(&env.obs_buf);
        }
    }

    // ── Viewer state queries ──────────────────────────────────────────────────

    pub fn grid_size(&self) -> (usize, usize) {
        (self.envs[0].grid.width, self.envs[0].grid.height)
    }

    pub fn get_tiles(&self, i: usize) -> Vec<u8> {
        self.envs[i].grid.iter().map(|(_, _, tile)| match tile {
            Tile::Free        => 0,
            Tile::Obstacle    => 1,
            Tile::Base(t)     => 10 + t,
            Tile::SafeZone(t) => 20 + t,
        }).collect()
    }

    pub fn get_agents(&self, i: usize) -> Vec<(i32, i32, u8, u8, u32)> {
        self.envs[i].agents.iter()
            .map(|a| (a.pos.x, a.pos.y, a.team, a.gold_carried, a.score))
            .collect()
    }

    pub fn get_items(&self, i: usize) -> Vec<(i32, i32, u8)> {
        self.envs[i].items.iter()
            .map(|it| (it.pos.x, it.pos.y, match it.kind {
                ItemKind::Gold       => 0,
                ItemKind::Health     => 1,
                ItemKind::Ammo       => 2,
                ItemKind::SpeedBoost => 3,
            }))
            .collect()
    }

    pub fn get_tick(&self, i: usize) -> u64        { self.envs[i].tick }
    pub fn get_match_ticks(&self, i: usize) -> u64 { self.envs[i].match_ticks() }
}
