// src/rl/env.rs
//
// Rayon-parallel batch environment.
//
// Performance design:
//   - obs_flat: pre-allocated contiguous buffer (n_envs × OBS_TOTAL f32s).
//   - step_batch: each Rayon thread steps its env AND copies obs_buf directly
//     into its slot in obs_flat in one pass — no serial gather phase.
//   - rews/dones: written into pre-allocated Vecs via par_iter_mut — no
//     intermediate Vec<(f32,bool)> allocation per step.
//   - reset_all: same pattern — parallel reset + parallel obs copy.
//
// Previous design had three serial phases after the parallel sim step:
//   1. gather_obs()            — serial memcpy of all obs_bufs into obs_flat
//   2. results.iter().map(r)   — serial unzip into rews Vec
//   3. results.iter().map(d)   — serial unzip into dones Vec
// All three are now absorbed into the single parallel step pass.
//
// Scaling note: gather was O(n_envs × OBS_TOTAL) serial memcpy. At
// n_envs=256 and OBS_TOTAL=3750 that's ~3.8M f32 copies (~15MB) on a
// single core. The new design distributes this across all Rayon threads,
// with each thread writing into its own non-overlapping obs_flat slice —
// no false sharing, no synchronisation needed.

use rayon::prelude::*;
use crate::item::ItemKind;
use crate::rl::obs::OBS_TOTAL;
use crate::sim_core::SimCore;
use crate::world::tile::Tile;

pub struct BatchEnv {
    pub envs:     Vec<SimCore>,
    pub obs_flat: Vec<f32>,    // n_envs * OBS_TOTAL, pre-allocated
    rews:         Vec<f32>,    // pre-allocated reward buffer
    dones:        Vec<bool>,   // pre-allocated done buffer
}

impl BatchEnv {
    pub fn new(n_envs: usize, config_path: String) -> Self {
        let envs     = (0..n_envs).map(|_| SimCore::new(&config_path)).collect();
        let obs_flat = vec![0.0f32; n_envs * OBS_TOTAL];
        let rews     = vec![0.0f32; n_envs];
        let dones    = vec![false;  n_envs];
        Self { envs, obs_flat, rews, dones }
    }

    pub fn n_envs(&self) -> usize { self.envs.len() }

    // ── Reset ──────────────────────────────────────────────────────────────────

    /// Reset all envs in parallel and write obs into obs_flat in the same pass.
    pub fn reset_all(&mut self) {
        self.envs
            .par_iter_mut()
            .zip(self.obs_flat.par_chunks_mut(OBS_TOTAL))
            .for_each(|(env, slot)| {
                env.reset();
                slot.copy_from_slice(&env.obs_buf);
            });
    }

    /// Reset a single env and update its slot in obs_flat.
    /// Called serially from Python on individual done envs — no parallelism needed.
    pub fn reset_env(&mut self, i: usize) {
        self.envs[i].reset();
        let start = i * OBS_TOTAL;
        self.obs_flat[start..start + OBS_TOTAL].copy_from_slice(&self.envs[i].obs_buf);
    }

    // ── Step ───────────────────────────────────────────────────────────────────

    /// Step all envs in parallel. Returns (&rews, &dones).
    ///
    /// Each Rayon thread:
    ///   1. steps its SimCore  (sim logic + obs_buf written inside env.step())
    ///   2. copies obs_buf → its slice of obs_flat  (cache-hot, same thread)
    ///   3. writes reward and done into pre-allocated output slices
    ///
    /// No serial gather pass, no intermediate Vec<(f32,bool)> allocation.
    pub fn step_batch(&mut self, actions: &[u32]) -> (&[f32], &[bool]) {
        self.envs
            .par_iter_mut()
            .zip(actions.par_iter())
            .zip(self.obs_flat.par_chunks_mut(OBS_TOTAL))
            .zip(self.rews.par_iter_mut())
            .zip(self.dones.par_iter_mut())
            .for_each(|((((env, &a), obs_slot), rew), done)| {
                let (r, d) = env.step(a);
                obs_slot.copy_from_slice(&env.obs_buf);
                *rew  = r;
                *done = d;
            });

        (&self.rews, &self.dones)
    }

    // ── Viewer state queries ───────────────────────────────────────────────────

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