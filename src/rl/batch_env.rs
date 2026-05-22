// src/rl/batch_env.rs
//
// Rayon-parallel batch environment. Replaces SubprocVecEnv.
// One BatchEnv = N independent SimCores stepped in parallel.

use rayon::prelude::*;
use crate::item::ItemKind;
use crate::sim_core::SimCore;
use crate::world::tile::Tile;

pub struct BatchEnv {
    envs: Vec<SimCore>,
}

impl BatchEnv {
    pub fn new(n_envs: usize, config_path: String) -> Self {
        let envs = (0..n_envs).map(|_| SimCore::new(&config_path)).collect();
        Self { envs }
    }

    pub fn n_envs(&self) -> usize { self.envs.len() }

    pub fn reset_all(&mut self) -> Vec<Vec<f32>> {
        self.envs.par_iter_mut().map(|e| e.reset()).collect()
    }

    pub fn reset_env(&mut self, i: usize) -> Vec<f32> {
        self.envs[i].reset()
    }

    /// Step all envs in parallel. Returns (obs, rewards, dones).
    pub fn step_batch(&mut self, actions: &[u32]) -> (Vec<Vec<f32>>, Vec<f32>, Vec<bool>) {
        let results: Vec<(Vec<f32>, f32, bool)> = self.envs
            .par_iter_mut()
            .zip(actions.par_iter())
            .map(|(env, &a)| env.step(a))
            .collect();

        let mut obs   = Vec::with_capacity(results.len());
        let mut rews  = Vec::with_capacity(results.len());
        let mut dones = Vec::with_capacity(results.len());
        for (o, r, d) in results {
            obs.push(o); rews.push(r); dones.push(d);
        }
        (obs, rews, dones)
    }

    // ── Viewer state queries ──────────────────────────────────────────────────

    /// (width, height) of the grid.
    pub fn grid_size(&self) -> (usize, usize) {
        (self.envs[0].grid.width, self.envs[0].grid.height)
    }

    /// Flat tile array for env `i`, row-major (index = y*width + x).
    /// Encoding: 0=Free, 1=Obstacle, 10+team=Base, 20+team=SafeZone.
    pub fn get_tiles(&self, i: usize) -> Vec<u8> {
        self.envs[i].grid.iter().map(|(_, _, tile)| match tile {
            Tile::Free        => 0,
            Tile::Obstacle    => 1,
            Tile::Base(t)     => 10 + t,
            Tile::SafeZone(t) => 20 + t,
        }).collect()
    }

    /// Agent states: (x, y, team, gold_carried, score).
    pub fn get_agents(&self, i: usize) -> Vec<(i32, i32, u8, u8, u32)> {
        self.envs[i].agents.iter()
            .map(|a| (a.pos.x, a.pos.y, a.team, a.gold_carried, a.score))
            .collect()
    }

    /// Item states: (x, y, kind). Kind: 0=Gold, 1=Health, 2=Ammo, 3=Speed.
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
