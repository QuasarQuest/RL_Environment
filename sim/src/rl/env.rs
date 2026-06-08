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
// Scaling note: gather was O(n_envs × OBS_TOTAL) serial memcpy. With
// OBS_TOTAL = 10222 (8750 crop + 1445 minimap + 27 cluster) at n_envs = 256
// that is ~2.6M f32 copies (~10.5MB) per step. The obs copy is now distributed
// across Rayon threads, each writing its own non-overlapping OBS_TOTAL-float
// slice of obs_flat (well above a cache line, so no false sharing on obs; the
// adjacent single-element writes into rews/dones can share a line only at the
// handful of thread-chunk boundaries — negligible vs the sim step).

use rayon::prelude::*;
use crate::entity::item::ItemKind;
use crate::rl::action::ACTION_SIZE;
use crate::rl::obs::OBS_TOTAL;
use crate::engine::SimCore;
use crate::world::tile::Tile;

pub struct BatchEnv {
    pub envs:     Vec<SimCore>,
    pub obs_flat: Vec<f32>,    // n_envs * OBS_TOTAL, pre-allocated
    masks:        Vec<bool>,   // n_envs * ACTION_SIZE, pre-allocated (MaskablePPO)
    rews:         Vec<f32>,    // pre-allocated reward buffer
    dones:        Vec<bool>,   // pre-allocated done buffer

    // Per-env decision telemetry from the most recent step_batch (see
    // SimCore::DecisionTelem). Gathered serially after the parallel step — n_envs
    // is small, so this is negligible vs the sim work.
    dec_dist:         Vec<i32>,
    dec_is_cluster:   Vec<bool>,
    dec_own_has_gold: Vec<bool>,
    dec_skipped:      Vec<bool>,

    // Per-env option length (sim ticks) from the most recent step — drives the
    // SMDP γ^k cross-option discount on the Python side.
    option_ticks: Vec<u64>,
}

impl BatchEnv {
    pub fn new(n_envs: usize, config_path: String) -> Self {
        let envs     = (0..n_envs).map(|_| SimCore::new(&config_path)).collect();
        let obs_flat = vec![0.0f32; n_envs * OBS_TOTAL];
        let masks    = vec![false; n_envs * ACTION_SIZE];
        let rews     = vec![0.0f32; n_envs];
        let dones    = vec![false;  n_envs];
        let mut this = Self {
            envs, obs_flat, masks, rews, dones,
            dec_dist:         vec![-1; n_envs],
            dec_is_cluster:   vec![false; n_envs],
            dec_own_has_gold: vec![false; n_envs],
            dec_skipped:      vec![false; n_envs],
            option_ticks:     vec![1; n_envs],
        };
        this.refresh_masks();
        this
    }

    pub fn n_envs(&self) -> usize { self.envs.len() }

    /// Current per-env action masks, flattened (n_envs × ACTION_SIZE row-major).
    pub fn action_masks(&self) -> &[bool] { &self.masks }

    /// Recompute every env's action mask from its current state.
    fn refresh_masks(&mut self) {
        self.envs
            .par_iter()
            .zip(self.masks.par_chunks_mut(ACTION_SIZE))
            .for_each(|(env, slot)| slot.copy_from_slice(&env.action_mask()));
    }

    // ── Reset ──────────────────────────────────────────────────────────────────

    /// Reset all envs in parallel and write obs + masks into the flat buffers.
    pub fn reset_all(&mut self) {
        self.envs
            .par_iter_mut()
            .zip(self.obs_flat.par_chunks_mut(OBS_TOTAL))
            .zip(self.masks.par_chunks_mut(ACTION_SIZE))
            .for_each(|((env, obs_slot), mask_slot)| {
                env.reset();
                obs_slot.copy_from_slice(&env.obs_buf);
                mask_slot.copy_from_slice(&env.action_mask());
            });
    }

    /// Reset a single env and update its slot in obs_flat + masks.
    /// Called serially from Python on individual done envs — no parallelism needed.
    pub fn reset_env(&mut self, i: usize) {
        self.envs[i].reset();
        let start = i * OBS_TOTAL;
        self.obs_flat[start..start + OBS_TOTAL].copy_from_slice(&self.envs[i].obs_buf);
        let mstart = i * ACTION_SIZE;
        self.masks[mstart..mstart + ACTION_SIZE].copy_from_slice(&self.envs[i].action_mask());
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
        // A short `actions` would make the zip silently truncate, leaving the
        // tail envs unstepped with stale rews/dones — catch that in debug builds.
        debug_assert_eq!(
            actions.len(), self.envs.len(),
            "actions length ({}) must equal n_envs ({})", actions.len(), self.envs.len()
        );

        self.envs
            .par_iter_mut()
            .zip(actions.par_iter())
            .zip(self.obs_flat.par_chunks_mut(OBS_TOTAL))
            .zip(self.masks.par_chunks_mut(ACTION_SIZE))
            .zip(self.rews.par_iter_mut())
            .zip(self.dones.par_iter_mut())
            .for_each(|(((((env, &a), obs_slot), mask_slot), rew), done)| {
                // Temporally-extended option execution (semi-MDP). See SimCore::step_option.
                let (r, d) = env.step_option(a);
                obs_slot.copy_from_slice(&env.obs_buf);
                mask_slot.copy_from_slice(&env.action_mask());
                *rew  = r;
                *done = d;
            });

        // Gather per-env decision telemetry + option length from this step
        // (serial; n_envs small).
        for (i, env) in self.envs.iter().enumerate() {
            let t = env.last_decision();
            self.dec_dist[i]         = t.chosen_dist;
            self.dec_is_cluster[i]   = t.is_cluster;
            self.dec_own_has_gold[i] = t.own_has_gold;
            self.dec_skipped[i]      = t.skipped_own;
            self.option_ticks[i]     = env.last_option_ticks();
        }

        (&self.rews, &self.dones)
    }

    /// Per-env decision telemetry from the most recent `step_batch`:
    /// (chosen_gold_distance, is_cluster_action, own_region_had_gold, skipped_own_region).
    pub fn decision_telemetry(&self) -> (&[i32], &[bool], &[bool], &[bool]) {
        (&self.dec_dist, &self.dec_is_cluster, &self.dec_own_has_gold, &self.dec_skipped)
    }

    /// Per-env option length (sim ticks) from the most recent `step_batch`.
    pub fn option_ticks(&self) -> &[u64] { &self.option_ticks }

    // ── Trace capture (offline recorder; off by default) ─────────────────────────

    /// Enable/disable per-tick trace capture on every env.
    pub fn set_trace(&mut self, on: bool) {
        for env in &mut self.envs { env.set_trace(on); }
    }

    /// Flatten env `i`'s most-recent-option trace into a row-major f32 buffer
    /// (`TRACE_FIELDS` columns per tick). Empty unless tracing is enabled.
    pub fn trace_flat(&self, i: usize) -> Vec<f32> {
        let trace = self.envs[i].trace();
        let mut out = Vec::with_capacity(trace.len() * crate::engine::TRACE_FIELDS);
        for r in trace {
            out.push(r.tick as f32);
            out.push(r.ax as f32);
            out.push(r.ay as f32);
            out.push(r.gold_carried as f32);
            out.push(r.score as f32);
            out.push(r.r_tick);
            out.push(r.r_pickup);
            out.push(r.r_deposit);
            out.push(r.r_wall);
            out.push(r.r_total);
            out.push(r.discount);
            out.push(r.gold_count as f32);
        }
        out
    }

    /// Reward weights (tick, pickup, deposit, wall_hit, option_gamma) from env 0.
    pub fn reward_weights(&self) -> (f32, f32, f32, f32, f32) {
        self.envs[0].reward_weights()
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
                ItemKind::Speed      => 1,
                ItemKind::Multiplier => 4,
                ItemKind::Trap       => 5,
            }))
            .collect()
    }

    pub fn get_tick(&self, i: usize) -> u64        { self.envs[i].tick }
    pub fn get_match_ticks(&self, i: usize) -> u64 { self.envs[i].match_ticks() }
}