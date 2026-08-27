// src/rl/pyo3.rs
//
// PyO3 bindings — exposes PyBatchEnv to Python.
//
// Observation protocol (replaces list[list[float]]):
//   Rust writes obs into a pre-allocated flat f32 buffer, then reinterprets
//   those bytes as a Python bytearray. Python calls np.frombuffer(ba, dtype=np.float32)
//   to get a zero-extra-copy numpy array. A single memcpy per step call
//   instead of ~200K Python float object allocations.
//
//   IMPORTANT: the per-env stride is OBS_TOTAL (crop + minimap + cluster
//   features), NOT obs_dim() (which is the egocentric crop only). Always slice
//   with obs_total(). obs_shape()/mm_shape() describe the crop and minimap
//   sub-tensors; the consumer (AtbCnnExtractor) splits the flat buffer with:
//       crop    = flat[:, 0 : OBS_DIM         ].reshape(n, *obs_shape())
//       minimap = flat[:, OBS_DIM : OBS_DIM+MM].reshape(n, *mm_shape())
//       cluster = flat[:, -cluster_features() :]
//
// Reward/done protocol:
//   step_batch returns (&[f32], &[bool]) slices into pre-allocated BatchEnv
//   buffers — no Vec<(f32,bool)> intermediate allocation, no serial unzip.
//   pyo3.rs converts those slices to Vec<f32>/Vec<bool> at the FFI boundary,
//   which is unavoidable (Python needs owned data), but the Rust-side work is
//   fully parallel with no serial passes.
//
// Python usage:
//
//   import atb, numpy as np
//   env  = atb.PyBatchEnv(64, "assets/world/config.ron")
//   N, T = env.n_envs(), atb.PyBatchEnv.obs_total()
//   obs  = np.frombuffer(env.reset_all(), dtype=np.float32).reshape(N, T)
//   obs_ba, rews, dones = env.step_batch(actions_list)   # bytearray, list[float], list[bool]
//   obs  = np.frombuffer(obs_ba, dtype=np.float32).reshape(N, T)

use pyo3::prelude::*;
use pyo3::types::PyByteArray;

use super::env::BatchEnv;
use super::action::ACTION_SIZE;
use super::obs::{CLUSTER_FEATURES, MM_SHAPE, OBS_DIM, OBS_SHAPE, OBS_TOTAL};

// ── PyBatchEnv ────────────────────────────────────────────────────────────────

#[pyclass]
pub struct PyBatchEnv {
    inner: BatchEnv,
}

impl PyBatchEnv {
    /// Reinterpret a &[f32] slice as raw bytes and wrap in a Python bytearray.
    /// One memcpy into Python-owned memory; no Python float object allocations.
    fn slice_to_bytearray<'py>(py: Python<'py>, floats: &[f32]) -> Bound<'py, PyByteArray> {
        // SAFETY: f32 has no invalid bit patterns; reinterpreting as u8 is always valid.
        let bytes = unsafe {
            std::slice::from_raw_parts(floats.as_ptr() as *const u8, floats.len() * 4)
        };
        PyByteArray::new(py, bytes)
    }
}

#[pymethods]
impl PyBatchEnv {
    /// `seed`: base RNG seed — env `i` is seeded with `seed + i` for reproducible
    /// batches. Omit for OS entropy (non-reproducible).
    #[new]
    #[pyo3(signature = (n_envs, config_path, seed=None))]
    pub fn new(n_envs: usize, config_path: String, seed: Option<u64>) -> Self {
        Self { inner: BatchEnv::new(n_envs, config_path, seed) }
    }

    pub fn n_envs(&self) -> usize { self.inner.n_envs() }

    // ── Reset ─────────────────────────────────────────────────────────────────

    /// Reset all envs in parallel. Returns flat bytearray [n_envs * OBS_TOTAL * 4 bytes].
    /// Python: np.frombuffer(ba, dtype=np.float32).reshape(n_envs, obs_total())
    pub fn reset_all<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyByteArray> {
        self.inner.reset_all();
        Self::slice_to_bytearray(py, &self.inner.obs_flat)
    }

    /// Reset one env. Returns bytearray for that env's obs (OBS_TOTAL * 4 bytes).
    pub fn reset_env<'py>(&mut self, py: Python<'py>, i: usize) -> Bound<'py, PyByteArray> {
        self.inner.reset_env(i);
        let start = i * OBS_TOTAL;
        Self::slice_to_bytearray(py, &self.inner.obs_flat[start..start + OBS_TOTAL])
    }

    /// Reset several envs in one FFI call — collapses what would otherwise be one
    /// reset_env() call per done env per step_wait (measured as real overhead: 8,869
    /// calls / ~1.07s in a 300k-timestep profile). Returns a compact bytearray of just
    /// the given envs' post-reset obs, in `indices` order (len(indices) * OBS_TOTAL * 4
    /// bytes) — caller scatters each chunk back into obs[indices[k]].
    pub fn reset_batch<'py>(&mut self, py: Python<'py>, indices: Vec<usize>) -> Bound<'py, PyByteArray> {
        self.inner.reset_batch(&indices);
        let mut out = Vec::with_capacity(indices.len() * OBS_TOTAL);
        for &i in &indices {
            let start = i * OBS_TOTAL;
            out.extend_from_slice(&self.inner.obs_flat[start..start + OBS_TOTAL]);
        }
        Self::slice_to_bytearray(py, &out)
    }

    // ── Step ──────────────────────────────────────────────────────────────────

    /// Step all envs in parallel. Returns (obs_bytearray, rewards, dones).
    ///
    /// obs_bytearray : flat [n_envs * OBS_TOTAL * 4 bytes]; reshape to (n_envs, obs_total()).
    /// rewards       : list[float] — copied from BatchEnv's pre-allocated reward buffer.
    /// dones         : list[bool]  — copied from BatchEnv's pre-allocated done buffer.
    ///
    /// The Vec conversions here are unavoidable — Python needs owned data.
    pub fn step_batch<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<u32>,
    ) -> (Bound<'py, PyByteArray>, Vec<f32>, Vec<bool>) {
        let (rews, dones) = self.inner.step_batch(&actions);
        let rews  = rews.to_vec();
        let dones = dones.to_vec();
        let obs   = Self::slice_to_bytearray(py, &self.inner.obs_flat);
        (obs, rews, dones)
    }

    /// Step all envs and auto-reset any that finished.
    /// Useful for the viewer — training uses step_batch + manual reset_env instead.
    pub fn step_batch_auto_reset<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<u32>,
    ) -> (Bound<'py, PyByteArray>, Vec<f32>, Vec<bool>) {
        let (rews, dones) = self.inner.step_batch(&actions);
        // Clone before reset_env borrows self.inner mutably.
        let rews_out:  Vec<f32>  = rews.to_vec();
        let dones_out: Vec<bool> = dones.to_vec();
        for (i, &done) in dones_out.iter().enumerate() {
            if done { self.inner.reset_env(i); }
        }
        let obs = Self::slice_to_bytearray(py, &self.inner.obs_flat);
        (obs, rews_out, dones_out)
    }

    /// Per-env action masks, flattened row-major (n_envs × ACTION_SIZE).
    /// Python reshapes to (n_envs, ACTION_SIZE) for MaskablePPO. Reflects the
    /// state after the most recent step_batch / reset.
    pub fn action_masks(&self) -> Vec<bool> {
        self.inner.action_masks().to_vec()
    }

    /// Per-env decision telemetry from the most recent step_batch:
    /// (chosen_gold_distance, is_cluster_action, own_region_had_gold, skipped_own_region).
    /// Used by PolicyTelemetryCallback to log chosen-region distance and own-region
    /// skip rate to TensorBoard.
    pub fn decision_telemetry(&self) -> (Vec<i32>, Vec<bool>, Vec<bool>, Vec<bool>) {
        let (d, c, o, s) = self.inner.decision_telemetry();
        (d.to_vec(), c.to_vec(), o.to_vec(), s.to_vec())
    }

    /// Per-env option length (sim ticks) from the most recent step_batch.
    /// Used by the SMDP rollout buffer to apply a γ^k cross-option discount.
    pub fn option_ticks(&self) -> Vec<u64> {
        self.inner.option_ticks().to_vec()
    }

    // ── Trace capture (offline recorder; off by default) ────────────────────────

    /// Enable/disable per-tick trace capture on all envs. Off by default — the
    /// training path pays nothing. Used by the single-env trace recorder.
    pub fn set_trace(&mut self, on: bool) {
        self.inner.set_trace(on);
    }

    /// Flat per-tick trace for env `i`'s most recent option, row-major with
    /// `trace_fields()` columns per tick. Python: np.asarray(...).reshape(-1, F).
    /// Column order: tick, ax, ay, gold_carried, score, r_tick, r_pickup,
    /// r_deposit, r_wall, r_total, discount, gold_count.
    pub fn get_trace(&self, i: usize) -> Vec<f32> {
        self.inner.trace_flat(i)
    }

    /// Reward weights: (tick, pickup, deposit, wall_hit, option_gamma).
    pub fn reward_weights(&self) -> (f32, f32, f32, f32, f32) {
        self.inner.reward_weights()
    }

    // ── Viewer state queries ──────────────────────────────────────────────────

    pub fn grid_size(&self) -> (usize, usize)                           { self.inner.grid_size() }
    pub fn get_tiles(&self, i: usize) -> Vec<u8>                        { self.inner.get_tiles(i) }
    pub fn get_agents(&self, i: usize) -> Vec<(i32, i32, u8, u8, u32)> { self.inner.get_agents(i) }
    pub fn get_items(&self, i: usize) -> Vec<(i32, i32, u8)>            { self.inner.get_items(i) }
    pub fn get_tick(&self, i: usize) -> u64                              { self.inner.get_tick(i) }
    pub fn get_match_ticks(&self, i: usize) -> u64                       { self.inner.get_match_ticks(i) }

    // ── Layout accessors (single source of truth for the Python side) ──────────
    // obs_total() is the per-env buffer stride. obs_dim()/obs_shape() describe the
    // egocentric crop ONLY; do not use them to compute the stride.

    #[staticmethod] pub fn obs_dim()          -> usize                 { OBS_DIM }
    #[staticmethod] pub fn obs_total()        -> usize                 { OBS_TOTAL }
    #[staticmethod] pub fn obs_shape()        -> (usize, usize, usize) { OBS_SHAPE }
    #[staticmethod] pub fn mm_shape()         -> (usize, usize, usize) { MM_SHAPE }
    #[staticmethod] pub fn cluster_features() -> usize                 { CLUSTER_FEATURES }
    #[staticmethod] pub fn action_size()      -> usize                 { ACTION_SIZE }
    #[staticmethod] pub fn trace_fields()     -> usize                 { crate::engine::TRACE_FIELDS }
}

// ── Module registration ───────────────────────────────────────────────────────

#[pymodule]
pub fn atb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBatchEnv>()?;
    Ok(())
}