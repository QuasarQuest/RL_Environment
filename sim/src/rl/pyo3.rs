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
// Python usage:
//
//   import atb, numpy as np
//   env = atb.PyBatchEnv(64, "assets/world/config.ron")
//   obs_ba = env.reset_all()                              # bytearray
//   obs = np.frombuffer(obs_ba, dtype=np.float32).reshape(64, *env.obs_shape())
//
//   obs_ba, rews, dones = env.step_batch(actions_list)   # bytearray, list[float], list[bool]

use pyo3::prelude::*;
use pyo3::types::PyByteArray;

use super::batch_env::BatchEnv;
use super::action::ACTION_SIZE;
use super::obs::{OBS_DIM, OBS_SHAPE, OBS_TOTAL};

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
    #[new]
    pub fn new(n_envs: usize, config_path: String) -> Self {
        Self { inner: BatchEnv::new(n_envs, config_path) }
    }

    pub fn n_envs(&self) -> usize { self.inner.n_envs() }

    // ── Reset ─────────────────────────────────────────────────────────────────

    /// Reset all envs. Returns flat bytearray of shape [n_envs * OBS_TOTAL * 4 bytes].
    /// Python: np.frombuffer(ba, dtype=np.float32).reshape(n_envs, *obs_shape)
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

    // ── Step ──────────────────────────────────────────────────────────────────

    /// Step all envs. Returns (obs_bytearray, rewards, dones).
    /// obs_bytearray: flat [n_envs * OBS_TOTAL * 4 bytes]; reshape to (n_envs, *obs_shape).
    pub fn step_batch<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<u32>,
    ) -> (Bound<'py, PyByteArray>, Vec<f32>, Vec<bool>) {
        let (rews, dones) = self.inner.step_batch(&actions);
        let obs = Self::slice_to_bytearray(py, &self.inner.obs_flat);
        (obs, rews, dones)
    }

    /// Step all envs and auto-reset any that finished.
    pub fn step_batch_auto_reset<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<u32>,
    ) -> (Bound<'py, PyByteArray>, Vec<f32>, Vec<bool>) {
        let (rews, dones) = self.inner.step_batch(&actions);
        for (i, &done) in dones.iter().enumerate() {
            if done { self.inner.reset_env(i); }
        }
        let obs = Self::slice_to_bytearray(py, &self.inner.obs_flat);
        (obs, rews, dones)
    }

    // ── Viewer state queries ──────────────────────────────────────────────────

    pub fn grid_size(&self) -> (usize, usize)                         { self.inner.grid_size() }
    pub fn get_tiles(&self, i: usize) -> Vec<u8>                      { self.inner.get_tiles(i) }
    pub fn get_agents(&self, i: usize) -> Vec<(i32, i32, u8, u8, u32)> { self.inner.get_agents(i) }
    pub fn get_items(&self, i: usize) -> Vec<(i32, i32, u8)>          { self.inner.get_items(i) }
    pub fn get_tick(&self, i: usize) -> u64                            { self.inner.get_tick(i) }
    pub fn get_match_ticks(&self, i: usize) -> u64                     { self.inner.get_match_ticks(i) }

    #[staticmethod] pub fn obs_dim()   -> usize                  { OBS_DIM }
    #[staticmethod] pub fn obs_shape() -> (usize, usize, usize)  { OBS_SHAPE }
    #[staticmethod] pub fn action_size() -> usize                { ACTION_SIZE }
}

// ── Module registration ───────────────────────────────────────────────────────

#[pymodule]
pub fn atb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBatchEnv>()?;
    Ok(())
}
