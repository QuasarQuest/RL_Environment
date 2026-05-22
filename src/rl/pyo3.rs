// src/rl/pyo3.rs
//
// PyO3 bindings — exposes PyBatchEnv to Python.
//
// Python usage:
//
//   import atb
//   env = atb.PyBatchEnv(64, "assets/world/config.ron")
//   obs_list = env.reset_all()                    # list[list[float]]
//   obs_list, rews, dones = env.step_batch(actions)

use pyo3::prelude::*;
use super::batch_env::BatchEnv;
use super::action::ACTION_SIZE;
use super::obs::{OBS_DIM, OBS_SHAPE};

// ── PyBatchEnv ────────────────────────────────────────────────────────────────

/// Rayon-parallel batch environment. Replaces SubprocVecEnv.
#[pyclass]
pub struct PyBatchEnv {
    inner: BatchEnv,
}

#[pymethods]
impl PyBatchEnv {
    #[new]
    pub fn new(n_envs: usize, config_path: String) -> Self {
        Self { inner: BatchEnv::new(n_envs, config_path) }
    }

    pub fn n_envs(&self) -> usize { self.inner.n_envs() }

    pub fn reset_all(&mut self) -> Vec<Vec<f32>> {
        self.inner.reset_all()
    }

    pub fn reset_env(&mut self, i: usize) -> Vec<f32> {
        self.inner.reset_env(i)
    }

    /// Step all envs. Returns (obs, rewards, dones).
    pub fn step_batch(&mut self, actions: Vec<u32>) -> (Vec<Vec<f32>>, Vec<f32>, Vec<bool>) {
        self.inner.step_batch(&actions)
    }

    /// Step all envs and auto-reset any that finished.
    pub fn step_batch_auto_reset(
        &mut self,
        actions: Vec<u32>,
    ) -> (Vec<Vec<f32>>, Vec<f32>, Vec<bool>) {
        let (mut obs, rews, dones) = self.inner.step_batch(&actions);
        for (i, &done) in dones.iter().enumerate() {
            if done { obs[i] = self.inner.reset_env(i); }
        }
        (obs, rews, dones)
    }

    // ── Viewer state queries ──────────────────────────────────────────────────

    pub fn grid_size(&self) -> (usize, usize)                   { self.inner.grid_size() }
    pub fn get_tiles(&self, i: usize) -> Vec<u8>                { self.inner.get_tiles(i) }
    pub fn get_agents(&self, i: usize) -> Vec<(i32, i32, u8, u8, u32)> { self.inner.get_agents(i) }
    pub fn get_items(&self, i: usize) -> Vec<(i32, i32, u8)>   { self.inner.get_items(i) }
    pub fn get_tick(&self, i: usize) -> u64                     { self.inner.get_tick(i) }
    pub fn get_match_ticks(&self, i: usize) -> u64              { self.inner.get_match_ticks(i) }

    #[staticmethod]
    pub fn obs_dim() -> usize { OBS_DIM }
    #[staticmethod]
    pub fn obs_shape() -> (usize, usize, usize) { OBS_SHAPE }
    #[staticmethod]
    pub fn action_size() -> usize { ACTION_SIZE }
}

// ── Module registration ───────────────────────────────────────────────────────

#[pymodule]
pub fn atb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBatchEnv>()?;
    Ok(())
}
