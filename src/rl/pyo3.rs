// src/rl/pyo3.rs
//
// PyO3 bindings — exposes RlEnv as `PyRlEnv` to Python.
//
// Python usage:
//
//   import atb
//   env = atb.PyRlEnv("assets/world/config.ron")           # default map
//   env = atb.PyRlEnv("assets/world/config_stage1.ron")    # small training map
//
//   obs = env.reset()               # list[float], len == OBS_DIM
//   obs, reward, done = env.step(4) # action int 0..ACTION_SIZE
//
// Thread safety:
//   PyRlEnv is not Send (Bevy App contains raw pointers). It must stay on
//   the thread that created it. For parallel training, spawn separate
//   Python processes each owning one PyRlEnv — do not share across threads.

use pyo3::prelude::*;
use super::env::{RlEnv, ACTION_SIZE, OBS_DIM};

// ── PyRlEnv ───────────────────────────────────────────────────────────────────

/// Headless Bevy RL environment. One instance = one episode stream.
#[pyclass(unsendable)]
pub struct PyRlEnv {
    inner: RlEnv,
}

#[pymethods]
impl PyRlEnv {
    /// Construct and initialise a headless simulation.
    ///
    /// Args:
    ///   config_path (str): path to the RON world config file.
    ///     The path is resolved by walking up from CWD, so a relative path
    ///     like "assets/world/config.ron" works from any project subdirectory.
    #[new]
    pub fn new(config_path: String) -> Self {
        Self { inner: RlEnv::new(config_path) }
    }

    /// Reset to a fresh episode. Returns the initial observation vector.
    ///
    /// Returns:
    ///   list[float] — length OBS_DIM
    pub fn reset(&mut self) -> Vec<f32> {
        self.inner.reset()
    }

    /// Step the simulation by one tick.
    ///
    /// Args:
    ///   action (int): discrete action index, 0 ≤ action < ACTION_SIZE
    ///
    /// Returns:
    ///   tuple[list[float], float, bool] — (obs, reward, done)
    pub fn step(&mut self, action: u32) -> (Vec<f32>, f32, bool) {
        self.inner.step(action)
    }

    /// Number of floats in the observation vector.
    #[staticmethod]
    pub fn obs_dim() -> usize { OBS_DIM }

    /// Number of discrete actions.
    #[staticmethod]
    pub fn action_size() -> usize { ACTION_SIZE }
}

// ── Module registration ───────────────────────────────────────────────────────

#[pymodule]
pub fn atb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRlEnv>()?;
    Ok(())
}