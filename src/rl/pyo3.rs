// src/rl/pyo3.rs
//
// PyO3 bindings — exposes RlEnv as `PyRlEnv` to Python.
//
// Python usage:
//
//   import atb                      # the compiled cdylib
//   env = atb.PyRlEnv()
//   obs = env.reset()               # list[float], len == OBS_DIM
//   obs, reward, done = env.step(4) # action int in 0..ACTION_SIZE
//
// Gymnasium wrapper (rl/src/env.py) should wrap this class rather than
// importing atb directly in training code.
//
// Thread safety:
//   PyRlEnv is not Send (Bevy App contains raw pointers). It must stay on
//   the thread that created it. For parallel training, spawn separate
//   Python processes each owning one PyRlEnv — do not share across threads.
//   pyo3 will enforce this via the Unsendable marker (implicit for non-Send).

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
    /// Runs Bevy Startup systems (loads config.ron, spawns agents, grid).
    #[new]
    pub fn new() -> Self {
        Self { inner: RlEnv::new() }
    }

    /// Reset to a fresh episode. Returns the initial observation vector.
    ///
    /// Returns:
    ///   list[float] — length OBS_DIM (53)
    pub fn reset(&mut self) -> Vec<f32> {
        self.inner.reset()
    }

    /// Step the simulation by one tick.
    ///
    /// Args:
    ///   action (int): discrete action index, 0 ≤ action < ACTION_SIZE (26)
    ///
    /// Returns:
    ///   tuple[list[float], float, bool] — (obs, reward, done)
    ///     obs    — next observation, length OBS_DIM
    ///     reward — scalar reward for this tick
    ///     done   — True when the episode is complete
    pub fn step(&mut self, action: u32) -> (Vec<f32>, f32, bool) {
        self.inner.step(action)
    }

    /// Number of floats in the observation vector. Use to size the input layer.
    #[staticmethod]
    pub fn obs_dim() -> usize {
        OBS_DIM
    }

    /// Number of discrete actions. Use to size the output layer / action space.
    #[staticmethod]
    pub fn action_size() -> usize {
        ACTION_SIZE
    }
}

// ── Module registration ───────────────────────────────────────────────────────

/// Register the `atb` Python module.
/// Called by PyO3's cdylib entry point — must match `[lib] name` in Cargo.toml.
#[pymodule]
pub fn atb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRlEnv>()?;
    Ok(())
}