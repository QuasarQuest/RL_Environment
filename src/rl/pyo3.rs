// src/rl/pyo3.rs
//
// PyO3 bindings — exposes RlEnv as `PyRlEnv` to Python.
//
// Python usage:
//
//   import atb
//   env = atb.PyRlEnv("assets/world/config.ron")
//
//   c, h, w = atb.PyRlEnv.obs_shape()       # (5, 25, 25) for the CNN obs
//   obs_flat = env.reset()                  # list[float], len == c*h*w
//   obs      = np.asarray(obs_flat).reshape(c, h, w)
//
//   obs, reward, done = env.step(4)         # action int 0..ACTION_SIZE
//
// The env returns a flat list rather than a 3D structure because the PyO3
// boundary is much cleaner for Vec<f32>. The Python side reshapes it.
// CHW ordering (channel-major) matches PyTorch's default tensor layout.
//
// Thread safety:
//   PyRlEnv is not Send (Bevy App contains raw pointers). It must stay on
//   the thread that created it. For parallel training, spawn separate
//   Python processes each owning one PyRlEnv — do not share across threads.

use pyo3::prelude::*;
use super::env::{RlEnv, ACTION_SIZE, OBS_DIM, OBS_SHAPE};

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
    ///     Resolved by walking up from CWD, so a relative path like
    ///     "assets/world/config.ron" works from any project subdirectory.
    #[new]
    pub fn new(config_path: String) -> Self {
        Self { inner: RlEnv::new(config_path) }
    }

    /// Reset to a fresh episode. Returns the initial observation as a flat
    /// list of floats. Caller should reshape to obs_shape().
    pub fn reset(&mut self) -> Vec<f32> {
        self.inner.reset()
    }

    /// Step the simulation by one tick.
    ///
    /// Args:
    ///   action (int): discrete action index, 0 ≤ action < ACTION_SIZE
    ///
    /// Returns:
    ///   tuple[list[float], float, bool] — (obs_flat, reward, done)
    pub fn step(&mut self, action: u32) -> (Vec<f32>, f32, bool) {
        self.inner.step(action)
    }

    /// Total floats in one observation (== C * H * W).
    #[staticmethod]
    pub fn obs_dim() -> usize { OBS_DIM }

    /// Observation tensor shape as (channels, height, width). Use this to
    /// reshape the flat list returned by reset()/step() into a 3D array.
    #[staticmethod]
    pub fn obs_shape() -> (usize, usize, usize) { OBS_SHAPE }

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