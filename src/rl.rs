// src/rl.rs
//
// RL module root.
//
// `marker` and `action` are always compiled — they are pure Rust with no
// PyO3 / numpy dependency and form part of the simulation's contract
// ("this entity's actions come from outside the brain system"). Production
// systems like `agent::systems::tick_agents` filter on `RlAgent` regardless
// of whether a Python policy is actually attached.
//
// `env`, `obs`, `reward`, and `pyo3` are gated behind the `python` feature
// because they pull in PyO3 and the headless RL training loop.

pub mod action;
pub mod marker;

#[cfg(feature = "python")]
pub mod env;
#[cfg(feature = "python")]
pub mod obs;        // CNN-shaped (C, H, W) observations
#[cfg(feature = "python")]
pub mod pyo3;
#[cfg(feature = "python")]
pub mod reward;     // gold-task reward

pub use action::{action_to_int, int_to_action, ACTION_SIZE};
pub use marker::RlAgent;

#[cfg(feature = "python")]
pub use env::RlEnv;
#[cfg(feature = "python")]
pub use obs::{build_obs_grid, OBS_SHAPE, OBS_TOTAL};
#[cfg(feature = "python")]
pub use reward::{compute_reward, PrevState};