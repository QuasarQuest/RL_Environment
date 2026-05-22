// src/lib.rs
//
// Library crate `atb`.
//
// Module tree for the pure-Rust simulation core and algorithm library.
// The Python cdylib entry point (`atb` module) is exposed via rl::pyo3
// when building with --features python.
//
// Bevy is no longer a dependency. Visualisation is handled by a separate
// Python pygame viewer that reads SimCore state.

pub mod agent;
pub mod algorithm;
pub mod config;
pub mod item;
pub mod world;
pub mod rl;

#[cfg(feature = "python")]
pub mod sim_core;

// PyO3 cdylib entry point.
#[cfg(feature = "python")]
pub use rl::pyo3::atb;
