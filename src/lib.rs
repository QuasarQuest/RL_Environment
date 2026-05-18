// src/lib.rs

pub mod agent;
pub mod algorithm;
pub mod config;
pub mod factory;
pub mod item;
pub mod sim;
pub mod style;
pub mod team;
pub mod world;

#[cfg(feature = "python")]
pub mod rl;

#[cfg(not(feature = "headless"))]
pub mod viz;

// PyO3 cdylib entry point — only present when building for Python.
#[cfg(feature = "python")]
pub use rl::pyo3::atb;