// src/lib.rs
//
// Crate root for the `atb` cdylib — imported by Python as `import atb`.
// Exposes only the PyO3 module entry point; all sim logic lives in its
// own modules and is accessed through RlEnv.

pub mod agent;
pub mod algorithm;
pub mod config;
pub mod factory;
pub mod item;
pub mod rl;
pub mod sim;
pub mod style;
pub mod team;
pub mod viz;
pub mod world;

// PyO3 cdylib entry point — name must match [lib] name = "atb" in Cargo.toml.
pub use rl::pyo3::atb;