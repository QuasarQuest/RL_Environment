// src/lib.rs
//
// Library crate `atb`. This is the single source of truth for the module
// tree — `main.rs` is a thin entry point that calls `run()` below, and the
// Python `cdylib` build exposes `rl::pyo3::atb` as the extension module.
//
// Do NOT redeclare these modules in `main.rs`; doing so compiles every file
// twice (once per crate) and produces two incompatible copies of every type.

pub mod agent;
pub mod algorithm;
pub mod config;
pub mod factory;
pub mod item;
pub mod sim;
pub mod style;
pub mod team;
pub mod world;

// The `rl` module file (src/rl.rs) is always compiled. Internal gating of
// the python-only submodules (env, obs, reward, pyo3) happens inside rl.rs
// itself, so that `marker` and `action` are available to production code
// (e.g. agent::systems filters on RlAgent) regardless of feature flags.
pub mod rl;

#[cfg(not(feature = "headless"))]
pub mod viz;

// PyO3 cdylib entry point — only present when building for Python.
#[cfg(feature = "python")]
pub use rl::pyo3::atb;

// ── App entry point ───────────────────────────────────────────────────────────
//
// Called by `main.rs`. Builds the Bevy App with the appropriate plugin set
// for the active feature flags and runs it.
//
//   default       → DefaultPlugins  + viz
//   headless      → MinimalPlugins, no viz
//   python        → implies headless; same as above
//
// The Python entry point doesn't use this — it constructs its own headless
// App inside `rl::env::build_headless_app`.

use bevy::prelude::*;

pub fn run() {
    let mut app = App::new();

    #[cfg(not(feature = "headless"))]
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title:      config::WINDOW_TITLE.into(),
            resolution: bevy::window::WindowResolution::new(
                config::WINDOW_WIDTH,
                config::WINDOW_HEIGHT,
            ),
            ..default()
        }),
        ..default()
    }));

    #[cfg(feature = "headless")]
    app.add_plugins(MinimalPlugins);

    app.add_plugins((
        world::WorldPlugin,
        sim::SimPlugin,
        item::ItemPlugin,
        agent::AgentPlugin,
    ));

    #[cfg(not(feature = "headless"))]
    app.add_plugins(viz::VizPlugin);

    app.run();
}