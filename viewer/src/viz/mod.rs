pub mod camera;
pub mod core_ui;
pub mod events;
pub mod grid_offset;
pub mod hud;
pub mod panels;
pub mod plugin;
pub mod renderer;
pub mod world;

pub use plugin::VizPlugin;

use bevy::prelude::*;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SimSet;
