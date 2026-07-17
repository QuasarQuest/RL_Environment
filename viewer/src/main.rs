// Bevy queries with multiple filters trip clippy's type_complexity lint by
// design; aliasing every query type would hurt readability more than it helps.
#![allow(clippy::type_complexity)]

use bevy::prelude::*;

mod policy;
mod sim_bridge;
mod sim_config;
mod style;
mod team;
mod viz;

use atb::config;

const ASSET_PATH:       &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets");
const ONNX_POLICY_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/model/policy.onnx");

fn main() {
    App::new()
        .insert_resource(ClearColor(style::color::APP_BACKGROUND))
        .add_plugins(DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title:        config::WINDOW_TITLE.to_string(),
                    resolution:   bevy::window::WindowResolution::new(config::WINDOW_WIDTH, config::WINDOW_HEIGHT),
                    ..default()
                }),
                ..default()
            })
            .set(bevy::asset::AssetPlugin {
                file_path: ASSET_PATH.to_string(),
                ..default()
            })
            // sctk-adwaita (Wayland client-side decorations) spams a WARN for every
            // title-bar button token it can't parse from the GNOME button-layout
            // gsetting. Harmless and out of our control, so drop it to error level
            // while keeping Bevy's other defaults.
            .set(bevy::log::LogPlugin {
                filter: format!("{},sctk_adwaita=error", bevy::log::DEFAULT_FILTER),
                ..default()
            })
        )
        .add_plugins(sim_bridge::SimBridgePlugin)
        .add_plugins(viz::VizPlugin)
        .run();
}
