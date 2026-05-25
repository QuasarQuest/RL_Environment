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
        )
        .configure_sets(Update, viz::SimSet)
        .add_plugins(sim_bridge::SimBridgePlugin)
        .add_plugins(viz::VizPlugin)
        .run();
}
