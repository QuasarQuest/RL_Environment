use bevy::prelude::*;

mod sim_bridge;
mod sim_config;
mod style;
mod team;
mod viz;

use atb::config;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title:        config::WINDOW_TITLE.to_string(),
                resolution:   bevy::window::WindowResolution::new(config::WINDOW_WIDTH, config::WINDOW_HEIGHT),
                ..default()
            }),
            ..default()
        }))
        .configure_sets(Update, viz::SimSet)
        .add_plugins(sim_bridge::SimBridgePlugin)
        .add_plugins(viz::VizPlugin)
        .run();
}
