// src/main.rs

mod agent;
mod algorithm;
mod config;
mod item;
mod sim;
mod style;
mod team;
mod viz;
mod world;

use bevy::prelude::*;
use sim::SimPlugin;
use viz::VizPlugin;
use world::WorldPlugin;
use agent::AgentPlugin;
use item::ItemPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title:      config::WINDOW_TITLE.into(),
                resolution: bevy::window::WindowResolution::new(
                    config::WINDOW_WIDTH,
                    config::WINDOW_HEIGHT,
                ),
                ..default()
            }),
            ..default()
        }))
        // Order matters: World loads map config, Item reads it, then agents spawn.
        .add_plugins((WorldPlugin, SimPlugin, ItemPlugin, VizPlugin, AgentPlugin))
        .run();
}