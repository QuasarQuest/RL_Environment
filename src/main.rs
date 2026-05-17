// src/main.rs

mod agent;
mod algorithm;
mod config;
mod factory;
mod item;
mod sim;
mod style;
mod team;
mod world;

#[cfg(not(feature = "headless"))]
mod viz;

use bevy::prelude::*;
use sim::SimPlugin;
use world::WorldPlugin;
use agent::AgentPlugin;
use item::ItemPlugin;

#[cfg(not(feature = "headless"))]
use viz::VizPlugin;

fn main() {
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

    app.add_plugins((WorldPlugin, SimPlugin, ItemPlugin, AgentPlugin));

    #[cfg(not(feature = "headless"))]
    app.add_plugins(VizPlugin);

    app.run();
}

//TODO: Define RL API Boundaries.
//TODO: Respawn not in base after dead.
//TODO: Async Timer sim problem with RL.
//TODO: Blue team always wins.
//TODO: Save Stats after ML run