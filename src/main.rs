// src/main.rs

mod agent;
mod algorithm;
mod config;
mod factory;
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
        .add_plugins((WorldPlugin, SimPlugin, ItemPlugin, AgentPlugin, VizPlugin))
        .run();
}

// TODO: Refactor factory/mod.rs — validate agent entity ordering matches MapConfig spawn order

// TODO: Visualize algorithm state
//       - Path planning: debug overlay restored (was broken by Phase 2 HideViz move)
//       - Behavior planner: show active BT branch / FSM state / GOAP plan step above agent sprite

// TODO: Visualize combat — animate attacks and show ammo + hearts next to each agent sprite

// TODO: Fix scoreboard — add visual divider between Red and Blue team sections

// TODO: Track kills and deaths per agent
//       - Add KillCount(u32) and DeathCount(u32) sim components
//       - Increment in combat.rs on kill / respawn
//       - Display K/D columns in scoreboard

// TODO: Award kill reward points to attacker on kill
//       - Configurable via KILL_REWARD in config.rs
//       - Incentivises combat for RL agent over pure gold hunting

// TODO: Match end screen — trigger when tick >= match_duration_ticks
//       - Show final scores, winner, K/D summary
//       - Restart button reloads MapConfig and respawns all agents