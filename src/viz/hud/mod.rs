// src/viz/hud/mod.rs

pub mod components;
pub mod layout;
pub mod scoreboard;
pub mod systems;

pub use layout::spawn_hud;
pub use scoreboard::{spawn_scoreboard, update_scoreboard};
pub use systems::{update_tick_label, update_time_label, update_team_scores};