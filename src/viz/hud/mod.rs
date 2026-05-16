// src/viz/hud/mod.rs

pub mod components;
pub mod layout;
pub mod scoreboard;
pub mod systems;

pub use layout::spawn_hud;
pub use scoreboard::{spawn_tab_scoreboard, toggle_tab_scoreboard, update_tab_scoreboard};
pub use systems::{
    update_tick_label, update_time_label, update_team_scores,
    handle_speed_buttons, handle_pause_button,
};