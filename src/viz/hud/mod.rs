// src/viz/hud/mod.rs

pub mod components;
pub mod layout;
pub mod scoreboard;
pub mod systems;

pub use layout::spawn_hud;
pub use scoreboard::{
    spawn_tab_scoreboard,
    toggle_tab_scoreboard,
    build_scoreboard_rows,
    refresh_scoreboard_stats,
    refresh_scoreboard_viz,
    refresh_scoreboard_avg,
};
pub use systems::{
    update_tick_label, update_time_label, update_team_scores,
    handle_speed_buttons, handle_pause_button, sync_pause_visuals,
};