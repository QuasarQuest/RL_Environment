pub mod components;
pub mod layout;
pub mod systems;

pub use layout::spawn_hud;
pub use systems::{
    update_tick_label, update_time_label,
    handle_speed_buttons, handle_pause_button, sync_pause_visuals,
    handle_policy_key, sync_policy_mode_label,
};
