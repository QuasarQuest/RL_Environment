// src/viz/plugin.rs
use bevy::prelude::*;

use super::camera::{spawn_camera, fit_camera_to_grid, init_pan_state, camera_controls};
use super::grid_offset::compute_grid_offset;
use super::renderer::tile_renderer::{spawn_tiles, sync_tile_colors};
use super::renderer::agent_renderer::sync_agent_transforms;
use super::world::{draw_agent_stats, draw_agent_range, draw_agent_path, draw_safe_zone_borders};
use super::panels::tooltip::{spawn_tooltip, update_tooltip};
use super::panels::help_overlay::{spawn_help_overlay, toggle_help_overlay};
use super::panels::end_screen::{
    spawn_end_screen, show_end_screen, populate_end_screen_cards,
    handle_quit_button, handle_restart_button,
};
use super::panels::scoreboard::{
    spawn_tab_scoreboard, toggle_tab_scoreboard,
    build_scoreboard_rows, refresh_scoreboard_stats,
    refresh_scoreboard_viz, refresh_scoreboard_avg,
    handle_viz_toggle,
};
use super::events::RestartEvent;
use super::events::restart::restart_exclusive;
use crate::factory::assign_display_components;
use crate::sim::plugin::{SimSystems, fire_sim_tick};
use super::hud::{
    spawn_hud,
    update_tick_label, update_time_label, update_team_scores,
    handle_speed_buttons, handle_pause_button, sync_pause_visuals,
};

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
struct HudUpdate;

pub struct VizPlugin;

impl Plugin for VizPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_message::<RestartEvent>()
            .add_systems(PreStartup, init_pan_state)
            .add_systems(Startup, (
                spawn_camera,
                compute_grid_offset,
                spawn_tiles,
                fit_camera_to_grid,
            ).chain())
            .add_systems(Startup, (
                spawn_hud,
                spawn_tab_scoreboard,
                spawn_tooltip,
                spawn_help_overlay,
                spawn_end_screen,
                assign_display_components,
            ).chain())
            .configure_sets(Update,
                            HudUpdate
                                .after(SimSystems)
                                .after(fire_sim_tick)
            )
            .add_systems(Update, (
                camera_controls,
                sync_tile_colors,
                sync_agent_transforms,
                draw_agent_stats,
                draw_agent_range,
                draw_agent_path,
                draw_safe_zone_borders,
            ))
            .add_systems(Update, (
                update_tooltip,
                toggle_tab_scoreboard,
                build_scoreboard_rows,
                refresh_scoreboard_stats,
                refresh_scoreboard_viz,
            ))
            .add_systems(Update, (
                refresh_scoreboard_avg,
                handle_viz_toggle,
            ))
            .add_systems(Update, (
                handle_speed_buttons,
                handle_pause_button,
                sync_pause_visuals,
                toggle_help_overlay,
            ))
            .add_systems(Update, (
                show_end_screen,
                populate_end_screen_cards,
            ).in_set(HudUpdate))
            .add_systems(Update, (
                handle_quit_button,
                handle_restart_button,
            ).in_set(HudUpdate))
            .add_systems(Update, (
                update_tick_label,
                update_time_label,
                update_team_scores,
            ).in_set(HudUpdate))
            .add_systems(Update, assign_display_components)
            .add_systems(Update, restart_exclusive.after(HudUpdate));
    }
}