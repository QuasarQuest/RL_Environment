// src/viz/plugin.rs

use bevy::prelude::*;

use super::camera::{spawn_camera, fit_camera_to_grid, init_pan_state, camera_controls};
use super::grid_offset::compute_grid_offset;
use super::tile_renderer::{spawn_tiles, sync_tile_colors};
use super::agent_renderer::sync_agent_transforms;
use super::algorithm::draw_agent_debug;
use super::tooltip::{spawn_tooltip, update_tooltip};
use super::help_overlay::{spawn_help_overlay, toggle_help_overlay};
use crate::style::ThemeMode;
use crate::sim::plugin::{SimSystems, fire_sim_tick};

use super::hud::{
    spawn_hud,
    spawn_tab_scoreboard, toggle_tab_scoreboard, update_tab_scoreboard,
    update_tick_label, update_time_label, update_team_scores,
    handle_speed_buttons, handle_pause_button,
};
use super::hud::scoreboard::handle_viz_toggle;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
struct HudUpdate;

pub struct VizPlugin;

impl Plugin for VizPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<ThemeMode>()
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
                draw_agent_debug,
                update_tooltip,
                toggle_tab_scoreboard,
                update_tab_scoreboard,
                handle_viz_toggle,
                handle_speed_buttons,
                handle_pause_button,
                toggle_help_overlay,
            ))
            .add_systems(Update, (
                update_tick_label,
                update_time_label,
                update_team_scores,
            ).in_set(HudUpdate));
    }
}