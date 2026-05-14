// src/viz/plugin.rs

use bevy::prelude::*;

use super::camera::{spawn_camera, fit_camera_to_grid, init_pan_state, camera_controls};
use super::grid_offset::compute_grid_offset;
use super::tile_renderer::{spawn_tiles, sync_tile_colors};
use super::agent_renderer::sync_agent_transforms;
use super::algorithm::draw_agent_debug;
use super::tooltip::{spawn_tooltip, update_tooltip};
use crate::style::ThemeMode;

use super::menu::components::{DebugVizConfig, MenuState};
use super::menu::{
    react_to_ui_changes,
    handle_hamburger_button,
    handle_drawer_overlay,
    handle_theme_toggle_button,
    handle_pause_button,
    handle_speed_buttons,
    update_button_styles,
    update_speed_label,
};

use super::hud::{
    spawn_hud, spawn_scoreboard, update_scoreboard, update_tick_label,
};

pub struct VizPlugin;

impl Plugin for VizPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<DebugVizConfig>()
            .init_resource::<ThemeMode>()
            .init_resource::<MenuState>()
            .add_systems(PreStartup, init_pan_state)
            .add_systems(Startup, (
                spawn_camera,
                compute_grid_offset,
                spawn_tiles,
                fit_camera_to_grid,
            ).chain())
            .add_systems(Startup, (
                spawn_hud,
                spawn_scoreboard,
                spawn_tooltip,
            ).chain())
            .add_systems(Update, (
                camera_controls,
                sync_tile_colors,
                sync_agent_transforms,
                draw_agent_debug,

                react_to_ui_changes,

                handle_hamburger_button,
                handle_drawer_overlay,
                handle_theme_toggle_button,
                handle_pause_button,
                handle_speed_buttons,
                update_button_styles,
                update_speed_label,

                update_tick_label,
                update_scoreboard,
                update_tooltip,
            ));
    }
}