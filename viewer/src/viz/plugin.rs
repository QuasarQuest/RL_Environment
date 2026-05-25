use bevy::prelude::*;

use super::camera::{spawn_camera, fit_camera_to_grid, init_pan_state, camera_controls};
use super::grid_offset::compute_grid_offset;
use super::renderer::tile_renderer::{spawn_tiles, sync_tile_colors};
use super::renderer::agent_renderer::{sync_agent_transforms, sync_item_transforms};
use super::renderer::debug_gizmos::{
    DebugGizmoFlags, toggle_path_viz, toggle_range_viz,
    draw_path_gizmo, draw_range_gizmos,
};
use super::hud::{
    spawn_hud,
    update_tick_label, update_time_label, update_team_scores,
    handle_speed_buttons, handle_pause_button, sync_pause_visuals,
    handle_policy_key, sync_policy_mode_label,
};
use super::panels::scoreboard::{
    spawn_tab_scoreboard, toggle_tab_scoreboard,
    build_scoreboard_rows, refresh_scoreboard_stats,
};
use super::panels::debug_overlay::{spawn_debug_overlay, toggle_debug_overlay, update_debug_overlay};
use super::panels::help_overlay::{spawn_help_overlay, toggle_help_overlay};
use super::panels::end_screen::{
    spawn_end_screen, show_end_screen,
    handle_quit_button, handle_restart_button,
};
use super::panels::tooltip::{spawn_tooltip, update_tooltip};
use super::SimSet;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
struct HudUpdate;

pub struct VizPlugin;

impl Plugin for VizPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<DebugGizmoFlags>()
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
                spawn_debug_overlay,
                spawn_end_screen,
            ))
            .configure_sets(Update, HudUpdate.after(SimSet))
            .add_systems(Update, (
                camera_controls,
                sync_tile_colors,
                sync_agent_transforms,
                sync_item_transforms,
            ))
            .add_systems(Update, (
                update_tooltip,
                toggle_tab_scoreboard,
                build_scoreboard_rows,
                refresh_scoreboard_stats,
            ))
            .add_systems(Update, (
                handle_speed_buttons,
                handle_pause_button,
                sync_pause_visuals,
                handle_policy_key,
                sync_policy_mode_label,
                toggle_help_overlay,
                toggle_debug_overlay,
                update_debug_overlay,
                toggle_path_viz,
                toggle_range_viz,
                draw_path_gizmo,
                draw_range_gizmos,
            ))
            .add_systems(Update, (
                show_end_screen,
                handle_quit_button,
                handle_restart_button,
            ).in_set(HudUpdate))
            .add_systems(Update, (
                update_tick_label,
                update_time_label,
                update_team_scores,
            ).in_set(HudUpdate));
    }
}
