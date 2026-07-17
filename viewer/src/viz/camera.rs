use bevy::prelude::*;
use bevy::input::mouse::{AccumulatedMouseScroll, MouseScrollUnit};
use crate::sim_bridge::SimBridge;
use atb::config;

#[derive(Component)]
pub struct MainCamera;

#[derive(Resource, Default)]
pub struct PanState {
    dragging: bool,
    last_pos: Vec2,
}

pub fn init_pan_state(mut commands: Commands) {
    commands.insert_resource(PanState::default());
}

pub fn spawn_camera(mut commands: Commands) {
    commands.spawn((Camera2d, MainCamera));
}

pub fn fit_camera_to_grid(
    bridge:  Res<SimBridge>,
    windows: Query<&Window>,
    mut cam: Single<&mut Transform, With<MainCamera>>,
    mut proj: Single<&mut Projection, With<MainCamera>>,
) {
    let Ok(window) = windows.single() else { return };
    let grid  = bridge.grid();
    let step  = config::TILE_SIZE + config::TILE_GAP;
    let scale_x = grid.width  as f32 * step / window.width()  * config::CAMERA_FIT_MARGIN;
    let scale_y = grid.height as f32 * step / window.height() * config::CAMERA_FIT_MARGIN;

    if let Projection::Orthographic(ref mut ortho) = **proj {
        ortho.scale = scale_x.max(scale_y).clamp(config::ZOOM_MIN, config::ZOOM_MAX);
    }
    cam.translation = Vec3::ZERO;
}

/// Refit after a config load — the new map's dimensions may not fit the old zoom.
/// Restart keeps the user's zoom/pan (same dimensions), so only loads refit.
pub fn refit_camera_on_load(
    mut ev:  MessageReader<crate::viz::events::ConfigLoaded>,
    bridge:  Res<SimBridge>,
    windows: Query<&Window>,
    cam:     Single<&mut Transform, With<MainCamera>>,
    proj:    Single<&mut Projection, With<MainCamera>>,
) {
    if ev.read().next().is_none() { return; }
    fit_camera_to_grid(bridge, windows, cam, proj);
}

pub fn camera_controls(
    windows:  Query<&Window>,
    mut cam:  Single<(&mut Transform, &mut Projection), With<MainCamera>>,
    scroll:   Res<AccumulatedMouseScroll>,
    mouse:    Res<ButtonInput<MouseButton>>,
    mut pan:  ResMut<PanState>,
) {
    let Ok(window) = windows.single() else { return };
    let (ref mut tf, ref mut projection) = *cam;
    let Projection::Orthographic(ref mut ortho) = **projection else { return };

    let raw = match scroll.unit {
        MouseScrollUnit::Line  => scroll.delta.y,
        MouseScrollUnit::Pixel => scroll.delta.y / 20.0,
    };
    let clamped = raw.clamp(-1.0, 1.0);

    if clamped.abs() > f32::EPSILON {
        let old_scale = ortho.scale;
        let factor    = if clamped > 0.0 { 1.0 - config::ZOOM_SPEED } else { 1.0 + config::ZOOM_SPEED };
        let new_scale = (old_scale * factor).clamp(config::ZOOM_MIN, config::ZOOM_MAX);

        if let Some(cursor) = window.cursor_position() {
            let win   = Vec2::new(window.width(), window.height());
            let ndc   = (cursor / win - 0.5) * 2.0;
            let world = tf.translation.truncate()
                + Vec2::new(ndc.x * win.x / 2.0 * old_scale, -ndc.y * win.y / 2.0 * old_scale);
            let new_cam = world
                - Vec2::new(ndc.x * win.x / 2.0 * new_scale, -ndc.y * win.y / 2.0 * new_scale);
            tf.translation.x = new_cam.x;
            tf.translation.y = new_cam.y;
        }
        ortho.scale = new_scale;
    }

    if mouse.just_pressed(MouseButton::Middle) {
        pan.dragging = true;
        pan.last_pos = window.cursor_position().unwrap_or(Vec2::ZERO);
    }
    if mouse.just_released(MouseButton::Middle) { pan.dragging = false; }
    if pan.dragging {
        if let Some(cursor) = window.cursor_position() {
            let delta = cursor - pan.last_pos;
            tf.translation.x -= delta.x * ortho.scale;
            tf.translation.y += delta.y * ortho.scale;
            pan.last_pos = cursor;
        }
    }
}
