use bevy::prelude::*;
use crate::sim_bridge::SimBridge;
use atb::config;

#[derive(Resource, Clone, Copy)]
pub struct GridOffset {
    pub x:    f32,
    pub y:    f32,
    pub step: f32,
}

impl GridOffset {
    pub fn world_pos(&self, gx: i32, gy: i32) -> Vec2 {
        Vec2::new(
            self.x + gx as f32 * self.step,
            self.y + gy as f32 * self.step,
        )
    }
}

pub fn compute_grid_offset(mut commands: Commands, bridge: Res<SimBridge>) {
    let grid = bridge.grid();
    let step = config::TILE_SIZE + config::TILE_GAP;
    commands.insert_resource(GridOffset {
        x:    -(grid.width  as f32 * step) / 2.0 + step / 2.0,
        y:    -(grid.height as f32 * step) / 2.0 + step / 2.0,
        step,
    });
}
