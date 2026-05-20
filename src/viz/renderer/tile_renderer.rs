// src/viz/tile_renderer.rs

use bevy::prelude::*;
use crate::world::Grid;
use crate::config;
use crate::viz::grid_offset::GridOffset;

#[derive(Component)]
pub struct TileMarker {
    pub x: usize,
    pub y: usize,
}

pub fn spawn_tiles(
    mut commands: Commands,
    grid:         Res<Grid>,
    offset:       Res<GridOffset>,
) {
    for (x, y, tile) in grid.iter() {
        let pos = offset.world_pos(x as i32, y as i32);
        commands.spawn((
            Sprite {
                color:       tile.color(),
                custom_size: Some(Vec2::splat(config::TILE_SIZE)),
                ..default()
            },
            Transform::from_xyz(pos.x, pos.y, 0.0),
            Visibility::default(),
            TileMarker { x, y },
        ));
    }
}

pub fn sync_tile_colors(
    grid:      Res<Grid>,
    mut query: Query<(&TileMarker, &mut Sprite)>,
) {
    if !grid.is_changed() { return; }
    for (marker, mut sprite) in query.iter_mut() {
        if let Some(tile) = grid.get(marker.x as i32, marker.y as i32) {
            sprite.color = tile.color();
        }
    }
}
