use bevy::prelude::*;
use bevy::color::Alpha;
use atb::{config, world::tile::Tile};
use crate::sim_bridge::SimBridge;
use crate::viz::grid_offset::GridOffset;
use crate::style::color::{team_color, TILE_FREE, TILE_OBSTACLE};

#[derive(Component)]
pub struct TileMarker {
    pub x: usize,
    pub y: usize,
}

fn tile_color(tile: Tile) -> Color {
    match tile {
        Tile::Free           => TILE_FREE,
        Tile::Obstacle       => TILE_OBSTACLE,
        Tile::Base(team)     => team_color(team).with_alpha(0.80),
        Tile::SafeZone(team) => team_color(team).with_alpha(0.28),
    }
}

/// Spawn tile sprites for the current bridge grid.
/// Extracted so it can be called both at startup and on hot-reload.
pub fn do_spawn_tiles(commands: &mut Commands, bridge: &SimBridge, offset: &GridOffset) {
    for (x, y, tile) in bridge.grid().iter() {
        let pos = offset.world_pos(x as i32, y as i32);
        commands.spawn((
            Sprite {
                color:       tile_color(tile),
                custom_size: Some(Vec2::splat(config::TILE_SIZE)),
                ..default()
            },
            Transform::from_xyz(pos.x, pos.y, 0.0),
            TileMarker { x, y },
        ));
    }
}

pub fn spawn_tiles(
    mut commands: Commands,
    bridge:       Res<SimBridge>,
    offset:       Res<GridOffset>,
) {
    do_spawn_tiles(&mut commands, &bridge, &offset);
}

/// Repaint tile sprites from the grid — only when the map actually changed
/// (restart/load). The grid is immutable mid-episode, so repainting all w×h
/// sprites on every stepped frame was pure waste.
pub fn sync_tile_colors(
    bridge:    Res<SimBridge>,
    mut ev:    MessageReader<crate::viz::events::MapChanged>,
    mut query: Query<(&TileMarker, &mut Sprite)>,
) {
    if ev.read().next().is_none() { return; }
    for (marker, mut sprite) in query.iter_mut() {
        if let Some(tile) = bridge.grid().get(marker.x as i32, marker.y as i32) {
            sprite.color = tile_color(tile);
        }
    }
}
