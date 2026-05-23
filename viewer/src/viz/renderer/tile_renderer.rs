use bevy::prelude::*;
use atb::{config, world::tile::Tile};
use crate::sim_bridge::SimBridge;
use crate::viz::grid_offset::GridOffset;
use crate::style::color::team_color;

#[derive(Component)]
pub struct TileMarker {
    pub x: usize,
    pub y: usize,
}

fn tile_color(tile: Tile) -> Color {
    match tile {
        Tile::Free           => Color::srgb(0.10, 0.10, 0.12),
        Tile::Obstacle       => Color::srgb(0.28, 0.28, 0.32),
        Tile::Base(team)     => {
            let c = team_color(team);
            Color::srgba(c.to_srgba().red, c.to_srgba().green, c.to_srgba().blue, 0.70)
        }
        Tile::SafeZone(team) => {
            let c = team_color(team);
            Color::srgba(c.to_srgba().red, c.to_srgba().green, c.to_srgba().blue, 0.18)
        }
    }
}

pub fn spawn_tiles(
    mut commands: Commands,
    bridge:       Res<SimBridge>,
    offset:       Res<GridOffset>,
) {
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

pub fn sync_tile_colors(
    bridge:    Res<SimBridge>,
    mut query: Query<(&TileMarker, &mut Sprite)>,
) {
    if !bridge.is_changed() { return; }
    for (marker, mut sprite) in query.iter_mut() {
        if let Some(tile) = bridge.grid().get(marker.x as i32, marker.y as i32) {
            sprite.color = tile_color(tile);
        }
    }
}
