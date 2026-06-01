use bevy::prelude::*;
use atb::world::config::SPAWN_POCKET_RADIUS;
use crate::sim_bridge::{SimBridge, AgentIndex, AgentMarker, ItemIndex, ItemMarker};
use crate::style::color::SPAWN_POCKET_FILL;
use crate::viz::grid_offset::GridOffset;

#[derive(Component)]
pub struct SpawnPocketOverlay;

/// Spawn the single spawn-pocket fill quad. Sized to the (2R+1)×(2R+1) pocket and
/// drawn just above the floor tiles (z=0.25) so items/agent render on top. Its
/// position is updated each frame by `sync_spawn_pocket_overlay`.
pub fn spawn_spawn_pocket_overlay(mut commands: Commands, offset: Res<GridOffset>) {
    let side = (2 * SPAWN_POCKET_RADIUS + 1) as f32 * offset.step;
    commands.spawn((
        Sprite {
            color: SPAWN_POCKET_FILL,
            custom_size: Some(Vec2::splat(side)),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.25),
        SpawnPocketOverlay,
    ));
}

/// Keep the spawn-pocket overlay centred on the base (the base is randomised each
/// episode, so this follows it on every reset).
pub fn sync_spawn_pocket_overlay(
    bridge:    Res<SimBridge>,
    offset:    Res<GridOffset>,
    mut query: Query<&mut Transform, With<SpawnPocketOverlay>>,
) {
    let Some(agent) = bridge.agents().first() else { return };
    let world = offset.world_pos(agent.base_pos.x, agent.base_pos.y);
    for mut transform in query.iter_mut() {
        transform.translation = Vec3::new(world.x, world.y, 0.25);
    }
}

pub fn sync_agent_transforms(
    bridge:    Res<SimBridge>,
    offset:    Res<GridOffset>,
    mut query: Query<(&AgentIndex, &mut Transform), With<AgentMarker>>,
) {
    let agents = bridge.agents();
    for (idx, mut transform) in query.iter_mut() {
        if let Some(agent) = agents.get(idx.0) {
            let world = offset.world_pos(agent.pos.x, agent.pos.y);
            transform.translation = Vec3::new(world.x, world.y, 1.0);
        }
    }
}

pub fn sync_item_transforms(
    bridge:    Res<SimBridge>,
    offset:    Res<GridOffset>,
    mut query: Query<(&ItemIndex, &mut Transform, &mut Visibility), With<ItemMarker>>,
) {
    let items = bridge.items();
    for (idx, mut transform, mut vis) in query.iter_mut() {
        if let Some(item) = items.get(idx.0) {
            let world = offset.world_pos(item.pos.x, item.pos.y);
            transform.translation = Vec3::new(world.x, world.y, 0.5);
            *vis = Visibility::Visible;
        } else {
            *vis = Visibility::Hidden;
        }
    }
}
