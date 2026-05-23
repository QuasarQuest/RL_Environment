use bevy::prelude::*;
use crate::sim_bridge::{SimBridge, AgentIndex, AgentMarker, ItemIndex, ItemMarker};
use crate::viz::grid_offset::GridOffset;

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
