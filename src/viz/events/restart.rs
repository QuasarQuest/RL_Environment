// src/viz/events/restart.rs
use bevy::prelude::*;
use crate::sim::reset::reset_episode;
use crate::viz::panels::end_screen::EndScreen;

#[derive(Message, Clone)]
pub struct RestartEvent;

pub fn restart_exclusive(world: &mut World) {
    let has_event = {
        let mut messages = world.resource_mut::<Messages<RestartEvent>>();
        let drained: Vec<_> = messages.drain().collect();
        !drained.is_empty()
    };
    if !has_event { return; }

    info!("=== Episode restart ===");

    reset_episode(world);

    let end_screens: Vec<Entity> = world
        .query_filtered::<Entity, With<EndScreen>>()
        .iter(world)
        .collect();
    for e in end_screens {
        if let Some(mut node) = world.get_mut::<Node>(e) {
            node.display = Display::None;
        }
        if let Some(mut vis) = world.get_mut::<Visibility>(e) {
            *vis = Visibility::Hidden;
        }
    }

    info!("Episode restarted — tick 0.");
}