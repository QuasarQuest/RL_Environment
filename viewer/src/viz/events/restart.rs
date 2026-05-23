use bevy::prelude::*;

#[derive(Resource, Default)]
pub struct RestartPending(pub bool);
