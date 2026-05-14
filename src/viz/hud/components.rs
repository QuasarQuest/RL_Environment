// src/viz/hud/components.rs

use bevy::prelude::*;

#[derive(Component)] pub struct TickLabelMarker;
#[derive(Component)] pub struct TimeLabelMarker;
#[derive(Component)] pub struct TeamScoreMarker(pub u8);