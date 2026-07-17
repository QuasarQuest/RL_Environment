pub mod restart;
pub use restart::RestartPending;

use bevy::prelude::*;

/// The sim's map was regenerated (episode restart or config load) — tile sprites
/// must repaint. Lets `sync_tile_colors` run only then instead of every tick.
#[derive(Message)]
pub struct MapChanged;

/// A new world config was loaded — grid dimensions and agent composition may
/// differ, so the camera refits and the scoreboard rows rebuild.
#[derive(Message)]
pub struct ConfigLoaded;
