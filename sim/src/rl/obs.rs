// src/rl/obs.rs
//
// Observation constants for the CNN-shaped RL observation.
//
// Layout: flat Vec<f32> of length OBS_TOTAL, CHW order.
// Channels:
//   0  Out-of-bounds mask
//   1  Own base
//   2  Gold item
//   3  Self at centre
//   4  Carrying gold (broadcast plane)

pub const OBS_CHANNELS:  usize = 5;
pub const OBS_CROP_SIZE: usize = 25;
pub const OBS_TOTAL:     usize = OBS_CHANNELS * OBS_CROP_SIZE * OBS_CROP_SIZE;

/// Tuple form for Python — (C, H, W).
pub const OBS_SHAPE: (usize, usize, usize) = (OBS_CHANNELS, OBS_CROP_SIZE, OBS_CROP_SIZE);

/// Total floats in one observation (alias for clarity).
pub const OBS_DIM: usize = OBS_TOTAL;
