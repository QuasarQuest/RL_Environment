use bevy::prelude::*;

#[derive(Component)] pub struct TickLabelMarker;
#[derive(Component)] pub struct TimeLabelMarker;
#[derive(Component)] pub struct TeamScoreMarker(pub u8);

#[derive(Component)] pub struct SpeedDecreaseButton;
#[derive(Component)] pub struct SpeedIncreaseButton;
#[derive(Component)] pub struct SpeedResetButton;
#[derive(Component)] pub struct CurrentSpeedLabel;
#[derive(Component)] pub struct PauseButtonMarker;
#[derive(Component)] pub struct PauseButtonText;

#[derive(Component)] pub struct PolicyModeLabel;
