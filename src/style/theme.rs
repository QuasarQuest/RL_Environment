// src/style/theme.rs

use bevy::prelude::*;
use super::color::*;

#[derive(Clone, Copy, Debug)]
pub enum ThemeColor {
    Background,
    TooltipBackground,
    Border,
    SurfaceHighlight,
    TextPrimary,
    TextDim,
    ButtonIdle,
    Success,
    SuccessText,
    AccentGold,
}

impl ThemeColor {
    pub fn resolve(self) -> Color {
        match self {
            Self::Background        => GRAY_900,
            Self::TooltipBackground => Color::srgba(0.06, 0.06, 0.08, 0.96),
            Self::SurfaceHighlight  => Color::srgba(1.0, 1.0, 1.0, 0.03),
            Self::Border            => Color::srgba(1.0, 1.0, 1.0, 0.07),
            Self::TextPrimary       => GRAY_100,
            Self::TextDim           => GRAY_400,
            Self::ButtonIdle        => GRAY_800,
            Self::Success           => GREEN_500,
            Self::SuccessText       => GREEN_400,
            Self::AccentGold        => GOLD_500,
        }
    }
}

#[derive(Component)]
pub struct UiRoot;

// Standard text sizes
pub const SIZE_SM: f32 = 11.0;
pub const SIZE_MD: f32 = 13.0;
pub const SIZE_LG: f32 = 15.0;
pub const SIZE_XL: f32 = 16.0;

// Standard UI dimensions
pub const TOOLBAR_H: f32 = 48.0;