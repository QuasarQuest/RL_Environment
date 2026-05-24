use bevy::prelude::*;
use super::color::*;

#[derive(Clone, Copy, Debug)]
pub enum ThemeColor {
    Background,
    TooltipBackground,
    Border,
    TextPrimary,
    TextDim,
    ButtonIdle,
    SuccessText,
    AccentGold,
}

impl ThemeColor {
    pub fn resolve(self) -> Color {
        match self {
            Self::Background        => GRAY_900,
            Self::TooltipBackground => SURFACE_TOOLTIP,
            Self::Border            => BORDER,
            Self::TextPrimary       => GRAY_100,
            Self::TextDim           => GRAY_400,
            Self::ButtonIdle        => GRAY_800,
            Self::SuccessText       => GREEN_400,
            Self::AccentGold        => GOLD_500,
        }
    }
}

#[derive(Component)]
pub struct UiRoot;

pub const SIZE_SM: f32 = 11.0;
pub const SIZE_MD: f32 = 13.0;
pub const SIZE_LG: f32 = 15.0;
pub const SIZE_XL: f32 = 16.0;
pub const TOOLBAR_H: f32 = 48.0;
pub const FONT_ICON: &str = "fonts/NotoSansSymbols2-Regular.ttf";
