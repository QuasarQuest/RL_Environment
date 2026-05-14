// src/viz/core_ui/theme.rs

use bevy::prelude::*;
use super::color::*;

#[derive(Resource, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThemeMode {
    #[default]
    Dark,
    Light,
}

#[derive(Clone, Copy, Debug)]
pub enum ThemeColor {
    Background,
    TooltipBackground,
    Border,
    SurfaceHighlight,
    TextPrimary,
    TextDim,
    ButtonIdle,
    ButtonHover,
    Success,
    SuccessText,
    Warning,
    WarningText,
    Error,
    ErrorText,
    AccentGold,
}

impl ThemeColor {
    pub fn resolve(&self, mode: ThemeMode) -> Color {
        match (self, mode) {
            (Self::Background,        ThemeMode::Dark) => GRAY_900,
            (Self::TooltipBackground, ThemeMode::Dark) => Color::srgba(0.06, 0.06, 0.08, 0.96),
            (Self::SurfaceHighlight,  ThemeMode::Dark) => Color::srgba(1.0, 1.0, 1.0, 0.03),
            (Self::Border,            ThemeMode::Dark) => Color::srgba(1.0, 1.0, 1.0, 0.07),
            (Self::TextPrimary,       ThemeMode::Dark) => GRAY_100,
            (Self::TextDim,           ThemeMode::Dark) => GRAY_400,
            (Self::ButtonIdle,        ThemeMode::Dark) => GRAY_800,
            (Self::ButtonHover,       ThemeMode::Dark) => GRAY_700,

            (Self::Background,        ThemeMode::Light) => WHITE_900,
            (Self::TooltipBackground, ThemeMode::Light) => Color::srgba(0.98, 0.98, 0.99, 0.96),
            (Self::SurfaceHighlight,  ThemeMode::Light) => Color::srgba(0.0, 0.0, 0.0, 0.03),
            (Self::Border,            ThemeMode::Light) => Color::srgba(0.0, 0.0, 0.0, 0.07),
            (Self::TextPrimary,       ThemeMode::Light) => DARK_TEXT,
            (Self::TextDim,           ThemeMode::Light) => GRAY_400,
            (Self::ButtonIdle,        ThemeMode::Light) => WHITE_800,
            (Self::ButtonHover,       ThemeMode::Light) => WHITE_700,

            (Self::Success,      _) => GREEN_500,
            (Self::SuccessText,  _) => GREEN_400,
            (Self::Warning,      _) => GOLD_500,
            (Self::WarningText,  _) => GRAY_900,
            (Self::Error,        _) => RED_500,
            (Self::ErrorText,    _) => RED_400,
            (Self::AccentGold,   _) => GOLD_500,
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

// Standard UI Dimensions
pub const TOOLBAR_H: f32 = 48.0;
pub const DRAWER_W: f32 = 240.0;