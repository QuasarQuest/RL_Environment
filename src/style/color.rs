// src/style/color

use bevy::prelude::Color;

// ── Grays (Dark theme) ────────────────────────────────────────────────────────
pub const GRAY_900: Color = Color::srgba(0.06, 0.06, 0.08, 0.97);
pub const GRAY_800: Color = Color::srgb(0.12, 0.12, 0.14);
pub const GRAY_400: Color = Color::srgb(0.55, 0.55, 0.60);
pub const GRAY_100: Color = Color::srgb(0.95, 0.95, 0.98);

// ── Semantic primitives ───────────────────────────────────────────────────────
pub const GREEN_500: Color = Color::srgb(0.12, 0.42, 0.24);
pub const GREEN_400: Color = Color::srgb(0.40, 0.90, 0.55);
pub const RED_500:   Color = Color::srgb(0.85, 0.25, 0.20);
pub const GOLD_500:  Color = Color::srgb(0.95, 0.78, 0.20);
pub const GOLD_800:  Color = Color::srgb(0.50, 0.35, 0.05);
pub const BLUE_500:  Color = Color::srgb(0.20, 0.50, 0.90);

// ── UI surface colors ─────────────────────────────────────────────────────────
pub const SURFACE_TOOLTIP:   Color = Color::srgba(0.06, 0.06, 0.08, 0.96);
pub const SURFACE_HIGHLIGHT: Color = Color::srgba(1.0,  1.0,  1.0,  0.03);
pub const BORDER:            Color = Color::srgba(1.0,  1.0,  1.0,  0.07);

// ── Team colors ───────────────────────────────────────────────────────────────
// Single source of truth for team → color mapping.
// Used by tile.rs (base tile rendering) and team/components.rs (agent color).

pub const TEAM_COLORS: &[Color] = &[
    RED_500,   // team 0 — Red
    BLUE_500,  // team 1 — Blue
    GREEN_400, // team 2 — Green
    GOLD_500,  // team 3 — Yellow
];

/// Returns the color for a given team id.
/// Falls back to GRAY_400 for unknown teams.
pub fn team_color(team_id: u8) -> Color {
    TEAM_COLORS.get(team_id as usize).copied().unwrap_or(GRAY_400)
}