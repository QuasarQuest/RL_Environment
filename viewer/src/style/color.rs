use bevy::prelude::Color;

pub const GRAY_900: Color = Color::srgba(0.06, 0.06, 0.08, 0.97);
pub const GRAY_800: Color = Color::srgb(0.12, 0.12, 0.14);
pub const GRAY_400: Color = Color::srgb(0.55, 0.55, 0.60);
pub const GRAY_100: Color = Color::srgb(0.95, 0.95, 0.98);

pub const GREEN_500:  Color = Color::srgb(0.12, 0.42, 0.24);
pub const GREEN_400:  Color = Color::srgb(0.40, 0.90, 0.55);
pub const RED_500:    Color = Color::srgb(0.85, 0.25, 0.20);
pub const ORANGE_400: Color = Color::srgb(0.95, 0.60, 0.10);
pub const GOLD_500:   Color = Color::srgb(0.95, 0.78, 0.20);
pub const GOLD_800:   Color = Color::srgb(0.50, 0.35, 0.05);
pub const BLUE_500:    Color = Color::srgb(0.20, 0.50, 0.90);
pub const CYAN_400:    Color = Color::srgb(0.20, 0.85, 0.90);
pub const LIME_400:    Color = Color::srgb(0.55, 0.90, 0.25);
pub const VIOLET_400:  Color = Color::srgb(0.65, 0.35, 0.95);
pub const AMBER_400:   Color = Color::srgb(0.98, 0.75, 0.15);

pub const SURFACE_TOOLTIP:   Color = Color::srgba(0.06, 0.06, 0.08, 0.96);
pub const SURFACE_HIGHLIGHT: Color = Color::srgba(1.0,  1.0,  1.0,  0.03);
pub const BORDER:            Color = Color::srgba(1.0,  1.0,  1.0,  0.07);

pub const TEAM_COLORS: &[Color] = &[
    RED_500,
    BLUE_500,
    GREEN_400,
    GOLD_500,
];

pub fn team_color(team_id: u8) -> Color {
    TEAM_COLORS.get(team_id as usize).copied().unwrap_or(GRAY_400)
}

// ── Item sprite colors ────────────────────────────────────────────────────────

pub const ITEM_GOLD:   Color = GOLD_500;
// Speed-boost tiers: three clearly distinct shades of green, dull → bright with
// rarity (tier 1 common/muted, tier 3 rare/vivid). Slow hazard: red. Multiplier: purple.
pub const ITEM_SPEED1: Color = Color::srgb(0.20, 0.45, 0.25); // tier 1 — dark/muted green
pub const ITEM_SPEED2: Color = Color::srgb(0.30, 0.75, 0.35); // tier 2 — medium green
pub const ITEM_SPEED3: Color = Color::srgb(0.45, 1.00, 0.50); // tier 3 — bright lime green
pub const ITEM_SLOW:   Color = Color::srgb(0.90, 0.25, 0.25);
pub const ITEM_MULT:   Color = Color::srgb(0.75, 0.40, 0.95);

/// Items shown in the help-menu legend: (label, swatch colour). Single source of
/// truth — keep in sync with `sim_bridge::item_color` and `entity::item::ItemKind`.
pub const ITEM_LEGEND: &[(&str, Color)] = &[
    ("Gold",       ITEM_GOLD),
    ("Speed I",    ITEM_SPEED1),
    ("Speed II",   ITEM_SPEED2),
    ("Speed III",  ITEM_SPEED3),
    ("Slow (avoid)", ITEM_SLOW),
    ("Multiplier", ITEM_MULT),
];

// ── App background / generic surfaces ───────────────────────────────────────────

pub const APP_BACKGROUND: Color = Color::srgb(0.03, 0.04, 0.07);
pub const PANEL_SCRIM:    Color = Color::srgba(0.0, 0.0, 0.0, 0.2);
pub const ROW_STRIPE:     Color = Color::srgba(1.0, 1.0, 1.0, 0.02);

// ── Floor tiles ─────────────────────────────────────────────────────────────────

pub const TILE_FREE:     Color = Color::srgb(0.06, 0.08, 0.14); // dark navy floor
pub const TILE_OBSTACLE: Color = Color::srgb(0.18, 0.20, 0.24); // dark stone grey

// ── Gizmo overlays ──────────────────────────────────────────────────────────────

pub const PATH_LINE: Color = Color::srgba(0.20, 0.90, 0.60, 0.85);
pub const PATH_GOAL: Color = Color::srgba(0.20, 0.90, 0.60, 0.45);

// ── Spawn-pocket overlay (obstacle-free 7×7 area around the base) ───────────────

pub const SPAWN_POCKET_FILL:   Color = Color::srgba(0.95, 0.22, 0.22, 0.10);
pub const SPAWN_POCKET_BORDER: Color = Color::srgba(0.95, 0.22, 0.22, 0.85);

// ── Load menu selection ─────────────────────────────────────────────────────────

pub const SELECTED_BG:     Color = Color::srgba(0.60, 0.45, 0.05, 0.40);
pub const SELECTED_BORDER: Color = Color::srgb(0.85, 0.65, 0.10);
