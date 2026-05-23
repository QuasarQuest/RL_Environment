use bevy::prelude::*;
use crate::style::{ThemeColor, UiRoot, SIZE_SM, SIZE_MD, SIZE_XL, TOOLBAR_H, FONT_ICON};
use crate::style::color::{team_color, GOLD_500, GOLD_800};
use crate::viz::core_ui::button::spawn_icon_button;
use crate::viz::core_ui::panel::spawn_button_group;
use super::components::{
    TickLabelMarker, TimeLabelMarker, TeamScoreMarker,
    SpeedDecreaseButton, SpeedIncreaseButton,
    CurrentSpeedLabel, PauseButtonMarker, PauseButtonText,
};

pub fn spawn_hud(mut commands: Commands, asset_server: Res<AssetServer>) {
    let bg         = ThemeColor::Background.resolve();
    let border     = ThemeColor::Border.resolve();
    let dim        = ThemeColor::TextDim.resolve();
    let red_color  = team_color(0);
    let blue_color = team_color(1);
    let icon_font  = asset_server.load(FONT_ICON);

    commands.spawn((
        UiRoot,
        Node {
            width:           Val::Percent(100.0),
            height:          Val::Px(TOOLBAR_H),
            position_type:   PositionType::Absolute,
            top:             Val::Px(0.0),
            left:            Val::Px(0.0),
            flex_direction:  FlexDirection::Row,
            align_items:     AlignItems::Center,
            justify_content: JustifyContent::SpaceBetween,
            padding:         UiRect::axes(Val::Px(16.0), Val::Px(0.0)),
            border:          UiRect::bottom(Val::Px(1.0)),
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(border),
        ZIndex(100),
    )).with_children(|top_bar| {

        // ── Left: scoreboard RED | tick | BLUE ───────────────────────────────
        top_bar.spawn(Node {
            flex_direction: FlexDirection::Row,
            align_items:    AlignItems::Stretch,
            ..default()
        }).with_children(|left| {
            // Team 0
            left.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items:    AlignItems::Center,
                column_gap:     Val::Px(8.0),
                padding:        UiRect::horizontal(Val::Px(16.0)),
                ..default()
            }).with_children(|t0| {
                t0.spawn((Text::new("RED"), TextFont { font_size: SIZE_SM, ..default() }, TextColor(red_color.with_alpha(0.7))));
                t0.spawn((Text::new("0"), TextFont { font_size: SIZE_XL, ..default() }, TextColor(red_color), TeamScoreMarker(0)));
            });

            // Separator
            left.spawn((Node { width: Val::Px(1.0), margin: UiRect::vertical(Val::Px(10.0)), ..default() }, BackgroundColor(border)));

            // Tick counter
            left.spawn(Node {
                flex_direction:  FlexDirection::Column,
                align_items:     AlignItems::Center,
                justify_content: JustifyContent::Center,
                padding:         UiRect::horizontal(Val::Px(16.0)),
                row_gap:         Val::Px(2.0),
                ..default()
            }).with_children(|mid| {
                mid.spawn((Text::new("0"), TextFont { font_size: SIZE_XL, ..default() }, TextColor(ThemeColor::TextPrimary.resolve()), TickLabelMarker));
                mid.spawn((Text::new("0 / 0"), TextFont { font_size: SIZE_SM, ..default() }, TextColor(dim), TimeLabelMarker));
            });

            // Separator
            left.spawn((Node { width: Val::Px(1.0), margin: UiRect::vertical(Val::Px(10.0)), ..default() }, BackgroundColor(border)));

            // Team 1
            left.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items:    AlignItems::Center,
                column_gap:     Val::Px(8.0),
                padding:        UiRect::horizontal(Val::Px(16.0)),
                ..default()
            }).with_children(|t1| {
                t1.spawn((Text::new("0"), TextFont { font_size: SIZE_XL, ..default() }, TextColor(blue_color), TeamScoreMarker(1)));
                t1.spawn((Text::new("BLUE"), TextFont { font_size: SIZE_SM, ..default() }, TextColor(blue_color.with_alpha(0.7))));
            });
        });

        // ── Centre: speed + pause controls ───────────────────────────────────
        top_bar.spawn(Node {
            flex_direction:  FlexDirection::Row,
            align_items:     AlignItems::Center,
            justify_content: JustifyContent::Center,
            column_gap:      Val::Px(8.0),
            ..default()
        }).with_children(|centre| {
            spawn_button_group(centre, |grp| {
                spawn_icon_button(grp, "◀", SpeedDecreaseButton);
                grp.spawn((
                    Text::new("10x"),
                    TextFont  { font_size: SIZE_MD, ..default() },
                    TextColor(dim),
                    CurrentSpeedLabel,
                ));
                spawn_icon_button(grp, "▶", SpeedIncreaseButton);
            });

            // Pause button
            centre.spawn((
                Button,
                PauseButtonMarker,
                Node {
                    padding:         UiRect::axes(Val::Px(16.0), Val::Px(6.0)),
                    justify_content: JustifyContent::Center,
                    align_items:     AlignItems::Center,
                    border_radius:   BorderRadius::all(Val::Px(4.0)),
                    ..default()
                },
                BackgroundColor(GOLD_800),
                BorderColor::all(border),
            )).with_children(|btn| {
                btn.spawn((Text::new("⏸"), TextFont { font_size: SIZE_XL, font: icon_font.clone(), ..default() }, TextColor(GOLD_500), PauseButtonText));
            });
        });

        // ── Right: help hint ─────────────────────────────────────────────────
        top_bar.spawn(Node {
            flex_direction: FlexDirection::Row,
            align_items:    AlignItems::Center,
            column_gap:     Val::Px(12.0),
            padding:        UiRect::horizontal(Val::Px(8.0)),
            ..default()
        }).with_children(|right| {
            right.spawn((Text::new("TAB"), TextFont { font_size: SIZE_SM, ..default() }, TextColor(dim)));
            right.spawn((Text::new("Scoreboard"), TextFont { font_size: SIZE_SM, ..default() }, TextColor(dim.with_alpha(0.5))));
            right.spawn((Node { width: Val::Px(1.0), height: Val::Px(16.0), ..default() }, BackgroundColor(border)));
            right.spawn((Text::new("H"), TextFont { font_size: SIZE_SM, ..default() }, TextColor(dim)));
            right.spawn((Text::new("Help"), TextFont { font_size: SIZE_SM, ..default() }, TextColor(dim.with_alpha(0.5))));
        });
    });
}
