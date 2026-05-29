use bevy::prelude::*;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use crate::style::{ThemeColor, UiRoot, SIZE_SM, SIZE_MD, SIZE_LG};
use crate::sim_bridge::{LoadRequest, DEFAULT_CONFIG_PATH};

// ── State ─────────────────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct LoadMenuState {
    pub visible:    bool,
    pub configs:    Vec<(String, String)>,  // (label, path)
    pub policies:   Vec<(String, String)>,  // (label, path) — empty = no onnx
    pub sel_config: usize,
    pub sel_policy: usize,
}

impl Default for LoadMenuState {
    fn default() -> Self {
        Self {
            visible:    false,
            configs:    scan_configs(),
            policies:   scan_policies(),
            sel_config: 0,
            sel_policy: 0,
        }
    }
}

const WORLD_DIR:     &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/world");
const DEPLOYED_ONNX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/model/policy.onnx");
const RUNS_DIR:      &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../runs");

fn scan_configs() -> Vec<(String, String)> {
    let mut entries = vec![];

    let default_ron = format!("{WORLD_DIR}/default.ron");
    if std::path::Path::new(&default_ron).exists() {
        entries.push(("Default".to_string(), default_ron));
    }

    let Ok(dir) = std::fs::read_dir(WORLD_DIR) else { return entries; };
    let mut paths: Vec<_> = dir.flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("config_stage") && n.ends_with(".ron"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    for path in paths {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
        let label = stem.strip_prefix("config_stage")
            .map(|n| format!("Stage {n}"))
            .unwrap_or_else(|| stem.to_string());
        entries.push((label, path.to_string_lossy().into_owned()));
    }
    entries
}

fn scan_policies() -> Vec<(String, String)> {
    let mut entries = vec![("None  (BT fallback)".to_string(), String::new())];

    if std::path::Path::new(DEPLOYED_ONNX).exists() {
        entries.push(("Default (assets/model)".to_string(), DEPLOYED_ONNX.to_string()));
    }

    let Ok(dir) = std::fs::read_dir(RUNS_DIR) else { return entries; };
    let mut paths: Vec<_> = dir.flatten()
        .filter(|e| {
            let p = e.path();
            p.is_dir() && p.join("models/eval_best/policy.onnx").exists()
        })
        .map(|e| e.path())
        .collect();
    paths.sort();
    for path in paths {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown").to_string();
        let policy_path = path.join("models/eval_best/policy.onnx").to_string_lossy().into_owned();
        entries.push((name, policy_path));
    }
    entries
}

// ── Components ────────────────────────────────────────────────────────────────

#[derive(Component)] pub struct LoadMenuPanel;
#[derive(Component)] pub(crate) struct ConfigEntryBtn(usize);
#[derive(Component)] pub(crate) struct PolicyEntryBtn(usize);
#[derive(Component)] pub struct LoadConfirmBtn;
#[derive(Component)] pub struct LoadCancelBtn;

const SELECTED_BG:     Color = Color::srgba(0.60, 0.45, 0.05, 0.40);
const SELECTED_BORDER: Color = Color::srgb(0.85, 0.65, 0.10);

// ── Spawn ─────────────────────────────────────────────────────────────────────

pub fn spawn_load_menu(mut commands: Commands, state: Res<LoadMenuState>) {
    let bg     = ThemeColor::Background.resolve();
    let border = ThemeColor::Border.resolve();
    let dim    = ThemeColor::TextDim.resolve();

    commands.spawn((
        UiRoot,
        LoadMenuPanel,
        Node {
            display:        Display::None,
            position_type:  PositionType::Absolute,
            left:           Val::Percent(50.0),
            top:            Val::Percent(50.0),
            margin:         UiRect { left: Val::Px(-300.0), top: Val::Px(-200.0), ..default() },
            flex_direction: FlexDirection::Column,
            min_width:      Val::Px(600.0),
            padding:        UiRect::all(Val::Px(24.0)),
            row_gap:        Val::Px(14.0),
            border:         UiRect::all(Val::Px(1.0)),
            border_radius:  BorderRadius::all(Val::Px(12.0)),
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(border),
        ZIndex(600),
    )).with_children(|modal| {
        modal.spawn((
            Text::new("Load Config & Policy"),
            TextFont { font_size: SIZE_LG, ..default() },
            TextColor(ThemeColor::TextPrimary.resolve()),
        ));
        modal.spawn((Node { height: Val::Px(1.0), ..default() }, BackgroundColor(border)));

        // Two-column body
        modal.spawn(Node {
            flex_direction: FlexDirection::Row,
            column_gap:     Val::Px(20.0),
            ..default()
        }).with_children(|row| {
            // Left — World Config
            row.spawn(Node {
                flex_direction: FlexDirection::Column,
                row_gap:        Val::Px(4.0),
                flex_grow:      1.0,
                ..default()
            }).with_children(|col| {
                col.spawn((
                    Text::new("World Config"),
                    TextFont { font_size: SIZE_SM, ..default() },
                    TextColor(dim),
                ));
                for (i, (label, _)) in state.configs.iter().enumerate() {
                    spawn_entry_btn(col, label, i == state.sel_config, ConfigEntryBtn(i));
                }
            });

            // Right — Policy Model
            row.spawn(Node {
                flex_direction: FlexDirection::Column,
                row_gap:        Val::Px(4.0),
                flex_grow:      1.0,
                ..default()
            }).with_children(|col| {
                col.spawn((
                    Text::new("Policy Model"),
                    TextFont { font_size: SIZE_SM, ..default() },
                    TextColor(dim),
                ));
                for (i, (label, _)) in state.policies.iter().enumerate() {
                    spawn_entry_btn(col, label, i == state.sel_policy, PolicyEntryBtn(i));
                }
            });
        });

        modal.spawn((Node { height: Val::Px(1.0), ..default() }, BackgroundColor(border)));

        // Footer buttons
        modal.spawn(Node {
            flex_direction:  FlexDirection::Row,
            justify_content: JustifyContent::FlexEnd,
            column_gap:      Val::Px(8.0),
            ..default()
        }).with_children(|footer| {
            footer_btn(footer, "Cancel",          ThemeColor::TextDim.resolve(),     LoadCancelBtn);
            footer_btn(footer, "Load & Restart",  ThemeColor::SuccessText.resolve(), LoadConfirmBtn);
        });
    });
}

fn spawn_entry_btn<M: Component>(
    parent: &mut ChildSpawnerCommands,
    label:  &str,
    active: bool,
    marker: M,
) {
    let bg     = if active { SELECTED_BG }     else { ThemeColor::ButtonIdle.resolve() };
    let border = if active { SELECTED_BORDER } else { ThemeColor::Border.resolve() };
    parent.spawn((
        Button,
        marker,
        Node {
            padding:       UiRect::axes(Val::Px(12.0), Val::Px(6.0)),
            border_radius: BorderRadius::all(Val::Px(6.0)),
            border:        UiRect::all(Val::Px(1.0)),
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(border),
    )).with_children(|btn| {
        btn.spawn((
            Text::new(label.to_string()),
            TextFont { font_size: SIZE_SM, ..default() },
            TextColor(ThemeColor::TextPrimary.resolve()),
        ));
    });
}

fn footer_btn<M: Component>(
    parent: &mut ChildSpawnerCommands,
    label:  &str,
    color:  Color,
    marker: M,
) {
    parent.spawn((
        Button,
        marker,
        Node {
            padding:       UiRect::axes(Val::Px(20.0), Val::Px(8.0)),
            border_radius: BorderRadius::all(Val::Px(6.0)),
            border:        UiRect::all(Val::Px(1.0)),
            ..default()
        },
        BackgroundColor(ThemeColor::ButtonIdle.resolve()),
        BorderColor::all(ThemeColor::Border.resolve()),
    )).with_children(|btn| {
        btn.spawn((
            Text::new(label),
            TextFont { font_size: SIZE_MD, ..default() },
            TextColor(color),
        ));
    });
}

// ── Systems ───────────────────────────────────────────────────────────────────

pub fn toggle_load_menu(
    keys:      Res<ButtonInput<KeyCode>>,
    mut state: ResMut<LoadMenuState>,
) {
    if keys.just_pressed(KeyCode::KeyL) {
        state.visible = !state.visible;
    }
    if keys.just_pressed(KeyCode::Escape) && state.visible {
        state.visible = false;
    }
}

pub fn sync_load_menu_visibility(
    state: Res<LoadMenuState>,
    mut q: Query<&mut Node, With<LoadMenuPanel>>,
) {
    if !state.is_changed() { return; }
    for mut node in q.iter_mut() {
        node.display = if state.visible { Display::Flex } else { Display::None };
    }
}

pub fn handle_entry_clicks(
    config_q:  Query<(&Interaction, &ConfigEntryBtn), Changed<Interaction>>,
    policy_q:  Query<(&Interaction, &PolicyEntryBtn), Changed<Interaction>>,
    mut state: ResMut<LoadMenuState>,
) {
    for (interaction, btn) in config_q.iter() {
        if *interaction == Interaction::Pressed { state.sel_config = btn.0; }
    }
    for (interaction, btn) in policy_q.iter() {
        if *interaction == Interaction::Pressed { state.sel_policy = btn.0; }
    }
}

pub fn sync_entry_highlights(
    state:     Res<LoadMenuState>,
    mut cfg_q: Query<(&ConfigEntryBtn, &mut BackgroundColor, &mut BorderColor), Without<PolicyEntryBtn>>,
    mut pol_q: Query<(&PolicyEntryBtn, &mut BackgroundColor, &mut BorderColor), Without<ConfigEntryBtn>>,
) {
    if !state.is_changed() { return; }
    for (btn, mut bg, mut border) in cfg_q.iter_mut() {
        let active = btn.0 == state.sel_config;
        bg.0    = if active { SELECTED_BG }     else { ThemeColor::ButtonIdle.resolve() };
        *border = if active { BorderColor::all(SELECTED_BORDER) } else { BorderColor::all(ThemeColor::Border.resolve()) };
    }
    for (btn, mut bg, mut border) in pol_q.iter_mut() {
        let active = btn.0 == state.sel_policy;
        bg.0    = if active { SELECTED_BG }     else { ThemeColor::ButtonIdle.resolve() };
        *border = if active { BorderColor::all(SELECTED_BORDER) } else { BorderColor::all(ThemeColor::Border.resolve()) };
    }
}

pub fn handle_load_confirm(
    btn_q:     Query<&Interaction, (Changed<Interaction>, With<LoadConfirmBtn>)>,
    mut state: ResMut<LoadMenuState>,
    mut req:   ResMut<LoadRequest>,
) {
    for interaction in btn_q.iter() {
        if *interaction == Interaction::Pressed {
            req.config_path = state.configs.get(state.sel_config)
                .map(|(_, p)| p.clone())
                .unwrap_or_else(|| DEFAULT_CONFIG_PATH.to_string());
            req.policy_path = state.policies.get(state.sel_policy)
                .map(|(_, p)| p.clone())
                .unwrap_or_default();
            req.pending = true;
            state.visible = false;
        }
    }
}

pub fn handle_load_cancel(
    btn_q:     Query<&Interaction, (Changed<Interaction>, With<LoadCancelBtn>)>,
    mut state: ResMut<LoadMenuState>,
) {
    for interaction in btn_q.iter() {
        if *interaction == Interaction::Pressed { state.visible = false; }
    }
}