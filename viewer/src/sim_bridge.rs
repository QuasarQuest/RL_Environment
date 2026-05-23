use bevy::prelude::*;
use atb::sim_core::{SimCore, AgentState, ItemState};
use atb::world::grid::Grid;
use atb::item::ItemKind;
use crate::sim_config::{SimConfig, TickTimer};
use crate::viz::events::RestartPending;

const CONFIG_PATH: &str = "assets/world/config.ron";

// ── Resource ──────────────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct SimBridge {
    pub sim:      SimCore,
    pub game_over: bool,
}

impl SimBridge {
    fn new() -> Self {
        Self {
            sim:      SimCore::new(CONFIG_PATH),
            game_over: false,
        }
    }

    pub fn tick(&self) -> u64 { self.sim.tick }
    pub fn grid(&self) -> &Grid { &self.sim.grid }
    pub fn agents(&self) -> &[AgentState] { &self.sim.agents }
    pub fn items(&self) -> &[ItemState] { &self.sim.items }

    pub fn remaining_display(&self) -> String {
        format!("{} / {}", self.sim.tick, self.sim.match_ticks())
    }

    pub fn team_score(&self, team: u8) -> u32 {
        self.sim.agents.iter()
            .filter(|a| a.team == team)
            .map(|a| a.score)
            .sum()
    }

    pub fn n_teams(&self) -> u8 {
        self.sim.agents.iter().map(|a| a.team).max().map(|t| t + 1).unwrap_or(0)
    }
}

// ── ECS markers spawned for each agent / item ─────────────────────────────────

#[derive(Component)]
pub struct AgentMarker;

#[derive(Component)]
pub struct AgentIndex(pub usize);

#[derive(Component)]
pub struct ItemMarker;

#[derive(Component)]
pub struct ItemIndex(pub usize);

// ── Plugin ────────────────────────────────────────────────────────────────────

pub struct SimBridgePlugin;

impl Plugin for SimBridgePlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<SimConfig>()
            .init_resource::<TickTimer>()
            .init_resource::<RestartPending>()
            .add_systems(Startup, setup_sim)
            .add_systems(Update, step_sim.in_set(crate::viz::SimSet));
    }
}

fn setup_sim(mut commands: Commands) {
    let bridge = SimBridge::new();

    // Spawn one display entity per agent.
    for (i, agent) in bridge.sim.agents.iter().enumerate() {
        commands.spawn((
            AgentMarker,
            AgentIndex(i),
            Sprite {
                color:       crate::style::color::team_color(agent.team),
                custom_size: Some(Vec2::splat(atb::config::TILE_SIZE * 0.75)),
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, 1.0),
        ));
    }

    // Spawn one display entity per item slot (items respawn, count stays fixed).
    for (i, item) in bridge.sim.items.iter().enumerate() {
        commands.spawn((
            ItemMarker,
            ItemIndex(i),
            Sprite {
                color:       item_color(item.kind),
                custom_size: Some(Vec2::splat(atb::config::TILE_SIZE * 0.45)),
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, 0.5),
        ));
    }

    commands.insert_resource(bridge);
}

pub fn step_sim(
    mut bridge:  ResMut<SimBridge>,
    mut timer:   ResMut<TickTimer>,
    cfg:         Res<SimConfig>,
    time:        Res<Time>,
    mut restart: ResMut<RestartPending>,
) {
    if restart.0 {
        bridge.sim.reset();
        bridge.game_over = false;
        restart.0 = false;
    }

    if cfg.paused || bridge.game_over { return; }

    timer.0.tick(time.delta());
    if timer.0.just_finished() {
        let (_rew, done) = bridge.sim.step(25); // Wait action; ONNX will replace this
        if done {
            bridge.game_over = true;
        }
    }
}

fn item_color(kind: ItemKind) -> Color {
    match kind {
        ItemKind::Gold       => Color::srgb(0.95, 0.78, 0.20),
        ItemKind::Health     => Color::srgb(0.90, 0.30, 0.30),
        ItemKind::Ammo       => Color::srgb(0.30, 0.60, 0.95),
        ItemKind::SpeedBoost => Color::srgb(0.30, 0.90, 0.50),
    }
}
