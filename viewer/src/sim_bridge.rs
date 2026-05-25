use bevy::prelude::*;
use atb::engine::{SimCore, AgentState, ItemState};
use atb::engine::enemy::{EnemyPathCache, compute_action, navigate_action};
use atb::world::config::EnemyKind;
use atb::world::coords::GridPos;
use atb::world::grid::Grid;
use atb::entity::item::ItemKind;
use atb::config::AGENT_MAX_GOLD;
use atb::rl::action::{action_to_int, ACTION_SIZE, ACTION_WAIT};
use atb::algorithm::behavior::goap;
use crate::policy::OnnxPolicy;
use crate::sim_config::{SimConfig, TickTimer};
use crate::viz::events::RestartPending;

const CONFIG_PATH: &str = "assets/world/config.ron";
const ONNX_POLICY_PATH: &str = crate::ONNX_POLICY_PATH;

// ── Policy mode ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolicyMode {
    #[default]
    Onnx,
    Random,
    BehaviorTree,
    Goap,
}

impl PolicyMode {
    pub fn next(self) -> Self {
        match self {
            Self::Onnx         => Self::Random,
            Self::Random       => Self::BehaviorTree,
            Self::BehaviorTree => Self::Goap,
            Self::Goap         => Self::Onnx,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Onnx         => "ONNX",
            Self::Random       => "RANDOM",
            Self::BehaviorTree => "BT",
            Self::Goap         => "GOAP",
        }
    }
}

// ── Resource ──────────────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct SimBridge {
    pub sim:            SimCore,
    pub game_over:      bool,
    pub last_action:    u32,
    pub action_counts:  [u32; ACTION_SIZE],
    pub episode_reward: f32,
    pub mode:           PolicyMode,
    policy:             Option<OnnxPolicy>,
    bt_cache:           EnemyPathCache,
    goap_cache:         EnemyPathCache,
}

fn normalize_path(raw: &str) -> std::path::PathBuf {
    use std::path::{Component, Path, PathBuf};
    Path::new(raw).components().fold(PathBuf::new(), |mut acc, c| {
        if c == Component::ParentDir { acc.pop(); } else { acc.push(c); }
        acc
    })
}

impl SimBridge {
    fn new() -> Self {
        let onnx_path = normalize_path(ONNX_POLICY_PATH);
        let policy = match OnnxPolicy::load(&onnx_path.to_string_lossy()) {
            Ok(p)  => { info!("ONNX policy loaded"); Some(p) }
            Err(e) => { warn!("No ONNX policy found ({e}), using BT agent"); None }
        };
        let mode = if policy.is_some() { PolicyMode::Onnx } else { PolicyMode::BehaviorTree };
        Self {
            sim: SimCore::new(CONFIG_PATH), game_over: false, policy,
            last_action: ACTION_WAIT, action_counts: [0; ACTION_SIZE], episode_reward: 0.0,
            mode,
            bt_cache:   EnemyPathCache::new(),
            goap_cache: EnemyPathCache::new(),
        }
    }

    pub fn tick(&self) -> u64 { self.sim.tick }
    pub fn grid(&self) -> &Grid { &self.sim.grid }
    pub fn agents(&self) -> &[AgentState] { &self.sim.agents }
    pub fn items(&self) -> &[ItemState] { &self.sim.items }
    pub fn bt_path(&self) -> &std::collections::VecDeque<atb::world::coords::GridPos> { self.bt_cache.path() }
    pub fn goap_path(&self) -> &std::collections::VecDeque<atb::world::coords::GridPos> { self.goap_cache.path() }

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
        bridge.bt_cache       = EnemyPathCache::new();
        bridge.goap_cache     = EnemyPathCache::new();
        bridge.game_over      = false;
        bridge.last_action    = ACTION_WAIT;
        bridge.action_counts  = [0; ACTION_SIZE];
        bridge.episode_reward = 0.0;
        restart.0 = false;
    }

    if cfg.paused || bridge.game_over { return; }

    timer.0.tick(time.delta());
    let steps = timer.0.times_finished_this_tick();
    for _ in 0..steps {
        let action = match bridge.mode {
            PolicyMode::Onnx => {
                if let Some(ref p) = bridge.policy {
                    p.act(&bridge.sim.obs_buf)
                } else {
                    ACTION_WAIT
                }
            }
            PolicyMode::Random => {
                // PCG-style LCG — different action every tick, no rng dependency.
                // Multiplier from Knuth vol.2 (MMIX LCG), shift discards low-quality low bits.
                const LCG_MUL: u64 = 6364136223846793005;
                let x = bridge.sim.tick.wrapping_mul(LCG_MUL).wrapping_add(1);
                (x >> 38) as u32 % ACTION_SIZE as u32
            }
            PolicyMode::BehaviorTree => {
                // Deref once to &mut SimBridge so the borrow checker sees
                // `sim` and `bt_cache` as disjoint fields.
                let b = &mut *bridge;
                if b.sim.agents.is_empty() { ACTION_WAIT } else {
                    let agent = b.sim.agents[0].clone();
                    let act   = compute_action(
                        EnemyKind::BehaviorTree,
                        &agent,
                        &b.sim.items,
                        &b.sim.grid,
                        &mut b.bt_cache,
                    );
                    action_to_int(act)
                }
            }
            PolicyMode::Goap => {
                let b = &mut *bridge;
                if b.sim.agents.is_empty() { ACTION_WAIT } else {
                    let agent  = b.sim.agents[0].clone();
                    let agents = b.sim.agents.clone();
                    goap_action(&agent, &agents, &b.sim.items, &b.sim.grid, &mut b.goap_cache)
                }
            }
        };
        bridge.last_action = action;
        bridge.action_counts[action as usize] += 1;
        let (rew, done) = bridge.sim.step(action);
        bridge.episode_reward += rew;
        if done {
            bridge.game_over = true;
            break;
        }
    }
}

fn goap_action(
    agent:  &AgentState,
    agents: &[AgentState],
    items:  &[ItemState],
    grid:   &Grid,
    cache:  &mut EnemyPathCache,
) -> u32 {
    const NEAR_SQ:          i32 = 10 * 10;
    const LOW_HEALTH_THRESH: u8 = 1;

    let gold_nearby = items.iter()
        .filter(|i| i.kind == ItemKind::Gold)
        .any(|i| agent.pos.dist_sq(i.pos) <= NEAR_SQ);

    let enemy_nearby = agents.iter()
        .filter(|a| a.team != agent.team)
        .any(|a| agent.pos.dist_sq(a.pos) <= NEAR_SQ);

    let dist_to_base = {
        let dx = agent.pos.x - agent.base_pos.x;
        let dy = agent.pos.y - agent.base_pos.y;
        dx.abs() + dy.abs()
    };
    let dist_to_gold = items.iter()
        .filter(|i| i.kind == ItemKind::Gold)
        .map(|i| {
            let dx = agent.pos.x - i.pos.x;
            let dy = agent.pos.y - i.pos.y;
            dx.abs() + dy.abs()
        })
        .min()
        .unwrap_or(i32::MAX);

    let ws = {
        let mut bits = 0u64;
        if agent.gold_carried > 0                          { bits |= goap::BIT_HAS_GOLD; }
        if agent.gold_carried >= AGENT_MAX_GOLD            { bits |= goap::BIT_INVENTORY_FULL; }
        if agent.gold_carried >= AGENT_MAX_GOLD / 2        { bits |= goap::BIT_INVENTORY_HALF; }
        if agent.pos == agent.base_pos                     { bits |= goap::BIT_ON_OWN_BASE; }
        if gold_nearby                                     { bits |= goap::BIT_GOLD_NEARBY; }
        if enemy_nearby                                    { bits |= goap::BIT_ENEMY_NEARBY; }
        if dist_to_base < dist_to_gold                     { bits |= goap::BIT_BASE_CLOSER; }
        if agent.hearts <= LOW_HEALTH_THRESH               { bits |= goap::BIT_LOW_HEALTH; }
        goap::WorldState(bits)
    };

    // Goal: have collected gold and returned to base.
    // BIT_INVENTORY_HALF gets set by collect_gold; BIT_ON_OWN_BASE by navigate_to_base.
    // Both must be set before goal is satisfied, producing a full collect→deposit plan.
    let goal = goap::GoalState(goap::BIT_INVENTORY_HALF | goap::BIT_ON_OWN_BASE);

    let first_step = goap::plan(ws, goal, goap::ACTIONS, goap::PlanConfig::default())
        .ok()
        .and_then(|r| r.steps.into_iter().next());

    let nav_target: Option<GridPos> = match first_step.as_deref() {
        Some(goap::ACT_NAVIGATE_TO_GOLD) | Some(goap::ACT_COLLECT_GOLD) => {
            items.iter()
                .filter(|i| i.kind == ItemKind::Gold)
                .min_by_key(|i| {
                    let dx = agent.pos.x - i.pos.x;
                    let dy = agent.pos.y - i.pos.y;
                    dx.abs() + dy.abs()
                })
                .map(|i| i.pos)
        }
        Some(goap::ACT_NAVIGATE_TO_BASE) | Some(goap::ACT_DROP_GOLD) => Some(agent.base_pos),
        Some(goap::ACT_FLEE) => Some(agent.base_pos), // retreat to own base
        _ => None,
    };

    match nav_target {
        None         => ACTION_WAIT,
        Some(target) => action_to_int(navigate_action(agent, target, grid, cache)),
    }
}

fn item_color(kind: ItemKind) -> Color {
    use crate::style::color::{ITEM_AMMO, ITEM_GOLD, ITEM_HEALTH, ITEM_SPEED};
    match kind {
        ItemKind::Gold       => ITEM_GOLD,
        ItemKind::Health     => ITEM_HEALTH,
        ItemKind::Ammo       => ITEM_AMMO,
        ItemKind::SpeedBoost => ITEM_SPEED,
    }
}
