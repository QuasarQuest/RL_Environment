use bevy::prelude::*;
use atb::engine::{SimCore, AgentState, ItemState, MAX_OPTION_TICKS};
use atb::world::grid::Grid;
use atb::entity::item::ItemKind;
use atb::config::AGENT_MAX_GOLD;
use atb::rl::action::{int_to_rl_action, ACTION_SIZE, ACTION_WAIT, CLUSTER_K};
use atb::algorithm::behavior::goap;
use crate::policy::OnnxPolicy;
use crate::sim_config::{SimConfig, TickTimer};
use crate::viz::events::RestartPending;
use crate::viz::grid_offset::GridOffset;
use crate::viz::renderer::tile_renderer::{TileMarker, do_spawn_tiles};

// Action indices — derived from CLUSTER_K so they track the RlAction ordering
// in the sim (0..CLUSTER_K are region-nav slots; NavigateToBase then Wait follow).
const ACTION_NAVIGATE_TO_BASE: u32 = CLUSTER_K as u32;       // 9

pub const DEFAULT_CONFIG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/world/default.ron");
const DEFAULT_ONNX_PATH:       &str = crate::ONNX_POLICY_PATH;

#[derive(Resource, Default)]
pub struct LoadRequest {
    pub pending:     bool,
    pub config_path: String,
    pub policy_path: String,
}

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

    // BT: committed cluster slot — only reselect when this slot becomes empty.
    // Prevents oscillation when two clusters are equidistant and swap rank each tick.
    bt_target_cluster: Option<u32>,

    // ONNX: option commitment so the trained agent is shown faithfully. The policy
    // is trained with temporally-extended options (SimCore::step_option), so the
    // viewer must keep executing a chosen navigation action until the option ends
    // (event / arrival / timeout) rather than re-querying every tick — otherwise it
    // thrashes and looks broken. Mirrors step_option's termination conditions.
    onnx_action:       Option<u32>,  // committed option action
    onnx_anchor_gold:  u8,           // gold_carried at option start (event detection)
    onnx_anchor_score: u32,          // score at option start
    onnx_option_ticks: u64,
    onnx_moved:        bool,         // did the agent move on the previous tick
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
        let onnx_path = normalize_path(DEFAULT_ONNX_PATH);
        let policy = match OnnxPolicy::load(&onnx_path.to_string_lossy()) {
            Ok(p)  => { info!("ONNX policy loaded"); Some(p) }
            Err(e) => { warn!("No ONNX policy found ({e}), using BT agent"); None }
        };
        let mode = if policy.is_some() { PolicyMode::Onnx } else { PolicyMode::BehaviorTree };
        Self {
            sim: SimCore::new(DEFAULT_CONFIG_PATH), game_over: false, policy,
            last_action: ACTION_WAIT, action_counts: [0; ACTION_SIZE], episode_reward: 0.0,
            mode,
            bt_target_cluster: None,
            onnx_action: None,
            onnx_anchor_gold: 0,
            onnx_anchor_score: 0,
            onnx_option_ticks: 0,
            onnx_moved: false,
        }
    }

    pub fn tick(&self) -> u64 { self.sim.tick }
    pub fn grid(&self) -> &Grid { &self.sim.grid }
    pub fn agents(&self) -> &[AgentState] { &self.sim.agents }
    pub fn items(&self) -> &[ItemState] { &self.sim.items }
    pub fn agent_path(&self) -> &std::collections::VecDeque<atb::world::coords::GridPos> {
        self.sim.agent_path()
    }

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
            .init_resource::<LoadRequest>()
            .add_systems(Startup, setup_sim)
            .add_systems(Update, handle_load_request.before(crate::viz::SimSet))
            .add_systems(Update, step_sim.in_set(crate::viz::SimSet));
    }
}

fn spawn_sim_entities(commands: &mut Commands, bridge: &SimBridge) {
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
}

fn setup_sim(mut commands: Commands) {
    let bridge = SimBridge::new();
    spawn_sim_entities(&mut commands, &bridge);
    commands.insert_resource(bridge);
}

pub fn handle_load_request(
    mut req:      ResMut<LoadRequest>,
    mut commands: Commands,
    mut bridge:   ResMut<SimBridge>,
    mut offset:   ResMut<GridOffset>,
    agent_q:      Query<Entity, With<AgentMarker>>,
    item_q:       Query<Entity, With<ItemMarker>>,
    tile_q:       Query<Entity, With<TileMarker>>,
) {
    if !req.pending { return; }
    req.pending = false;

    for e in agent_q.iter() { commands.entity(e).despawn(); }
    for e in item_q.iter()  { commands.entity(e).despawn(); }
    for e in tile_q.iter()  { commands.entity(e).despawn(); }

    bridge.policy = if req.policy_path.is_empty() {
        info!("No policy — falling back to BT");
        None
    } else {
        let p = normalize_path(&req.policy_path);
        match OnnxPolicy::load(&p.to_string_lossy()) {
            Ok(p)  => { info!("Policy loaded: {}", req.policy_path); Some(p) }
            Err(e) => { warn!("Policy load failed ({e}), falling back to BT"); None }
        }
    };

    bridge.sim              = SimCore::new(&req.config_path);
    bridge.game_over        = false;
    bridge.last_action      = ACTION_WAIT;
    bridge.action_counts    = [0; ACTION_SIZE];
    bridge.episode_reward   = 0.0;
    bridge.bt_target_cluster = None;
    bridge.onnx_action       = None;
    bridge.mode = if bridge.policy.is_some() { PolicyMode::Onnx } else { PolicyMode::BehaviorTree };

    // Recompute GridOffset — the new config may have different map dimensions.
    let step = atb::config::TILE_SIZE + atb::config::TILE_GAP;
    let (gw, gh) = (bridge.grid().width, bridge.grid().height);
    *offset = GridOffset {
        x: -(gw as f32 * step) / 2.0 + step / 2.0,
        y: -(gh as f32 * step) / 2.0 + step / 2.0,
        step,
    };

    // Respawn tiles for the new grid (obstacles, base tiles may differ per stage).
    do_spawn_tiles(&mut commands, &bridge, &offset);

    spawn_sim_entities(&mut commands, &bridge);
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
        bridge.game_over         = false;
        bridge.last_action       = ACTION_WAIT;
        bridge.action_counts     = [0; ACTION_SIZE];
        bridge.episode_reward    = 0.0;
        bridge.bt_target_cluster = None;
        bridge.onnx_action       = None;
        if let Some(ref mut p) = bridge.policy { p.reset(); }
        restart.0 = false;
    }

    if cfg.paused || bridge.game_over { return; }

    timer.0.tick(time.delta());
    let steps = timer.0.times_finished_this_tick();
    for _ in 0..steps {
        let action = match bridge.mode {
            PolicyMode::Onnx => {
                if bridge.sim.agents.is_empty() || bridge.policy.is_none() {
                    ACTION_WAIT
                } else {
                    // Option commitment: keep executing the committed navigation
                    // action until the option ends (mirrors SimCore::step_option),
                    // so the option-trained policy is shown faithfully instead of
                    // re-querying — and thrashing — every tick.
                    let agent_gold  = bridge.sim.agents[0].gold_carried;
                    let agent_score = bridge.sim.agents[0].score;
                    let need_new = match bridge.onnx_action {
                        None => true,
                        Some(a) => {
                            !int_to_rl_action(a).is_navigation()       // Wait: single tick
                            || bridge.onnx_option_ticks >= MAX_OPTION_TICKS
                            || agent_gold  != bridge.onnx_anchor_gold  // pickup event
                            || agent_score != bridge.onnx_anchor_score // deposit event
                            || !bridge.onnx_moved                      // arrived / blocked
                        }
                    };
                    if need_new {
                        let obs  = bridge.sim.obs_buf.to_vec();
                        let mask = bridge.sim.action_mask();
                        let act  = bridge.policy.as_mut().unwrap().act_masked(&obs, &mask);
                        bridge.onnx_action       = Some(act);
                        bridge.onnx_anchor_gold  = agent_gold;
                        bridge.onnx_anchor_score = agent_score;
                        bridge.onnx_option_ticks = 0;
                    }
                    bridge.onnx_action.unwrap_or(ACTION_WAIT)
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
                if bridge.sim.agents.is_empty() { ACTION_WAIT } else {
                    let agent = &bridge.sim.agents[0];
                    if agent.gold_carried >= AGENT_MAX_GOLD {
                        bridge.bt_target_cluster = None;
                        if bridge.sim.tick % 50 == 0 {
                            debug!("[BT] tick={} full inventory → navigate_to_base", bridge.sim.tick);
                        }
                        ACTION_NAVIGATE_TO_BASE
                    } else {
                        // Reselect when: (a) committed slot is gone, or (b) another cluster
                        // has gold that is meaningfully closer (hysteresis of 3 tiles prevents
                        // oscillation when clusters are equidistant).
                        let slot_valid = bridge.bt_target_cluster.map_or(false, |k| {
                            use atb::engine::clusters::{chebyshev, find_clusters};
                            let golds: Vec<_> = bridge.sim.items.iter()
                                .filter(|i| i.kind == ItemKind::Gold)
                                .map(|i| i.pos)
                                .collect();
                            let agent_pos = bridge.sim.agents[0].pos;
                            let (gw, gh) = (bridge.sim.grid.width as i32, bridge.sim.grid.height as i32);
                            let clusters  = find_clusters(&golds, gw, gh);
                            let committed_dist = clusters.get(k as usize)
                                .and_then(|c| c.as_ref())
                                .and_then(|c| c.nearest_gold(agent_pos))
                                .map(|g| chebyshev(agent_pos, g));
                            match committed_dist {
                                None => false,
                                Some(cd) => {
                                    let best = clusters.iter().enumerate()
                                        .filter(|&(i, _)| i != k as usize)
                                        .filter_map(|(_, c)| {
                                            c.as_ref()?.nearest_gold(agent_pos)
                                                .map(|g| chebyshev(agent_pos, g))
                                        })
                                        .min()
                                        .unwrap_or(i32::MAX);
                                    cd <= best + 3
                                }
                            }
                        });
                        if !slot_valid {
                            let new_k = bt_nearest_cluster(&bridge.sim);
                            bridge.bt_target_cluster = if new_k != ACTION_WAIT { Some(new_k) } else { None };
                            let agent = &bridge.sim.agents[0];
                            debug!(
                                "[BT] tick={} cluster → {:?}  pos=({},{})",
                                bridge.sim.tick, bridge.bt_target_cluster,
                                agent.pos.x, agent.pos.y,
                            );
                        }
                        bridge.bt_target_cluster.unwrap_or(ACTION_WAIT)
                    }
                }
            }
            PolicyMode::Goap => {
                if bridge.sim.agents.is_empty() { ACTION_WAIT } else {
                    // Replan every tick — the planner (8-bit state, 6 actions) is trivially
                    // fast and caching only on carried-changes caused the agent to hold a
                    // stale cluster target while nearby gold went uncollected.
                    goap_action(&bridge.sim)
                }
            }
        };
        bridge.last_action = action;
        bridge.action_counts[action as usize] += 1;
        // Track movement for the ONNX option-commitment logic (no-op for other modes).
        let pos_before = bridge.sim.agents.first().map(|a| a.pos);
        let (rew, done) = bridge.sim.step(action);
        bridge.onnx_moved = bridge.sim.agents.first().map(|a| a.pos) != pos_before;
        bridge.onnx_option_ticks += 1;
        bridge.episode_reward += rew;
        if done {
            bridge.game_over = true;
            break;
        }
    }
}

/// Pick the fixed region whose nearest gold piece is closest to the agent.
/// Regions are now stable 3×3 map cells, so the BT must scan them to find the
/// one holding the nearest gold (it is no longer always region 0).
fn bt_nearest_cluster(sim: &atb::engine::SimCore) -> u32 {
    use atb::engine::clusters::{chebyshev, find_clusters};

    let gold: Vec<_> = sim.items.iter()
        .filter(|i| i.kind == ItemKind::Gold)
        .map(|i| i.pos)
        .collect();

    if gold.is_empty() { return ACTION_WAIT; }

    let agent_pos = sim.agents[0].pos;
    let (gw, gh) = (sim.grid.width as i32, sim.grid.height as i32);
    let clusters = find_clusters(&gold, gw, gh);

    clusters.iter().enumerate()
        .filter_map(|(k, c)| {
            let g = c.as_ref()?.nearest_gold(agent_pos)?;
            Some((chebyshev(agent_pos, g), k as u32))
        })
        .min()
        .map(|(_, k)| k)
        .unwrap_or(ACTION_WAIT)
}

fn goap_action(sim: &atb::engine::SimCore) -> u32 {
    let agent = &sim.agents[0];
    let items = &sim.items;

    const NEAR_SQ: i32 = 10 * 10;

    let gold_nearby = items.iter()
        .filter(|i| i.kind == ItemKind::Gold)
        .any(|i| agent.pos.dist_sq(i.pos) <= NEAR_SQ);

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

    // Combat-era world bits (enemy-nearby, low-health) are gone in the single-agent
    // gold rush, so only the gold/inventory/base bits are set here. The GOAP planner
    // still defines the combat bits/actions (it's a reusable library); they simply
    // never fire without those preconditions.
    let ws = {
        let mut bits = 0u64;
        if agent.gold_carried > 0                    { bits |= goap::BIT_HAS_GOLD; }
        if agent.gold_carried >= AGENT_MAX_GOLD      { bits |= goap::BIT_INVENTORY_FULL; }
        if agent.gold_carried >= AGENT_MAX_GOLD / 2  { bits |= goap::BIT_INVENTORY_HALF; }
        if agent.pos == agent.base_pos               { bits |= goap::BIT_ON_OWN_BASE; }
        if gold_nearby                               { bits |= goap::BIT_GOLD_NEARBY; }
        if dist_to_base < dist_to_gold               { bits |= goap::BIT_BASE_CLOSER; }
        goap::WorldState(bits)
    };

    // Goal: have collected gold and returned to base.
    let goal = goap::GoalState(goap::BIT_INVENTORY_HALF | goap::BIT_ON_OWN_BASE);

    let first_step = goap::plan(ws, goal, goap::ACTIONS, goap::PlanConfig::default())
        .ok()
        .and_then(|r| r.steps.into_iter().next());

    match first_step.as_deref() {
        Some(goap::ACT_NAVIGATE_TO_GOLD) | Some(goap::ACT_COLLECT_GOLD) => bt_nearest_cluster(sim),
        Some(goap::ACT_NAVIGATE_TO_BASE) | Some(goap::ACT_DROP_GOLD) => ACTION_NAVIGATE_TO_BASE,
        _ => ACTION_WAIT,
    }
}

fn item_color(kind: ItemKind) -> Color {
    use crate::style::color::{ITEM_GOLD, ITEM_MULT, ITEM_SPEED};
    match kind {
        ItemKind::Gold       => ITEM_GOLD,
        ItemKind::Speed      => ITEM_SPEED,
        ItemKind::Multiplier => ITEM_MULT,
    }
}
