// src/agent/planning/strategy.rs

use serde::Deserialize;
use crate::agent::action::{Action, Dir};
use crate::agent::components::GridPos;
use crate::agent::observation::Observation;
use crate::agent::planning::planner::PathPlanner;
use crate::algorithm::behavior_planning::behavior_tree::{BtNode, Status, selector, sequence, cond, leaf};
use crate::algorithm::behavior_planning::fsm::{Fsm, FsmState};
use crate::algorithm::behavior_planning::goap::{
    self, GoalState, PlanConfig, PlanError, WorldState,
    BIT_HAS_GOLD, BIT_ON_OWN_BASE, ACTIONS,
};
use crate::algorithm::path_planning::graph_utils::dir_to;
use crate::item::ItemKind;
use crate::world::tile::Tile;

// ── Trait ─────────────────────────────────────────────────────────────────────
//
// Generic over P so BT strategies can name the concrete planner type,
// while FSM / GOAP / Random implement it for any P with a blanket impl.

pub trait DecisionStrategy<P: PathPlanner>: Send + Sync {
    fn name(&self) -> &'static str;
    fn decide(&mut self, obs: &Observation, planner: &mut P) -> Action;
    fn reset(&mut self) {}
}

// ── Shared nav helper ─────────────────────────────────────────────────────────

pub fn try_move(
    planner:     &mut impl PathPlanner,
    obs:         &Observation,
    stuck_ticks: &mut u8,
) -> Result<Action, ()> {
    let Some(next) = planner.next_step() else { return Ok(Action::Wait); };
    if !obs.is_walkable(next) {
        *stuck_ticks += 1;
        if *stuck_ticks >= 3 { *stuck_ticks = 0; return Err(()); }
        return Ok(Action::Wait);
    }
    *stuck_ticks = 0;
    Ok(dir_to(obs.pos, next).map(Action::Move).unwrap_or(Action::Wait))
}

// ── Combat helpers ────────────────────────────────────────────────────────────

/// Dir toward the nearest adjacent (Chebyshev 1) enemy, if any.
fn adjacent_enemy_dir(obs: &Observation) -> Option<Dir> {
    Dir::all().iter().copied().find(|&dir| {
        let (dx, dy) = dir.delta();
        let check    = GridPos::new(obs.pos.x + dx, obs.pos.y + dy);
        obs.visible_agents().iter()
            .any(|a| a.is_enemy(obs.team) && a.pos == check)
    })
}

/// Dir toward the nearest enemy within RANGED_RANGE along a ray, blocked by
/// obstacles. Returns None if no clear shot exists.
fn ranged_enemy_dir(obs: &Observation) -> Option<Dir> {
    let mut best_dir:  Option<Dir> = None;
    let mut best_dist: i32         = i32::MAX;
    for &dir in Dir::all() {
        let (dx, dy) = dir.delta();
        for dist in 1..=crate::config::RANGED_RANGE {
            let check = GridPos::new(obs.pos.x + dx * dist, obs.pos.y + dy * dist);
            if obs.grid_tile(check) == Some(Tile::Obstacle) { break; }
            if obs.visible_agents().iter().any(|a| a.is_enemy(obs.team) && a.pos == check) {
                if dist < best_dist { best_dist = dist; best_dir = Some(dir); }
                break;
            }
        }
    }
    best_dir
}

/// True if we have strictly more hearts than the nearest visible enemy.
fn hp_advantage(obs: &Observation) -> bool {
    obs.nearest_enemy().map(|e| obs.hearts.0 > e.hearts.0).unwrap_or(false)
}

/// Pick a random walkable direction, or Wait if none available.
fn roam(obs: &Observation) -> Action {
    let dirs: Vec<Dir> = Dir::all().iter().copied()
        .filter(|&d| {
            let (dx, dy) = d.delta();
            obs.is_walkable(GridPos::new(obs.pos.x + dx, obs.pos.y + dy))
        })
        .collect();
    if dirs.is_empty() { Action::Wait }
    else { Action::Move(dirs[rand::random_range(0..dirs.len())]) }
}

// ── BT context ────────────────────────────────────────────────────────────────
//
// Bundles the observation and mutable planner / nav state for leaf closures.
// Raw pointers are used so the tree (which stores Fn closures, not FnMut)
// can mutate state owned by the calling stack frame.
//
// SAFETY: BtCtx is created, ticked, and dropped within a single synchronous
// call to decide(). The raw pointers never escape the tick call.

struct NavState {
    current_goal: Option<GridPos>,
    stuck_ticks:  u8,
}

impl NavState {
    fn new()   -> Self { Self { current_goal: None, stuck_ticks: 0 } }
    fn reset(&mut self) { self.current_goal = None; self.stuck_ticks = 0; }
}

pub struct BtCtx<'a, P: PathPlanner> {
    obs:     &'a Observation,
    planner: *mut P,
    nav:     *mut NavState,
}

unsafe impl<P: PathPlanner> Send for BtCtx<'_, P> {}
unsafe impl<P: PathPlanner> Sync for BtCtx<'_, P> {}

fn nav_leaf<P: PathPlanner>(
    goal: GridPos,
    ctx:  &BtCtx<P>,
    out:  &mut Option<Action>,
) -> Status {
    let planner = unsafe { &mut *ctx.planner };
    let nav     = unsafe { &mut *ctx.nav };
    if nav.current_goal != Some(goal) {
        planner.set_goal(ctx.obs.pos, goal, ctx.obs.walkability_fn());
        nav.current_goal = Some(goal);
    }
    planner.update(ctx.obs.pos, ctx.obs.walkability_fn());
    match try_move(planner, ctx.obs, &mut nav.stuck_ticks) {
        Ok(action) => { *out = Some(action); Status::Running }
        Err(())    => { nav.reset(); planner.reset(); Status::Failure }
    }
}

// ── Cautious BT tree ──────────────────────────────────────────────────────────
//
// Priority (high → low):
//   Survive → Flee → Ranged(hp-advantage) → Melee(hp-advantage)
//   → Drop → Deliver → Collect → Roam

fn build_cautious_tree<P: PathPlanner + 'static>()
    -> Box<dyn BtNode<BtCtx<'static, P>, Action>>
{
    selector(vec![
        // 1. Survive — low health + health item on map
        sequence(vec![
            cond(|c: &BtCtx<P>| c.obs.needs_health() && c.obs.nearest_item(ItemKind::Health).is_some()),
            leaf(|c: &BtCtx<P>, out: &mut Option<Action>| {
                let Some(hp) = c.obs.nearest_item(ItemKind::Health) else { return Status::Failure };
                nav_leaf(hp, c, out)
            }),
        ]),
        // 2. Flee — low health + no health available + enemy nearby
        sequence(vec![
            cond(|c: &BtCtx<P>| {
                c.obs.needs_health()
                    && c.obs.nearest_item(ItemKind::Health).is_none()
                    && c.obs.nearest_enemy().is_some()
            }),
            leaf(|c: &BtCtx<P>, out: &mut Option<Action>| {
                let Some(e) = c.obs.nearest_enemy() else { return Status::Failure };
                let dx   = (c.obs.pos.x - e.pos.x).signum();
                let dy   = (c.obs.pos.y - e.pos.y).signum();
                let flee = GridPos::new(c.obs.pos.x + dx, c.obs.pos.y + dy);
                if c.obs.is_walkable(flee) {
                    *out = dir_to(c.obs.pos, flee).map(Action::Move);
                    if out.is_some() { Status::Running } else { Status::Failure }
                } else {
                    Status::Failure
                }
            }),
        ]),
        // 3. Ranged — ammo available + HP advantage + clear shot
        sequence(vec![
            cond(|c: &BtCtx<P>| c.obs.has_ammo() && hp_advantage(c.obs) && ranged_enemy_dir(c.obs).is_some()),
            leaf(|c: &BtCtx<P>, out: &mut Option<Action>| {
                match ranged_enemy_dir(c.obs) {
                    Some(d) => { *out = Some(Action::RangedAttack(d)); Status::Running }
                    None    => Status::Failure,
                }
            }),
        ]),
        // 4. Melee — adjacent enemy + HP advantage
        sequence(vec![
            cond(|c: &BtCtx<P>| hp_advantage(c.obs) && adjacent_enemy_dir(c.obs).is_some()),
            leaf(|c: &BtCtx<P>, out: &mut Option<Action>| {
                match adjacent_enemy_dir(c.obs) {
                    Some(d) => { *out = Some(Action::Attack(d)); Status::Running }
                    None    => Status::Failure,
                }
            }),
        ]),
        // 5. Drop gold if standing on own base
        sequence(vec![
            cond(|c: &BtCtx<P>| {
                !c.obs.gold_carried.is_empty()
                    && matches!(c.obs.grid_tile(c.obs.pos), Some(Tile::Base(t)) if t == c.obs.team.0)
            }),
            leaf(|_: &BtCtx<P>, out: &mut Option<Action>| {
                *out = Some(Action::Drop); Status::Running
            }),
        ]),
        // 6. Deliver — full OR base closer than nearest gold
        sequence(vec![
            cond(|c: &BtCtx<P>| {
                if c.obs.gold_carried.is_empty() { return false; }
                if c.obs.gold_carried.is_full()  { return true; }
                let bd = c.obs.nearest_own_base().map(|b| c.obs.pos.dist_sq(b)).unwrap_or(i32::MAX);
                let gd = c.obs.nearest_item(ItemKind::Gold).map(|g| c.obs.pos.dist_sq(g)).unwrap_or(i32::MAX);
                bd < gd
            }),
            leaf(|c: &BtCtx<P>, out: &mut Option<Action>| {
                let Some(base) = c.obs.nearest_own_base() else { return Status::Failure };
                nav_leaf(base, c, out)
            }),
        ]),
        // 7. Collect nearest gold
        sequence(vec![
            cond(|c: &BtCtx<P>| c.obs.nearest_item(ItemKind::Gold).is_some()),
            leaf(|c: &BtCtx<P>, out: &mut Option<Action>| {
                let Some(gold) = c.obs.nearest_item(ItemKind::Gold) else { return Status::Failure };
                nav_leaf(gold, c, out)
            }),
        ]),
        // 8. Roam
        leaf(|c: &BtCtx<P>, out: &mut Option<Action>| {
            *out = Some(roam(c.obs)); Status::Running
        }),
    ])
}

// ── Aggressive BT tree ────────────────────────────────────────────────────────
//
// Priority (high → low):
//   Melee(any) → Ranged(any) → Drop → Deliver(full) → Collect
//   → Survive(last-resort) → Roam

fn build_aggressive_tree<P: PathPlanner + 'static>()
    -> Box<dyn BtNode<BtCtx<'static, P>, Action>>
{
    selector(vec![
        // 1. Melee — strike any adjacent enemy unconditionally
        sequence(vec![
            cond(|c: &BtCtx<P>| adjacent_enemy_dir(c.obs).is_some()),
            leaf(|c: &BtCtx<P>, out: &mut Option<Action>| {
                match adjacent_enemy_dir(c.obs) {
                    Some(d) => { *out = Some(Action::Attack(d)); Status::Running }
                    None    => Status::Failure,
                }
            }),
        ]),
        // 2. Ranged — shoot if ammo and clear shot, regardless of HP
        sequence(vec![
            cond(|c: &BtCtx<P>| c.obs.has_ammo() && ranged_enemy_dir(c.obs).is_some()),
            leaf(|c: &BtCtx<P>, out: &mut Option<Action>| {
                match ranged_enemy_dir(c.obs) {
                    Some(d) => { *out = Some(Action::RangedAttack(d)); Status::Running }
                    None    => Status::Failure,
                }
            }),
        ]),
        // 3. Drop gold on own base
        sequence(vec![
            cond(|c: &BtCtx<P>| {
                !c.obs.gold_carried.is_empty()
                    && matches!(c.obs.grid_tile(c.obs.pos), Some(Tile::Base(t)) if t == c.obs.team.0)
            }),
            leaf(|_: &BtCtx<P>, out: &mut Option<Action>| {
                *out = Some(Action::Drop); Status::Running
            }),
        ]),
        // 4. Deliver when full
        sequence(vec![
            cond(|c: &BtCtx<P>| c.obs.gold_carried.is_full()),
            leaf(|c: &BtCtx<P>, out: &mut Option<Action>| {
                let Some(base) = c.obs.nearest_own_base() else { return Status::Failure };
                nav_leaf(base, c, out)
            }),
        ]),
        // 5. Collect gold
        sequence(vec![
            cond(|c: &BtCtx<P>| c.obs.nearest_item(ItemKind::Gold).is_some()),
            leaf(|c: &BtCtx<P>, out: &mut Option<Action>| {
                let Some(gold) = c.obs.nearest_item(ItemKind::Gold) else { return Status::Failure };
                nav_leaf(gold, c, out)
            }),
        ]),
        // 6. Survive — only when no enemies present (last resort)
        sequence(vec![
            cond(|c: &BtCtx<P>| {
                c.obs.needs_health()
                    && c.obs.nearest_enemy().is_none()
                    && c.obs.nearest_item(ItemKind::Health).is_some()
            }),
            leaf(|c: &BtCtx<P>, out: &mut Option<Action>| {
                let Some(hp) = c.obs.nearest_item(ItemKind::Health) else { return Status::Failure };
                nav_leaf(hp, c, out)
            }),
        ]),
        // 7. Roam
        leaf(|c: &BtCtx<P>, out: &mut Option<Action>| {
            *out = Some(roam(c.obs)); Status::Running
        }),
    ])
}

// ── BtCautiousStrategy ────────────────────────────────────────────────────────

pub struct BtCautiousStrategy<P: PathPlanner + 'static> {
    tree: Box<dyn BtNode<BtCtx<'static, P>, Action>>,
    nav:  NavState,
}

impl<P: PathPlanner + 'static> BtCautiousStrategy<P> {
    pub fn new() -> Self {
        Self { tree: build_cautious_tree::<P>(), nav: NavState::new() }
    }
}

impl<P: PathPlanner + 'static> DecisionStrategy<P> for BtCautiousStrategy<P> {
    fn name(&self) -> &'static str { "BT-Cautious" }

    fn decide(&mut self, obs: &Observation, planner: &mut P) -> Action {
        let mut ctx = BtCtx {
            obs,
            planner: planner as *mut P,
            nav:     &mut self.nav as *mut NavState,
        };
        // SAFETY: tree tick is synchronous; ctx outlives the tick call.
        let ctx_s: &mut BtCtx<'static, P> = unsafe { std::mem::transmute(&mut ctx) };
        let mut out = None;
        self.tree.tick(ctx_s, &mut out);
        out.unwrap_or(Action::Wait)
    }

    fn reset(&mut self) { self.nav.reset(); }
}

// ── BtAggressiveStrategy ──────────────────────────────────────────────────────

pub struct BtAggressiveStrategy<P: PathPlanner + 'static> {
    tree: Box<dyn BtNode<BtCtx<'static, P>, Action>>,
    nav:  NavState,
}

impl<P: PathPlanner + 'static> BtAggressiveStrategy<P> {
    pub fn new() -> Self {
        Self { tree: build_aggressive_tree::<P>(), nav: NavState::new() }
    }
}

impl<P: PathPlanner + 'static> DecisionStrategy<P> for BtAggressiveStrategy<P> {
    fn name(&self) -> &'static str { "BT-Aggressive" }

    fn decide(&mut self, obs: &Observation, planner: &mut P) -> Action {
        let mut ctx = BtCtx {
            obs,
            planner: planner as *mut P,
            nav:     &mut self.nav as *mut NavState,
        };
        let ctx_s: &mut BtCtx<'static, P> = unsafe { std::mem::transmute(&mut ctx) };
        let mut out = None;
        self.tree.tick(ctx_s, &mut out);
        out.unwrap_or(Action::Wait)
    }

    fn reset(&mut self) { self.nav.reset(); }
}

// ── FSM ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectPhase { Collecting, Returning, Dropping }

pub enum FsmPhase {
    Idle,
    Active { phase: CollectPhase, target: GridPos },
}

impl FsmState for FsmPhase {
    fn name(&self) -> &'static str {
        match self {
            Self::Idle                                            => "Idle",
            Self::Active { phase: CollectPhase::Collecting, .. } => "Collecting",
            Self::Active { phase: CollectPhase::Returning,  .. } => "Returning",
            Self::Active { phase: CollectPhase::Dropping,   .. } => "Dropping",
        }
    }
    fn idle() -> Self { Self::Idle }
}

pub struct FsmStrategy {
    fsm:         Fsm<FsmPhase>,
    stuck_ticks: u8,
}

impl FsmStrategy {
    pub fn new() -> Self { Self { fsm: Fsm::new(FsmPhase::Idle), stuck_ticks: 0 } }

    fn current_target(&self) -> Option<GridPos> {
        match self.fsm.state() {
            FsmPhase::Active { target, .. } => Some(*target),
            FsmPhase::Idle                  => None,
        }
    }
}

impl<P: PathPlanner> DecisionStrategy<P> for FsmStrategy {
    fn name(&self) -> &'static str { "FSM" }

    fn decide(&mut self, obs: &Observation, planner: &mut P) -> Action {
        planner.update(obs.pos, obs.walkability_fn());

        let on_base = matches!(obs.grid_tile(obs.pos), Some(Tile::Base(t)) if t == obs.team.0);
        if on_base && !obs.gold_carried.is_empty() {
            self.fsm.transition(FsmPhase::Active { phase: CollectPhase::Dropping, target: obs.pos });
            return Action::Drop;
        }
        if let FsmPhase::Active { phase: CollectPhase::Dropping, .. } = self.fsm.state() {
            self.fsm.transition(FsmPhase::Idle);
            planner.reset();
        }

        if obs.gold_carried.is_full() {
            let base = match obs.nearest_own_base() { Some(b) => b, None => return Action::Wait };
            if self.current_target() != Some(base) || planner.next_step().is_none() {
                planner.set_goal(obs.pos, base, obs.walkability_fn());
                self.fsm.transition(FsmPhase::Active { phase: CollectPhase::Returning, target: base });
            }
            return match try_move(planner, obs, &mut self.stuck_ticks) {
                Ok(a)   => a,
                Err(()) => { planner.reset(); self.fsm.transition(FsmPhase::Idle); Action::Wait }
            };
        }

        if let Some(gold) = obs.nearest_item(ItemKind::Gold) {
            if self.current_target() != Some(gold) || planner.next_step().is_none() {
                planner.set_goal(obs.pos, gold, obs.walkability_fn());
                self.fsm.transition(FsmPhase::Active { phase: CollectPhase::Collecting, target: gold });
            }
            return match try_move(planner, obs, &mut self.stuck_ticks) {
                Ok(a)   => a,
                Err(()) => { planner.reset(); self.fsm.transition(FsmPhase::Idle); Action::Wait }
            };
        }

        self.fsm.transition(FsmPhase::Idle);
        Action::Wait
    }

    fn reset(&mut self) { self.fsm.transition(FsmPhase::Idle); self.stuck_ticks = 0; }
}

// ── Random ────────────────────────────────────────────────────────────────────

pub struct RandomStrategy;

impl<P: PathPlanner> DecisionStrategy<P> for RandomStrategy {
    fn name(&self) -> &'static str { "Random" }

    fn decide(&mut self, obs: &Observation, _planner: &mut P) -> Action {
        let on_base = matches!(obs.grid_tile(obs.pos), Some(Tile::Base(t)) if t == obs.team.0);
        if on_base && !obs.gold_carried.is_empty() { return Action::Drop; }
        let dirs = Dir::all();
        Action::Move(dirs[rand::random_range(0..dirs.len())])
    }
}

// ── GOAP ──────────────────────────────────────────────────────────────────────

pub struct GoapStrategy {
    config:       PlanConfig,
    cached_plan:  Vec<&'static str>,
    last_ws:      Option<WorldState>,
    current_goal: Option<GridPos>,
    stuck_ticks:  u8,
}

impl GoapStrategy {
    pub fn new() -> Self {
        Self {
            config:       PlanConfig::default(),
            cached_plan:  Vec::new(),
            last_ws:      None,
            current_goal: None,
            stuck_ticks:  0,
        }
    }

    fn maybe_replan(&mut self, ws: WorldState) -> Option<&'static str> {
        let stale = self.last_ws.map(|lws| lws != ws).unwrap_or(true)
            || self.cached_plan.is_empty();
        if stale {
            self.last_ws = Some(ws);
            let goal = if ws.0 & BIT_HAS_GOLD != 0 {
                GoalState(BIT_ON_OWN_BASE)
            } else {
                GoalState(BIT_HAS_GOLD | BIT_ON_OWN_BASE)
            };
            self.cached_plan = match goap::plan(ws, goal, ACTIONS, self.config) {
                Ok(r)                       => r.steps,
                Err(PlanError::NoPathFound) => vec![goap::ACT_WAIT],
                Err(_)                      => vec![],
            };
        }
        self.cached_plan.first().copied()
    }

    fn nav_target(step: &'static str, obs: &Observation) -> Option<GridPos> {
        if step == goap::ACT_NAVIGATE_TO_GOLD      { obs.nearest_item(ItemKind::Gold) }
        else if step == goap::ACT_NAVIGATE_TO_BASE
            || step == goap::ACT_FLEE              { obs.nearest_own_base() }
        else                                        { None }
    }
}

impl<P: PathPlanner> DecisionStrategy<P> for GoapStrategy {
    fn name(&self) -> &'static str { "GOAP" }

    fn decide(&mut self, obs: &Observation, planner: &mut P) -> Action {
        planner.update(obs.pos, obs.walkability_fn());

        let on_base = matches!(obs.grid_tile(obs.pos), Some(Tile::Base(t)) if t == obs.team.0);
        if on_base && !obs.gold_carried.is_empty() {
            self.last_ws = None;
            return Action::Drop;
        }

        let ws   = goap::obs_to_world_state(obs);
        let step = match self.maybe_replan(ws) {
            Some(s) => s,
            None    => return Action::Wait,
        };

        if step == goap::ACT_COLLECT_GOLD { self.last_ws = None; return Action::Wait; }
        if step == goap::ACT_DROP_GOLD    { self.last_ws = None; return Action::Drop; }

        let Some(goal_pos) = Self::nav_target(step, obs) else { return Action::Wait; };

        if self.current_goal != Some(goal_pos) {
            planner.set_goal(obs.pos, goal_pos, obs.walkability_fn());
            self.current_goal = Some(goal_pos);
        }

        match try_move(planner, obs, &mut self.stuck_ticks) {
            Ok(action) => action,
            Err(()) => {
                planner.reset();
                self.current_goal = None;
                self.last_ws      = None;
                Action::Wait
            }
        }
    }

    fn reset(&mut self) {
        self.cached_plan.clear();
        self.last_ws      = None;
        self.current_goal = None;
        self.stuck_ticks  = 0;
    }
}

// ── StrategyKind ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub enum StrategyKind {
    Fsm,
    BehaviorTree,
    BehaviorTreeAggressive,
    Random,
    Goap,
}