// src/agent/planning/strategy.rs

use serde::Deserialize;
use crate::agent::action::{Action, Dir};
use crate::agent::components::GridPos;
use crate::agent::observation::Observation;
use crate::agent::planning::planner::{AStarPlanner, DStarPlanner, PathPlanner};
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

pub trait DecisionStrategy: Send + Sync {
    fn name(&self) -> &'static str;
    fn decide(&mut self, obs: &Observation, planner: &mut impl PathPlanner) -> Action;
    fn reset(&mut self) {}
}

// ── Shared nav helper ─────────────────────────────────────────────────────────

pub fn try_move(
    planner:     &mut dyn PathPlanner,
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

fn adjacent_enemy_dir(obs: &Observation) -> Option<Dir> {
    Dir::all().iter().copied().find(|&dir| {
        let (dx, dy) = dir.delta();
        let check    = GridPos::new(obs.pos.x + dx, obs.pos.y + dy);
        obs.visible_agents().iter()
            .any(|a| a.is_enemy(obs.team) && a.pos == check)
    })
}

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

// ── BT intent ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum NavTarget { Gold, Base, Health, FleeFrom(GridPos) }

#[derive(Clone, Copy, Debug)]
enum BtOutput {
    Direct(Action),
    Navigate(NavTarget),
}

// ── Nav state ─────────────────────────────────────────────────────────────────

struct NavState {
    current_goal: Option<GridPos>,
    stuck_ticks:  u8,
}

impl NavState {
    fn new()        -> Self { Self { current_goal: None, stuck_ticks: 0 } }
    fn reset(&mut self)     { self.current_goal = None; self.stuck_ticks = 0; }
}

fn resolve_nav(
    target:  NavTarget,
    obs:     &Observation,
    planner: &mut dyn PathPlanner,
    nav:     &mut NavState,
) -> Action {
    let walkable = obs.walkability_fn();

    let goal = match target {
        NavTarget::Gold   => obs.nearest_item(ItemKind::Gold),
        NavTarget::Base   => obs.nearest_own_base(),
        NavTarget::Health => obs.nearest_item(ItemKind::Health),
        NavTarget::FleeFrom(enemy_pos) => {
            let dx   = (obs.pos.x - enemy_pos.x).signum();
            let dy   = (obs.pos.y - enemy_pos.y).signum();
            let flee = GridPos::new(obs.pos.x + dx, obs.pos.y + dy);
            if obs.is_walkable(flee) { Some(flee) } else { None }
        }
    };

    let Some(goal) = goal else { nav.reset(); return Action::Wait; };

    if nav.current_goal != Some(goal) {
        planner.set_goal(obs.pos, goal, &walkable);
        nav.current_goal = Some(goal);
    }
    planner.update(obs.pos, &walkable);

    match try_move(planner, obs, &mut nav.stuck_ticks) {
        Ok(action) => action,
        Err(())    => { nav.reset(); planner.reset(); Action::Wait }
    }
}

// ── BT tree ───────────────────────────────────────────────────────────────────
//
// Opportunistic: fight when it makes sense, collect gold, survive.
//
// Priority:
//   1. Critical survival  — 1 heart left → seek health immediately
//   2. Ranged attack      — has ammo + clear shot (always worth shooting)
//   3. Melee attack       — adjacent enemy (always engage in melee)
//   4. Drop gold on base
//   5. Deliver            — full OR base closer than gold
//   6. Collect gold
//   7. Opportunistic heal — hurt + health nearby + no enemies around
//   8. Flee               — hurt + enemy nearby + no health available
//   9. Roam

fn build_bt_tree() -> Box<dyn BtNode<Observation, BtOutput>> {
    selector(vec![

        // 1. Critical survival — 1 heart, go heal regardless of anything
        sequence(vec![
            cond(|o: &Observation| o.hearts.0 <= 1 && o.nearest_item(ItemKind::Health).is_some()),
            leaf(|_: &Observation, out: &mut Option<BtOutput>| {
                *out = Some(BtOutput::Navigate(NavTarget::Health));
                Status::Running
            }),
        ]),

        // 2. Ranged — has ammo + clear shot exists
        sequence(vec![
            cond(|o: &Observation| o.has_ammo() && ranged_enemy_dir(o).is_some()),
            leaf(|o: &Observation, out: &mut Option<BtOutput>| {
                match ranged_enemy_dir(o) {
                    Some(d) => { *out = Some(BtOutput::Direct(Action::RangedAttack(d))); Status::Running }
                    None    => Status::Failure,
                }
            }),
        ]),

        // 3. Melee — adjacent enemy, always engage
        sequence(vec![
            cond(|o: &Observation| adjacent_enemy_dir(o).is_some()),
            leaf(|o: &Observation, out: &mut Option<BtOutput>| {
                match adjacent_enemy_dir(o) {
                    Some(d) => { *out = Some(BtOutput::Direct(Action::Attack(d))); Status::Running }
                    None    => Status::Failure,
                }
            }),
        ]),

        // 4. Drop gold on own base
        sequence(vec![
            cond(|o: &Observation| {
                !o.gold_carried.is_empty()
                    && matches!(o.grid_tile(o.pos), Some(Tile::Base(t)) if t == o.team.0)
            }),
            leaf(|_: &Observation, out: &mut Option<BtOutput>| {
                *out = Some(BtOutput::Direct(Action::Drop));
                Status::Running
            }),
        ]),

        // 5. Deliver — full OR base closer than nearest gold
        sequence(vec![
            cond(|o: &Observation| {
                if o.gold_carried.is_empty() { return false; }
                if o.gold_carried.is_full()  { return true; }
                let bd = o.nearest_own_base().map(|b| o.pos.dist_sq(b)).unwrap_or(i32::MAX);
                let gd = o.nearest_item(ItemKind::Gold).map(|g| o.pos.dist_sq(g)).unwrap_or(i32::MAX);
                bd < gd
            }),
            leaf(|_: &Observation, out: &mut Option<BtOutput>| {
                *out = Some(BtOutput::Navigate(NavTarget::Base));
                Status::Running
            }),
        ]),

        // 6. Collect gold
        sequence(vec![
            cond(|o: &Observation| o.nearest_item(ItemKind::Gold).is_some()),
            leaf(|_: &Observation, out: &mut Option<BtOutput>| {
                *out = Some(BtOutput::Navigate(NavTarget::Gold));
                Status::Running
            }),
        ]),

        // 7. Opportunistic heal — hurt, health nearby, no enemies
        sequence(vec![
            cond(|o: &Observation| {
                o.needs_health()
                    && o.nearest_item(ItemKind::Health).is_some()
                    && o.nearest_enemy().is_none()
            }),
            leaf(|_: &Observation, out: &mut Option<BtOutput>| {
                *out = Some(BtOutput::Navigate(NavTarget::Health));
                Status::Running
            }),
        ]),

        // 8. Flee — hurt + enemy nearby + no health available
        sequence(vec![
            cond(|o: &Observation| {
                o.needs_health()
                    && o.nearest_enemy().is_some()
                    && o.nearest_item(ItemKind::Health).is_none()
            }),
            leaf(|o: &Observation, out: &mut Option<BtOutput>| {
                let Some(e) = o.nearest_enemy() else { return Status::Failure };
                *out = Some(BtOutput::Navigate(NavTarget::FleeFrom(e.pos)));
                Status::Running
            }),
        ]),

        // 9. Roam
        leaf(|o: &Observation, out: &mut Option<BtOutput>| {
            *out = Some(BtOutput::Direct(roam(o)));
            Status::Running
        }),
    ])
}

// ── BtStrategy ────────────────────────────────────────────────────────────────

pub struct BtStrategy {
    tree:    Box<dyn BtNode<Observation, BtOutput>>,
    planner: Box<dyn PathPlanner>,
    nav:     NavState,
}

impl BtStrategy {
    pub fn new_astar() -> Self { Self::with(Box::new(AStarPlanner::new())) }
    pub fn new_dstar() -> Self { Self::with(Box::new(DStarPlanner::new())) }

    fn with(planner: Box<dyn PathPlanner>) -> Self {
        Self { tree: build_bt_tree(), planner, nav: NavState::new() }
    }
}

impl DecisionStrategy for BtStrategy {
    fn name(&self) -> &'static str { "BT" }

    fn decide(&mut self, obs: &Observation, _planner: &mut impl PathPlanner) -> Action {
        let mut out = None;
        self.tree.tick(obs, &mut out);
        match out.unwrap_or(BtOutput::Direct(Action::Wait)) {
            BtOutput::Direct(action)   => action,
            BtOutput::Navigate(target) =>
                resolve_nav(target, obs, self.planner.as_mut(), &mut self.nav),
        }
    }

    fn reset(&mut self) { self.nav.reset(); self.planner.reset(); }
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

impl DecisionStrategy for FsmStrategy {
    fn name(&self) -> &'static str { "FSM" }

    fn decide(&mut self, obs: &Observation, planner: &mut impl PathPlanner) -> Action {
        let walkable = obs.walkability_fn();
        planner.update(obs.pos, &walkable);

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
                planner.set_goal(obs.pos, base, &walkable);
                self.fsm.transition(FsmPhase::Active { phase: CollectPhase::Returning, target: base });
            }
            return match try_move(planner, obs, &mut self.stuck_ticks) {
                Ok(a)   => a,
                Err(()) => { planner.reset(); self.fsm.transition(FsmPhase::Idle); Action::Wait }
            };
        }

        if let Some(gold) = obs.nearest_item(ItemKind::Gold) {
            if self.current_target() != Some(gold) || planner.next_step().is_none() {
                planner.set_goal(obs.pos, gold, &walkable);
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

impl DecisionStrategy for RandomStrategy {
    fn name(&self) -> &'static str { "Random" }

    fn decide(&mut self, obs: &Observation, _planner: &mut impl PathPlanner) -> Action {
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

impl DecisionStrategy for GoapStrategy {
    fn name(&self) -> &'static str { "GOAP" }

    fn decide(&mut self, obs: &Observation, planner: &mut impl PathPlanner) -> Action {
        let walkable = obs.walkability_fn();
        planner.update(obs.pos, &walkable);

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
            planner.set_goal(obs.pos, goal_pos, &walkable);
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
    Random,
    Goap,
}