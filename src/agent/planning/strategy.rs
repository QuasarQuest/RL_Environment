// src/agent/planning/strategy.rs
//
// All decision strategies in one file.

use serde::Deserialize;
use crate::agent::action::Action;
use crate::agent::components::GridPos;
use crate::agent::observation::Observation;
use crate::agent::planning::planner::PathPlanner;
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

// ── Shared move helper ────────────────────────────────────────────────────────

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

// ── BT ────────────────────────────────────────────────────────────────────────

pub struct BtStrategy {
    current_goal: Option<GridPos>,
    stuck_ticks:  u8,
}

impl BtStrategy {
    pub fn new() -> Self { Self { current_goal: None, stuck_ticks: 0 } }

    fn should_return_early(obs: &Observation) -> bool {
        if obs.gold_carried.is_empty() { return false; }
        if obs.gold_carried.is_full()  { return true; }
        let base_dist = obs.nearest_own_base().map(|b| obs.pos.dist_sq(b)).unwrap_or(i32::MAX);
        let gold_dist = obs.nearest_item(ItemKind::Gold).map(|g| obs.pos.dist_sq(g)).unwrap_or(i32::MAX);
        let half_full = obs.gold_carried.0 >= crate::config::AGENT_MAX_GOLD / 2;
        half_full && base_dist < gold_dist
    }
}

impl DecisionStrategy for BtStrategy {
    fn name(&self) -> &'static str { "BehaviorTree" }

    fn decide(&mut self, obs: &Observation, planner: &mut impl PathPlanner) -> Action {
        planner.update(obs.pos, obs.walkability_fn());

        let on_base = matches!(obs.grid_tile(obs.pos), Some(Tile::Base(t)) if t == obs.team.0);
        if on_base && !obs.gold_carried.is_empty() {
            self.current_goal = None;
            planner.reset();
            return Action::Drop;
        }

        if Self::should_return_early(obs) {
            let base = match obs.nearest_own_base() { Some(b) => b, None => return Action::Wait };
            if self.current_goal != Some(base) {
                planner.set_goal(obs.pos, base, obs.walkability_fn());
                self.current_goal = Some(base);
            }
            return match try_move(planner, obs, &mut self.stuck_ticks) {
                Ok(a)   => a,
                Err(()) => { planner.reset(); self.current_goal = None; Action::Wait }
            };
        }

        if let Some(gold) = obs.nearest_item(ItemKind::Gold) {
            if self.current_goal != Some(gold) {
                planner.set_goal(obs.pos, gold, obs.walkability_fn());
                self.current_goal = Some(gold);
            }
            return match try_move(planner, obs, &mut self.stuck_ticks) {
                Ok(a)   => a,
                Err(()) => { planner.reset(); self.current_goal = None; Action::Wait }
            };
        }

        self.current_goal = None;
        Action::Wait
    }

    fn reset(&mut self) { self.current_goal = None; self.stuck_ticks = 0; }
}

// ── Random ────────────────────────────────────────────────────────────────────

pub struct RandomStrategy;

impl DecisionStrategy for RandomStrategy {
    fn name(&self) -> &'static str { "Random" }

    fn decide(&mut self, obs: &Observation, _planner: &mut impl PathPlanner) -> Action {
        use crate::agent::action::Dir;
        let on_base = matches!(obs.grid_tile(obs.pos), Some(Tile::Base(t)) if t == obs.team.0);
        if on_base && !obs.gold_carried.is_empty() { return Action::Drop; }
        let dirs = Dir::all();
        Action::Move(dirs[rand::random_range(0..dirs.len())])
    }
}

// ── GOAP ──────────────────────────────────────────────────────────────────────
//
// The GOAP planner works in abstract action space. Two action types exist:
//
//   Navigation actions (navigate_to_gold, navigate_to_base, flee):
//     → GoapStrategy maps these to a nav target and delegates to PathPlanner.
//     → The agent keeps moving toward the target until it arrives.
//     → On arrival the world state bits update (GOLD_NEARBY / ON_OWN_BASE set),
//       triggering a replan which picks the next step (collect / drop).
//
//   Immediate actions (collect_gold, drop_gold):
//     → Executed in place as Bevy Actions (Pickup / Drop).
//     → After execution world state changes, triggering replan.
//
// Goal alternates based on current state:
//   no gold  → GoalState(BIT_HAS_GOLD | BIT_ON_OWN_BASE)  (full cycle)
//   has gold → GoalState(BIT_ON_OWN_BASE) — skip navigate_to_gold, go deliver

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

            // Goal: complete full collect→deliver cycle.
            // Carrying gold → skip straight to deliver goal.
            let goal = if ws.0 & BIT_HAS_GOLD != 0 {
                GoalState(BIT_ON_OWN_BASE)                  // just need to reach base & drop
            } else {
                GoalState(BIT_HAS_GOLD | BIT_ON_OWN_BASE)  // full cycle: get gold then deliver
            };

            self.cached_plan = match goap::plan(ws, goal, ACTIONS, self.config) {
                Ok(r)                       => r.steps,
                Err(PlanError::NoPathFound) => vec![goap::ACT_WAIT],
                Err(_)                      => vec![],
            };
        }

        self.cached_plan.first().copied()
    }

    /// Map a GOAP action name to a navigation target.
    /// Returns None for immediate actions (collect, drop) — handled below.
    fn nav_target(step: &'static str, obs: &Observation) -> Option<GridPos> {
        if step == goap::ACT_NAVIGATE_TO_GOLD {
            obs.nearest_item(ItemKind::Gold)
        } else if step == goap::ACT_NAVIGATE_TO_BASE || step == goap::ACT_FLEE {
            obs.nearest_own_base()
        } else {
            None // collect_gold, drop_gold, wait — not navigation
        }
    }
}

impl DecisionStrategy for GoapStrategy {
    fn name(&self) -> &'static str { "GOAP" }

    fn decide(&mut self, obs: &Observation, planner: &mut impl PathPlanner) -> Action {
        planner.update(obs.pos, obs.walkability_fn());

        // Immediate: drop if on base with gold (world state will update → replan)
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

        // Immediate actions — execute in place, don't navigate.
        if step == goap::ACT_COLLECT_GOLD {
            // Already at gold (GOLD_NEARBY was set) — pickup handled by
            // the pickup system when agent is on the tile. Just wait here;
            // world state will change on next tick triggering replan.
            self.last_ws = None;
            return Action::Wait;
        }
        if step == goap::ACT_DROP_GOLD {
            self.last_ws = None;
            return Action::Drop;
        }

        // Navigation actions — set goal and move.
        let Some(goal_pos) = Self::nav_target(step, obs) else {
            return Action::Wait;
        };

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
    Random,
    Goap,
}