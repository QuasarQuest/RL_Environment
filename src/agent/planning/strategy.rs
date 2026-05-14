// src/agent/planning/strategy.rs

use serde::Deserialize;
use crate::agent::action::Action;
use crate::agent::components::GridPos;
use crate::agent::observation::Observation;
use crate::agent::planning::planner::PathPlanner;
use crate::algorithm::behavior_planning::fsm::{Fsm, FsmState};
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
    let Some(next) = planner.next_step() else {
        return Ok(Action::Wait);
    };
    if !obs.is_walkable(next) {
        *stuck_ticks += 1;
        if *stuck_ticks >= 3 {
            *stuck_ticks = 0;
            return Err(());
        }
        return Ok(Action::Wait);
    }
    *stuck_ticks = 0;
    Ok(dir_to(obs.pos, next).map(Action::Move).unwrap_or(Action::Wait))
}

// ── FSM strategy ──────────────────────────────────────────────────────────────

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
    pub fn new() -> Self {
        Self { fsm: Fsm::new(FsmPhase::Idle), stuck_ticks: 0 }
    }

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
            self.fsm.transition(FsmPhase::Active {
                phase: CollectPhase::Dropping, target: obs.pos,
            });
            return Action::Drop;
        }

        if let FsmPhase::Active { phase: CollectPhase::Dropping, .. } = self.fsm.state() {
            self.fsm.transition(FsmPhase::Idle);
            planner.reset();
        }

        if !obs.gold_carried.is_empty() {
            let base = match obs.nearest_own_base() { Some(b) => b, None => return Action::Wait };
            if self.current_target() != Some(base) || planner.next_step().is_none() {
                planner.set_goal(obs.pos, base, obs.walkability_fn());
                self.fsm.transition(FsmPhase::Active {
                    phase: CollectPhase::Returning, target: base,
                });
            }
            return match try_move(planner, obs, &mut self.stuck_ticks) {
                Ok(a)   => a,
                Err(()) => { planner.reset(); self.fsm.transition(FsmPhase::Idle); Action::Wait }
            };
        }

        if let Some(gold) = obs.nearest_item(ItemKind::Gold) {
            if self.current_target() != Some(gold) || planner.next_step().is_none() {
                planner.set_goal(obs.pos, gold, obs.walkability_fn());
                self.fsm.transition(FsmPhase::Active {
                    phase: CollectPhase::Collecting, target: gold,
                });
            }
            return match try_move(planner, obs, &mut self.stuck_ticks) {
                Ok(a)   => a,
                Err(()) => { planner.reset(); self.fsm.transition(FsmPhase::Idle); Action::Wait }
            };
        }

        self.fsm.transition(FsmPhase::Idle);
        Action::Wait
    }

    fn reset(&mut self) {
        self.fsm.transition(FsmPhase::Idle);
        self.stuck_ticks = 0;
    }
}

// ── BT strategy ───────────────────────────────────────────────────────────────

pub struct BtStrategy {
    current_goal: Option<GridPos>,
    stuck_ticks:  u8,
}

impl BtStrategy {
    pub fn new() -> Self {
        Self { current_goal: None, stuck_ticks: 0 }
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

        if !obs.gold_carried.is_empty() {
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

    fn reset(&mut self) {
        self.current_goal = None;
        self.stuck_ticks  = 0;
    }
}

// ── Random strategy ───────────────────────────────────────────────────────────

pub struct RandomStrategy;

impl DecisionStrategy for RandomStrategy {
    fn name(&self) -> &'static str { "Random" }

    fn decide(&mut self, obs: &Observation, _planner: &mut impl PathPlanner) -> Action {
        use crate::agent::action::Dir;
        let on_base = matches!(obs.grid_tile(obs.pos), Some(Tile::Base(t)) if t == obs.team.0);
        if on_base && !obs.gold_carried.is_empty() {
            return Action::Drop;
        }
        let dirs = Dir::all();
        Action::Move(dirs[rand::random_range(0..dirs.len())])
    }
}

// ── StrategyKind ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub enum StrategyKind {
    Fsm,
    BehaviorTree,
    Random,
}