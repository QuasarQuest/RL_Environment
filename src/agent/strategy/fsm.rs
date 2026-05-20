// src/agent/strategy/fsm.rs

use crate::agent::action::Action;
use crate::agent::components::GridPos;
use crate::agent::debug::DebugDraw;
use crate::agent::observation::Observation;
use crate::agent::planner::{AStarPlanner, DStarPlanner, PathPlanner};
use crate::algorithm::behavior_planning::fsm::{Fsm, FsmState};
use crate::item::ItemKind;
use crate::world::tile::Tile;
use super::{DecisionStrategy, try_move};

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
    planner:     Box<dyn PathPlanner>,
    stuck_ticks: u8,
}

impl FsmStrategy {
    // new_none() removed — FSM requires a planner to navigate; pairing it
    // with NoPlanner produces an agent that always waits. Use new_astar()
    // as the default fallback instead.
    pub fn new_astar() -> Self { Self::with(Box::new(AStarPlanner::new())) }
    pub fn new_dstar() -> Self { Self::with(Box::new(DStarPlanner::new())) }

    fn with(planner: Box<dyn PathPlanner>) -> Self {
        Self { fsm: Fsm::new(FsmPhase::Idle), planner, stuck_ticks: 0 }
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

    fn decide(&mut self, obs: &Observation) -> Action {
        let walkable = obs.walkability_fn();
        self.planner.update(obs.pos, &walkable);

        let on_base = matches!(obs.grid_tile(obs.pos), Some(Tile::Base(t)) if t == obs.team.0);
        if on_base && !obs.gold_carried.is_empty() {
            self.fsm.transition(FsmPhase::Active { phase: CollectPhase::Dropping, target: obs.pos });
            return Action::Drop;
        }
        if let FsmPhase::Active { phase: CollectPhase::Dropping, .. } = self.fsm.state() {
            self.fsm.transition(FsmPhase::Idle);
            self.planner.reset();
        }

        if obs.gold_carried.is_full() {
            let base = match obs.nearest_own_base() { Some(b) => b, None => return Action::Wait };
            if self.current_target() != Some(base) || self.planner.next_step().is_none() {
                self.planner.set_goal(obs.pos, base, &walkable);
                self.fsm.transition(FsmPhase::Active { phase: CollectPhase::Returning, target: base });
            }
            return match try_move(self.planner.as_mut(), obs, &mut self.stuck_ticks) {
                Ok(a)   => a,
                Err(()) => { self.planner.reset(); self.fsm.transition(FsmPhase::Idle); Action::Wait }
            };
        }

        if let Some(gold) = obs.nearest_item(ItemKind::Gold) {
            if self.current_target() != Some(gold) || self.planner.next_step().is_none() {
                self.planner.set_goal(obs.pos, gold, &walkable);
                self.fsm.transition(FsmPhase::Active { phase: CollectPhase::Collecting, target: gold });
            }
            return match try_move(self.planner.as_mut(), obs, &mut self.stuck_ticks) {
                Ok(a)   => a,
                Err(()) => { self.planner.reset(); self.fsm.transition(FsmPhase::Idle); Action::Wait }
            };
        }

        self.fsm.transition(FsmPhase::Idle);
        Action::Wait
    }

    fn debug_draw(&self) -> Option<Box<dyn DebugDraw>> {
        self.planner.debug_draw()
    }

    fn reset(&mut self) {
        self.fsm.transition(FsmPhase::Idle);
        self.planner.reset();
        self.stuck_ticks = 0;
    }
}