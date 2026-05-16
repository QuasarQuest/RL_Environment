// src/agent/strategy/goap.rs

use crate::agent::action::Action;
use crate::agent::components::GridPos;
use crate::agent::observation::Observation;
use crate::agent::planner::PathPlanner;
use crate::algorithm::behavior_planning::goap::{
    self, GoalState, PlanConfig, PlanError, WorldState,
    BIT_HAS_GOLD, BIT_ON_OWN_BASE, ACTIONS,
};
use crate::item::ItemKind;
use super::{DecisionStrategy, try_move};

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

        let on_base = matches!(obs.grid_tile(obs.pos),
            Some(crate::world::tile::Tile::Base(t)) if t == obs.team.0);
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