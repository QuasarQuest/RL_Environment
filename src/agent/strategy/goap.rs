// src/agent/strategy/goap.rs

use crate::agent::action::Action;
use crate::agent::components::GridPos;
use crate::agent::debug::DebugDraw;
use crate::agent::observation::Observation;
use crate::agent::planner::{AStarPlanner, DStarPlanner, PathPlanner};
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
    planner:      Box<dyn PathPlanner>,
    stuck_ticks:  u8,
}

impl GoapStrategy {
    pub fn new_astar() -> Self { Self::with(Box::new(AStarPlanner::new())) }
    pub fn new_dstar() -> Self { Self::with(Box::new(DStarPlanner::new())) }

    fn with(planner: Box<dyn PathPlanner>) -> Self {
        Self {
            config:       PlanConfig::default(),
            cached_plan:  Vec::new(),
            last_ws:      None,
            current_goal: None,
            planner,
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

    fn decide(&mut self, obs: &Observation) -> Action {
        let walkable = obs.walkability_fn();
        self.planner.update(obs.pos, &walkable);

        let on_base = matches!(obs.grid_tile(obs.pos),
            Some(crate::world::tile::Tile::Base(t)) if t == obs.team.0);
        if on_base && !obs.gold_carried.is_empty() {
            // Invalidate plan — world state has changed (gold delivered).
            self.last_ws = None;
            return Action::Drop;
        }

        let ws   = goap::obs_to_world_state(obs);
        let step = match self.maybe_replan(ws) {
            Some(s) => s,
            None    => return Action::Wait,
        };

        // ACT_COLLECT_GOLD: the item system handles pickup via tile collision —
        // the agent just needs to stand on the gold tile, which navigation
        // already achieves. Do NOT invalidate last_ws here; the world state
        // hasn't changed yet and forcing a replan every tick is wasteful.
        // last_ws will naturally become stale once gold is picked up and
        // obs_to_world_state returns a new value.
        if step == goap::ACT_COLLECT_GOLD { return Action::Wait; }

        if step == goap::ACT_DROP_GOLD {
            // Invalidate plan — drop is about to change world state.
            self.last_ws = None;
            return Action::Drop;
        }

        let Some(goal_pos) = Self::nav_target(step, obs) else { return Action::Wait; };

        if self.current_goal != Some(goal_pos) {
            self.planner.set_goal(obs.pos, goal_pos, &walkable);
            self.current_goal = Some(goal_pos);
        }

        match try_move(self.planner.as_mut(), obs, &mut self.stuck_ticks) {
            Ok(action) => action,
            Err(()) => {
                self.planner.reset();
                self.current_goal = None;
                self.last_ws      = None;
                Action::Wait
            }
        }
    }

    fn debug_draw(&self) -> Option<Box<dyn DebugDraw>> {
        self.planner.debug_draw()
    }

    fn reset(&mut self) {
        self.cached_plan.clear();
        self.last_ws      = None;
        self.current_goal = None;
        self.planner.reset();
        self.stuck_ticks  = 0;
    }
}