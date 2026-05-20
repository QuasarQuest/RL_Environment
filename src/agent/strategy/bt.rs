// src/agent/strategy/bt.rs
//
// Opportunistic behaviour tree agent.
//
// Priority:
//   1. Critical survival  — 1 heart → heal immediately
//   2. Ranged attack      — has ammo + clear shot
//   3. Melee attack       — adjacent enemy
//   4. Drop gold on base
//   5. Deliver            — full OR base closer than gold
//   6. Collect gold
//   7. Opportunistic heal — hurt + health nearby + no enemies
//   8. Flee               — hurt + enemy nearby + no health
//   9. Roam

use crate::agent::action::{Action, Dir};
use crate::agent::components::GridPos;
use crate::agent::debug::DebugDraw;
use crate::agent::observation::Observation;
use crate::agent::planner::{AStarPlanner, DStarPlanner, PathPlanner};
use crate::algorithm::behavior_planning::behavior_tree::{BtNode, Status, selector, sequence, cond, leaf};
use crate::item::ItemKind;
use crate::world::tile::Tile;
use super::{DecisionStrategy, adjacent_enemy_dir, ranged_enemy_dir, roam, try_move};

// ── Intent types ──────────────────────────────────────────────────────────────

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

// ── Nav resolution ────────────────────────────────────────────────────────────

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

        // Flee: pick the walkable neighbour that maximises distance from the
        // enemy. O(8), no planner needed — replanning every tick as the enemy
        // moves would thrash the path cache and standing still is the worst
        // possible outcome when injured.
        NavTarget::FleeFrom(enemy_pos) => {
            return Dir::all().iter().copied()
                .map(|d| {
                    let (dx, dy) = d.delta();
                    GridPos::new(obs.pos.x + dx, obs.pos.y + dy)
                })
                .filter(|&p| obs.is_walkable(p))
                .max_by_key(|p| p.dist_sq(enemy_pos))
                .map(|best| {
                    // Convert best tile back to a Move action.
                    let dx = (best.x - obs.pos.x).signum();
                    let dy = (best.y - obs.pos.y).signum();
                    Dir::all().iter().copied()
                        .find(|d| d.delta() == (dx, dy))
                        .map(Action::Move)
                        .unwrap_or(Action::Wait)
                })
                .unwrap_or(Action::Wait);
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

// ── Tree construction ─────────────────────────────────────────────────────────

fn build_tree() -> Box<dyn BtNode<Observation, BtOutput>> {
    selector(vec![
        // 1. Critical survival
        sequence(vec![
            cond(|o: &Observation| o.hearts.0 <= 1 && o.nearest_item(ItemKind::Health).is_some()),
            leaf(|_: &Observation, out: &mut Option<BtOutput>| {
                *out = Some(BtOutput::Navigate(NavTarget::Health));
                Status::Running
            }),
        ]),
        // 2. Ranged attack — evaluate dir once, store in output
        sequence(vec![
            cond(|o: &Observation| o.has_ammo() && ranged_enemy_dir(o).is_some()),
            leaf(|o: &Observation, out: &mut Option<BtOutput>| {
                // ranged_enemy_dir called once here; cond already confirmed Some
                if let Some(d) = ranged_enemy_dir(o) {
                    *out = Some(BtOutput::Direct(Action::RangedAttack(d)));
                    Status::Running
                } else {
                    Status::Failure
                }
            }),
        ]),
        // 3. Melee attack — evaluate dir once, store in output
        sequence(vec![
            cond(|o: &Observation| adjacent_enemy_dir(o).is_some()),
            leaf(|o: &Observation, out: &mut Option<BtOutput>| {
                if let Some(d) = adjacent_enemy_dir(o) {
                    *out = Some(BtOutput::Direct(Action::Attack(d)));
                    Status::Running
                } else {
                    Status::Failure
                }
            }),
        ]),
        // 4. Drop gold on base
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
        // 5. Deliver — full bag, or base is closer than nearest gold
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
        // 7. Opportunistic heal — hurt, health available, no enemy nearby
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
        // 8. Flee — hurt, enemy nearby, no health pickup available
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
        Self { tree: build_tree(), planner, nav: NavState::new() }
    }
}

impl DecisionStrategy for BtStrategy {
    fn name(&self) -> &'static str { "BT" }

    fn decide(&mut self, obs: &Observation) -> Action {
        let mut out = None;
        self.tree.tick(obs, &mut out);
        match out.unwrap_or(BtOutput::Direct(Action::Wait)) {
            BtOutput::Direct(action)   => action,
            BtOutput::Navigate(target) =>
                resolve_nav(target, obs, self.planner.as_mut(), &mut self.nav),
        }
    }

    fn debug_draw(&self) -> Option<Box<dyn DebugDraw>> {
        self.planner.debug_draw()
    }

    fn reset(&mut self) { self.nav.reset(); self.planner.reset(); }
}