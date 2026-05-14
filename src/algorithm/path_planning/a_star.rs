// src/algorithm/path_planning/a_star.rs
//
// Change from original:
//   - AStarResult::path is now Vec<GridPos> (unchanged externally), but the
//     reconstruction loop now uses a plain Vec with .reverse() — no change needed here.
//     The VecDeque optimisation lives in AStarAgent (the consumer), not here.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use crate::agent::action::Dir;
use crate::agent::components::GridPos;

use super::graph_utils::{CARDINAL, DIAGONAL, octile};

#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    cost: i32,
    pos:  GridPos,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost) // min-heap
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct AStarResult {
    pub path:       Vec<GridPos>,
    pub closed_set: HashSet<GridPos>,
    pub open_set:   Vec<GridPos>,
}

pub fn compute_path<F>(start: GridPos, goal: GridPos, is_walkable: F) -> AStarResult
where
    F: Fn(GridPos) -> bool,
{
    let mut frontier     = BinaryHeap::new();
    let mut came_from:   HashMap<GridPos, GridPos> = HashMap::new();
    let mut cost_so_far: HashMap<GridPos, i32>     = HashMap::new();
    let mut closed_set   = HashSet::new();

    frontier.push(State { cost: 0, pos: start });
    cost_so_far.insert(start, 0);

    while let Some(State { pos: current, .. }) = frontier.pop() {
        if closed_set.contains(&current) { continue; }
        closed_set.insert(current);

        if current == goal { break; }

        for &dir in Dir::all() {
            let (dx, dy) = dir.delta();
            let next     = GridPos::new(current.x + dx, current.y + dy);

            if !is_walkable(next) && next != goal { continue; }

            if dir.is_diagonal() {
                let check1 = GridPos::new(current.x + dx, current.y);
                let check2 = GridPos::new(current.x, current.y + dy);
                if !is_walkable(check1) || !is_walkable(check2) { continue; }
            }

            let step_cost = if dir.is_diagonal() { DIAGONAL } else { CARDINAL };
            let new_cost  = cost_so_far[&current] + step_cost;

            if !cost_so_far.contains_key(&next) || new_cost < cost_so_far[&next] {
                cost_so_far.insert(next, new_cost);
                let h = octile(next, goal);
                frontier.push(State { cost: new_cost + h, pos: next });
                came_from.insert(next, current);
            }
        }
    }

    let mut path = Vec::new();
    if came_from.contains_key(&goal) || start == goal {
        let mut current = goal;
        while current != start {
            path.push(current);
            match came_from.get(&current) {
                Some(&prev) => current = prev,
                None        => break,
            }
        }
        path.reverse();
    }

    let open_set = frontier.into_iter().map(|s| s.pos).collect();
    AStarResult { path, closed_set, open_set }
}