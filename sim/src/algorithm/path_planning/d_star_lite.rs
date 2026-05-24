// src/algorithm/path_planning/d_star_lite.rs
//
// Changes from original:
//   - neighbors() no longer allocates a Vec on every call.
//     It returns a fixed-size array [GridPos; 8] which the caller iterates.
//     All call sites updated accordingly.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use crate::entity::agent::Dir;
use crate::world::coords::GridPos;

use super::graph_utils::{CARDINAL, DIAGONAL, octile};

const INF: i32 = 1_000_000;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Key(i32, i32);

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.0.cmp(&other.0) {
            Ordering::Equal => self.1.cmp(&other.1),
            ord             => ord,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct State {
    pub key: Key,
    pub pos: GridPos,
}
impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering { other.key.cmp(&self.key) }
}

pub struct DStarLite {
    pub start: GridPos,
    pub goal:  GridPos,
    k_m:       i32,

    g:   HashMap<GridPos, i32>,
    rhs: HashMap<GridPos, i32>,

    queue:    BinaryHeap<State>,
    in_queue: HashMap<GridPos, Key>,

    pub known_obstacles: HashSet<GridPos>,
}

impl DStarLite {
    pub fn new(start: GridPos, goal: GridPos) -> Self {
        let mut planner = Self {
            start,
            goal,
            k_m: 0,
            g:        HashMap::new(),
            rhs:      HashMap::new(),
            queue:    BinaryHeap::new(),
            in_queue: HashMap::new(),
            known_obstacles: HashSet::new(),
        };
        planner.rhs.insert(goal, 0);
        planner.push_queue(goal);
        planner
    }

    fn calculate_key(&self, pos: GridPos) -> Key {
        let g   = self.get_g(pos);
        let rhs = self.get_rhs(pos);
        let min = g.min(rhs);
        Key(
            min.saturating_add(octile(self.start, pos)).saturating_add(self.k_m),
            min,
        )
    }

    fn update_vertex(&mut self, u: GridPos) {
        if u != self.goal {
            let mut min_rhs = INF;
            for s in self.neighbors(u) {
                let c = self.cost(u, s);
                if c == INF && self.get_g(s) == INF { continue; }
                min_rhs = min_rhs.min(c.saturating_add(self.get_g(s)));
            }
            self.rhs.insert(u, min_rhs);
        }
        self.in_queue.remove(&u);
        if self.get_g(u) != self.get_rhs(u) {
            self.push_queue(u);
        }
    }

    pub fn compute_shortest_path(&mut self) {
        while let Some(u_state) = self.peek_queue() {
            let k_old     = u_state.key;
            let start_key = self.calculate_key(self.start);

            if k_old >= start_key && self.get_rhs(self.start) == self.get_g(self.start) {
                break;
            }

            let state = self.pop_queue().unwrap();
            let u     = state.pos;
            let k_new = self.calculate_key(u);

            if k_old < k_new {
                self.push_queue(u);
            } else if self.get_g(u) > self.get_rhs(u) {
                self.g.insert(u, self.get_rhs(u));
                for s in self.neighbors(u) {
                    self.update_vertex(s);
                }
            } else {
                self.g.insert(u, INF);
                let preds: Vec<GridPos> = self.neighbors(u).into_iter()
                    .chain(std::iter::once(u))
                    .collect();
                for s in preds {
                    self.update_vertex(s);
                }
            }
        }
    }

    pub fn update_start(&mut self, new_start: GridPos) {
        self.k_m  += octile(self.start, new_start);
        self.start  = new_start;
    }

    pub fn add_obstacle(&mut self, pos: GridPos) {
        if self.known_obstacles.insert(pos) {
            self.update_vertex(pos);
            for n in self.neighbors(pos) {
                self.update_vertex(n);
            }
        }
    }

    pub fn get_next_step(&self) -> Option<GridPos> {
        if self.start == self.goal         { return None; }
        if self.get_rhs(self.start) == INF { return None; }

        let mut best_step = None;
        let mut min_cost  = INF;

        for s in self.neighbors(self.start) {
            let c = self.cost(self.start, s);
            if c == INF { continue; }
            let total = c.saturating_add(self.get_g(s));
            if total < min_cost {
                min_cost  = total;
                best_step = Some(s);
            }
        }
        best_step
    }

    pub fn generate_path(&self) -> Vec<GridPos> {
        let mut path    = Vec::new();
        let mut curr    = self.start;
        let mut visited = HashSet::new();

        while curr != self.goal {
            if !visited.insert(curr) { break; }
            let mut best_step = None;
            let mut min_cost  = INF;

            for s in self.neighbors(curr) {
                let c = self.cost(curr, s);
                if c == INF { continue; }
                let total = c.saturating_add(self.get_g(s));
                if total < min_cost {
                    min_cost  = total;
                    best_step = Some(s);
                }
            }

            match best_step {
                Some(step) => { path.push(step); curr = step; }
                None       => break,
            }
        }
        path
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn get_g(&self, pos: GridPos) -> i32 {
        *self.g.get(&pos).unwrap_or(&INF)
    }

    fn get_rhs(&self, pos: GridPos) -> i32 {
        *self.rhs.get(&pos).unwrap_or(&INF)
    }

    fn push_queue(&mut self, pos: GridPos) {
        let k = self.calculate_key(pos);
        self.in_queue.insert(pos, k);
        self.queue.push(State { key: k, pos });
    }

    fn peek_queue(&mut self) -> Option<State> {
        while let Some(state) = self.queue.peek() {
            if self.in_queue.get(&state.pos) == Some(&state.key) {
                return Some(*state);
            }
            self.queue.pop();
        }
        None
    }

    fn pop_queue(&mut self) -> Option<State> {
        while let Some(state) = self.queue.pop() {
            if self.in_queue.get(&state.pos) == Some(&state.key) {
                self.in_queue.remove(&state.pos);
                return Some(state);
            }
        }
        None
    }

    /// Returns all 8 neighbours as a fixed array — zero heap allocation.
    fn neighbors(&self, pos: GridPos) -> [GridPos; 8] {
        let dirs = Dir::all();
        std::array::from_fn(|i| {
            let (dx, dy) = dirs[i].delta();
            GridPos::new(pos.x + dx, pos.y + dy)
        })
    }

    fn cost(&self, u: GridPos, v: GridPos) -> i32 {
        if self.known_obstacles.contains(&u) || self.known_obstacles.contains(&v) {
            return INF;
        }
        let dx = v.x - u.x;
        let dy = v.y - u.y;
        if dx != 0 && dy != 0 {
            let check1 = GridPos::new(u.x + dx, u.y);
            let check2 = GridPos::new(u.x, u.y + dy);
            if self.known_obstacles.contains(&check1) || self.known_obstacles.contains(&check2) {
                return INF;
            }
            return DIAGONAL;
        }
        CARDINAL
    }
}