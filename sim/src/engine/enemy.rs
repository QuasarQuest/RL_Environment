// src/engine/enemy.rs
//
// Scripted enemy AI for stages 4+.
//
// SimpleChaser: greedy step toward the nearest gold (or own base).
//   No pathfinding — easy to trap against walls.
//
// BehaviorTree: A* pathfinding to the nearest gold (or own base).
//   Path is cached per-enemy and recomputed only when the goal changes or
//   the path is exhausted.

use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::cmp::Reverse;

use crate::config::AGENT_MAX_GOLD;
use crate::entity::agent::{AgentState, Action, Dir};
use crate::entity::item::{ItemKind, ItemState};
use crate::world::coords::GridPos;
use crate::world::config::EnemyKind;
use crate::world::grid::Grid;

// ── Shared helpers ────────────────────────────────────────────────────────────

fn manhattan(a: GridPos, b: GridPos) -> i32 {
    (a.x - b.x).abs() + (a.y - b.y).abs()
}

fn nearest_gold(pos: GridPos, items: &[ItemState]) -> Option<GridPos> {
    items.iter()
        .filter(|i| i.kind == ItemKind::Gold)
        .min_by_key(|i| manhattan(pos, i.pos))
        .map(|i| i.pos)
}

// Cardinal directions with (dx, dy, Dir) triples.
const CARDINAL: [(i32, i32, Dir); 4] = [
    ( 0,  1, Dir::N),
    ( 0, -1, Dir::S),
    ( 1,  0, Dir::E),
    (-1,  0, Dir::W),
];

fn is_walkable(grid: &Grid, x: i32, y: i32) -> bool {
    grid.get(x, y).map_or(false, |t| t.is_walkable())
}

// ── SimpleChaser ──────────────────────────────────────────────────────────────

/// Move one step along the axis with the largest remaining delta.
/// Tries the secondary axis if the primary is blocked.
fn greedy_move(from: GridPos, to: GridPos, grid: &Grid) -> Option<Dir> {
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    if dx == 0 && dy == 0 { return None; }

    let primary_first = dx.abs() >= dy.abs();

    let candidates: [(i32, i32); 2] = if primary_first {
        [(dx.signum(), 0), (0, dy.signum())]
    } else {
        [(0, dy.signum()), (dx.signum(), 0)]
    };

    for (ddx, ddy) in candidates {
        if ddx == 0 && ddy == 0 { continue; }
        if is_walkable(grid, from.x + ddx, from.y + ddy) {
            return CARDINAL.iter()
                .find(|&&(ex, ey, _)| ex == ddx && ey == ddy)
                .map(|&(_, _, d)| d);
        }
    }
    None
}

// ── A* pathfinding ────────────────────────────────────────────────────────────

fn astar(grid: &Grid, start: GridPos, goal: GridPos) -> VecDeque<GridPos> {
    if start == goal { return VecDeque::new(); }

    // Open set: (f-cost, g-cost, position)
    let mut open: BinaryHeap<(Reverse<i32>, i32, GridPos)> = BinaryHeap::new();
    let mut came_from: HashMap<GridPos, GridPos>            = HashMap::new();
    let mut g_score:   HashMap<GridPos, i32>                = HashMap::new();

    g_score.insert(start, 0);
    open.push((Reverse(manhattan(start, goal)), 0, start));

    while let Some((_, g, current)) = open.pop() {
        if current == goal {
            let mut path = VecDeque::new();
            let mut cur = current;
            while cur != start {
                path.push_front(cur);
                cur = *came_from.get(&cur).unwrap();
            }
            return path;
        }
        if g > *g_score.get(&current).unwrap_or(&i32::MAX) { continue; }

        for &(dx, dy, _) in &CARDINAL {
            let nx = current.x + dx;
            let ny = current.y + dy;
            if !is_walkable(grid, nx, ny) { continue; }
            let neighbor   = GridPos::new(nx, ny);
            let tentative_g = g + 1;
            if tentative_g < *g_score.get(&neighbor).unwrap_or(&i32::MAX) {
                g_score.insert(neighbor, tentative_g);
                came_from.insert(neighbor, current);
                open.push((Reverse(tentative_g + manhattan(neighbor, goal)), tentative_g, neighbor));
            }
        }
    }
    VecDeque::new() // no path found
}

fn path_next_dir(current: GridPos, path: &VecDeque<GridPos>) -> Option<Dir> {
    let next = path.front()?;
    let dx = next.x - current.x;
    let dy = next.y - current.y;
    CARDINAL.iter().find(|&&(ex, ey, _)| ex == dx && ey == dy).map(|&(_, _, d)| d)
}

// ── Path cache ────────────────────────────────────────────────────────────────

/// One instance per non-RL agent; lives in `SimCore`.
pub struct EnemyPathCache {
    path:        VecDeque<GridPos>,
    cached_goal: Option<GridPos>,
}

impl EnemyPathCache {
    pub fn new() -> Self { Self::default() }
}

impl Default for EnemyPathCache {
    fn default() -> Self { Self { path: VecDeque::new(), cached_goal: None } }
}

// ── Public interface ──────────────────────────────────────────────────────────

/// Compute the next `Action` for a scripted enemy agent at index `idx`.
pub fn compute_action(
    kind:  EnemyKind,
    agent: &AgentState,
    items: &[ItemState],
    grid:  &Grid,
    cache: &mut EnemyPathCache,
) -> Action {
    match kind {
        EnemyKind::None         => Action::Wait,
        EnemyKind::SimpleChaser => chaser_action(agent, items, grid),
        EnemyKind::BehaviorTree => bt_action(agent, items, grid, cache),
    }
}

/// Shared goal selection: fetch gold until full, then return to base.
fn select_goal(agent: &AgentState, items: &[ItemState]) -> Option<GridPos> {
    if agent.gold_carried < AGENT_MAX_GOLD {
        nearest_gold(agent.pos, items)
    } else {
        Some(agent.base_pos)
    }
}

fn chaser_action(agent: &AgentState, items: &[ItemState], grid: &Grid) -> Action {
    match select_goal(agent, items) {
        Some(target) => greedy_move(agent.pos, target, grid)
            .map(Action::Move)
            .unwrap_or(Action::Wait),
        None => Action::Wait,
    }
}

fn bt_action(
    agent: &AgentState,
    items: &[ItemState],
    grid:  &Grid,
    cache: &mut EnemyPathCache,
) -> Action {
    let goal = match select_goal(agent, items) {
        Some(g) => g,
        None    => return Action::Wait,
    };

    // Recompute path when goal changes or path is consumed.
    if cache.cached_goal != Some(goal) || cache.path.is_empty() {
        cache.path        = astar(grid, agent.pos, goal);
        cache.cached_goal = Some(goal);
    }

    // Pop any stale front nodes (agent already there due to multi-tile moves).
    while cache.path.front() == Some(&agent.pos) {
        cache.path.pop_front();
    }

    match path_next_dir(agent.pos, &cache.path) {
        Some(dir) => { cache.path.pop_front(); Action::Move(dir) }
        None      => Action::Wait,
    }
}
