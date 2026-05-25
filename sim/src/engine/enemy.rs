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

// Chebyshev distance — correct admissible heuristic for 8-directional movement
// where diagonal steps cost the same as cardinal steps.
fn chebyshev(a: GridPos, b: GridPos) -> i32 {
    (a.x - b.x).abs().max((a.y - b.y).abs())
}

fn nearest_gold(pos: GridPos, items: &[ItemState]) -> Option<GridPos> {
    items.iter()
        .filter(|i| i.kind == ItemKind::Gold)
        .min_by_key(|i| chebyshev(pos, i.pos))
        .map(|i| i.pos)
}

// All 8 movement directions: (dx, dy, Dir).
const DIRS: [(i32, i32, Dir); 8] = [
    ( 0,  1, Dir::N),
    ( 0, -1, Dir::S),
    ( 1,  0, Dir::E),
    (-1,  0, Dir::W),
    ( 1,  1, Dir::NE),
    (-1,  1, Dir::NW),
    ( 1, -1, Dir::SE),
    (-1, -1, Dir::SW),
];

fn is_walkable(grid: &Grid, x: i32, y: i32) -> bool {
    grid.get(x, y).map_or(false, |t| t.is_walkable())
}

// Diagonal moves require both adjacent cardinal tiles to be walkable to
// prevent the agent from slipping through diagonal wall gaps.
fn diagonal_clear(grid: &Grid, fx: i32, fy: i32, dx: i32, dy: i32) -> bool {
    is_walkable(grid, fx + dx, fy) && is_walkable(grid, fx, fy + dy)
}

// ── SimpleChaser ──────────────────────────────────────────────────────────────

/// Move one step toward `to`, preferring the diagonal when both axes have
/// remaining distance. Falls back to cardinal-only when the diagonal is
/// blocked or only one axis needs movement.
fn greedy_move(from: GridPos, to: GridPos, grid: &Grid) -> Option<Dir> {
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    if dx == 0 && dy == 0 { return None; }

    let sdx = dx.signum();
    let sdy = dy.signum();

    // Build candidate (dx, dy) pairs ordered from most preferred to fallback.
    let candidates: &[(i32, i32)] = if sdx != 0 && sdy != 0 {
        &[(sdx, sdy), (sdx, 0), (0, sdy)]   // diagonal first, then each axis
    } else if sdx != 0 {
        &[(sdx, 0)]
    } else {
        &[(0, sdy)]
    };

    for &(cdx, cdy) in candidates {
        let is_clear = if cdx != 0 && cdy != 0 {
            diagonal_clear(grid, from.x, from.y, cdx, cdy)
                && is_walkable(grid, from.x + cdx, from.y + cdy)
        } else {
            is_walkable(grid, from.x + cdx, from.y + cdy)
        };
        if is_clear {
            return DIRS.iter()
                .find(|&&(ex, ey, _)| ex == cdx && ey == cdy)
                .map(|&(_, _, d)| d);
        }
    }
    None
}

// ── A* pathfinding ────────────────────────────────────────────────────────────

fn astar(grid: &Grid, start: GridPos, goal: GridPos) -> VecDeque<GridPos> {
    if start == goal { return VecDeque::new(); }

    // Cross-product tie-breaking: for a candidate node N, compute the signed
    // area of the triangle (start, goal, N).  Nodes that lie exactly on the
    // start→goal line score 0; nodes that deviate score higher.  Storing
    // Reverse(cross) as the second sort key breaks ties toward the straight
    // line, preventing the "right-hook" artifact that appears when GridPos
    // lexicographic ordering is used as the sole tiebreaker.
    let sdx = start.x - goal.x;
    let sdy = start.y - goal.y;
    let cross = |n: GridPos| -> i32 {
        ((n.x - goal.x) * sdy - sdx * (n.y - goal.y)).abs()
    };

    // Open set: (Reverse<f>, Reverse<cross>, g, position)
    let mut open: BinaryHeap<(Reverse<i32>, Reverse<i32>, i32, GridPos)> = BinaryHeap::new();
    let mut came_from: HashMap<GridPos, GridPos> = HashMap::new();
    let mut g_score:   HashMap<GridPos, i32>     = HashMap::new();

    g_score.insert(start, 0);
    open.push((Reverse(chebyshev(start, goal)), Reverse(cross(start)), 0, start));

    while let Some((_, _, g, current)) = open.pop() {
        if current == goal {
            let mut path = VecDeque::new();
            let mut cur = current;
            while cur != start {
                path.push_front(cur);
                match came_from.get(&cur) {
                    Some(&prev) => cur = prev,
                    None        => break,
                }
            }
            return path;
        }
        if g > *g_score.get(&current).unwrap_or(&i32::MAX) { continue; }

        for &(dx, dy, _) in &DIRS {
            let nx = current.x + dx;
            let ny = current.y + dy;
            // Prevent corner-cutting: both adjacent cardinal tiles must be
            // walkable before allowing a diagonal step.
            if dx != 0 && dy != 0 && !diagonal_clear(grid, current.x, current.y, dx, dy) {
                continue;
            }
            if !is_walkable(grid, nx, ny) { continue; }
            let neighbor = GridPos::new(nx, ny);
            let tentative_g = g + 1;
            if tentative_g < *g_score.get(&neighbor).unwrap_or(&i32::MAX) {
                g_score.insert(neighbor, tentative_g);
                came_from.insert(neighbor, current);
                let f = tentative_g + chebyshev(neighbor, goal);
                open.push((Reverse(f), Reverse(cross(neighbor)), tentative_g, neighbor));
            }
        }
    }
    VecDeque::new() // no path found
}

fn path_next_dir(current: GridPos, path: &VecDeque<GridPos>) -> Option<Dir> {
    let next = path.front()?;
    let dx = next.x - current.x;
    let dy = next.y - current.y;
    DIRS.iter().find(|&&(ex, ey, _)| ex == dx && ey == dy).map(|&(_, _, d)| d)
}

// ── Path cache ────────────────────────────────────────────────────────────────

/// One instance per non-RL agent; lives in `SimCore`.
pub struct EnemyPathCache {
    path:        VecDeque<GridPos>,
    cached_goal: Option<GridPos>,
}

impl EnemyPathCache {
    pub fn new() -> Self { Self::default() }
    pub fn path(&self) -> &VecDeque<GridPos> { &self.path }
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

/// Navigate toward `target` using A* with path caching.
/// Suitable for any scripted mode that needs directed movement.
pub fn navigate_action(
    agent:  &AgentState,
    target: GridPos,
    grid:   &Grid,
    cache:  &mut EnemyPathCache,
) -> Action {
    if cache.cached_goal != Some(target) || cache.path.is_empty() {
        cache.path        = astar(grid, agent.pos, target);
        cache.cached_goal = Some(target);
    }
    while cache.path.front() == Some(&agent.pos) {
        cache.path.pop_front();
    }
    match path_next_dir(agent.pos, &cache.path) {
        Some(dir) => { cache.path.pop_front(); Action::Move(dir) }
        None      => Action::Wait,
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
