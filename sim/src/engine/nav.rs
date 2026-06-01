// src/engine/nav.rs
//
// Low-level navigation for the RL agent: A* to a goal tile with a per-agent path
// cache. The policy chooses high-level goals (a gold region, the base, a buff
// item); this module is the A* "controller" that walks there one step at a time.
//
// (Previously this lived in engine/enemy.rs alongside scripted-enemy AI; combat
// was removed for the single-agent gold rush, so only the navigation half remains.)

use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::cmp::Reverse;

use crate::entity::agent::{AgentState, Action, Dir};
use crate::world::coords::GridPos;
use crate::world::grid::Grid;

// Chebyshev distance — admissible heuristic for 8-directional movement where a
// diagonal step costs the same as a cardinal step.
fn chebyshev(a: GridPos, b: GridPos) -> i32 {
    (a.x - b.x).abs().max((a.y - b.y).abs())
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

// Diagonal moves require both adjacent cardinal tiles to be walkable so the agent
// can't slip through diagonal wall gaps.
fn diagonal_clear(grid: &Grid, fx: i32, fy: i32, dx: i32, dy: i32) -> bool {
    is_walkable(grid, fx + dx, fy) && is_walkable(grid, fx, fy + dy)
}

fn astar(grid: &Grid, start: GridPos, goal: GridPos) -> VecDeque<GridPos> {
    if start == goal { return VecDeque::new(); }

    // Cross-product tie-breaking: nodes on the start→goal line score 0, deviating
    // nodes score higher. Reverse(cross) as the second key breaks ties toward the
    // straight line, preventing the "right-hook" artifact from lexicographic order.
    let sdx = start.x - goal.x;
    let sdy = start.y - goal.y;
    let cross = |n: GridPos| -> i32 {
        ((n.x - goal.x) * sdy - sdx * (n.y - goal.y)).abs()
    };

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

// ── Path cache ──────────────────────────────────────────────────────────────────

/// Per-agent A* path cache. Recomputes only when the goal changes or the path is
/// consumed.
pub struct NavCache {
    path:        VecDeque<GridPos>,
    cached_goal: Option<GridPos>,
}

impl NavCache {
    pub fn new() -> Self { Self::default() }
    pub fn path(&self) -> &VecDeque<GridPos> { &self.path }
}

impl Default for NavCache {
    fn default() -> Self { Self { path: VecDeque::new(), cached_goal: None } }
}

/// Navigate one step toward `target` using cached A*.
pub fn navigate_action(
    agent:  &AgentState,
    target: GridPos,
    grid:   &Grid,
    cache:  &mut NavCache,
) -> Action {
    if cache.cached_goal != Some(target) || cache.path.is_empty() {
        cache.path        = astar(grid, agent.pos, target);
        cache.cached_goal = Some(target);
    }
    // Pop stale front nodes (agent already there after a multi-tile move).
    while cache.path.front() == Some(&agent.pos) {
        cache.path.pop_front();
    }
    match path_next_dir(agent.pos, &cache.path) {
        Some(dir) => { cache.path.pop_front(); Action::Move(dir) }
        None      => Action::Wait,
    }
}
