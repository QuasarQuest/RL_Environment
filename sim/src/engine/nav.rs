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

use rustc_hash::FxHashSet;

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

/// A tile the agent may *enter*: walkable terrain that is not a `blocked` hazard
/// tile (trap). The agent's current tile is never tested against `blocked`, so a
/// trap that spawns under it never strands the pathfinder (it can still step off).
fn passable(grid: &Grid, blocked: &FxHashSet<GridPos>, x: i32, y: i32) -> bool {
    is_walkable(grid, x, y) && !blocked.contains(&GridPos::new(x, y))
}

// Diagonal moves require both adjacent cardinal tiles to be enterable so the agent
// can't slip through diagonal wall (or trap) gaps.
fn diagonal_clear(grid: &Grid, blocked: &FxHashSet<GridPos>, fx: i32, fy: i32, dx: i32, dy: i32) -> bool {
    passable(grid, blocked, fx + dx, fy) && passable(grid, blocked, fx, fy + dy)
}

fn astar(grid: &Grid, start: GridPos, goal: GridPos, blocked: &FxHashSet<GridPos>) -> VecDeque<GridPos> {
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
            if dx != 0 && dy != 0 && !diagonal_clear(grid, blocked, current.x, current.y, dx, dy) {
                continue;
            }
            if !passable(grid, blocked, nx, ny) { continue; }
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

/// Breadth-first true-movement distance from `start` to every walkable tile, using
/// the agent's own 8-connected movement (diagonals require both adjacent cardinals
/// clear, matching `astar`). Returns a row-major `width × height` buffer of step
/// counts; unreachable tiles (and the start tile if blocked) are `-1`.
///
/// Movement cost is uniform 1 per step (a diagonal costs the same as a cardinal),
/// so a plain BFS gives exact shortest path lengths. This is the "path-aware"
/// distance the policy sees in its cluster features — unlike Chebyshev it accounts
/// for walls, so a region that is near in straight-line but walled off reads as far.
pub fn dist_field(grid: &Grid, start: GridPos, blocked: &FxHashSet<GridPos>) -> Vec<i32> {
    let (w, h) = (grid.width as i32, grid.height as i32);
    let mut dist = vec![-1i32; (w * h) as usize];
    if !is_walkable(grid, start.x, start.y) {
        return dist;
    }
    let idx = |x: i32, y: i32| (y as usize) * grid.width + x as usize;
    let mut q = VecDeque::new();
    dist[idx(start.x, start.y)] = 0;
    q.push_back(start);
    while let Some(c) = q.pop_front() {
        let d = dist[idx(c.x, c.y)];
        for &(dx, dy, _) in &DIRS {
            let (nx, ny) = (c.x + dx, c.y + dy);
            if dx != 0 && dy != 0 && !diagonal_clear(grid, blocked, c.x, c.y, dx, dy) {
                continue;
            }
            if !passable(grid, blocked, nx, ny) {
                continue;
            }
            let ni = idx(nx, ny);
            if dist[ni] == -1 {
                dist[ni] = d + 1;
                q.push_back(GridPos::new(nx, ny));
            }
        }
    }
    dist
}

/// Path distance (from a `dist_field`) to `p`, or `None` if `p` is OOB/unreachable.
#[inline]
pub fn dist_at(dist: &[i32], grid: &Grid, p: GridPos) -> Option<i32> {
    if !grid.in_bounds(p.x, p.y) {
        return None;
    }
    let d = dist[p.y as usize * grid.width + p.x as usize];
    if d < 0 { None } else { Some(d) }
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

/// Navigate one step toward `target` using cached A*. `blocked` is the set of
/// trap tiles the agent routes around (impassable to the pathfinder).
pub fn navigate_action(
    agent:   &AgentState,
    target:  GridPos,
    grid:    &Grid,
    cache:   &mut NavCache,
    blocked: &FxHashSet<GridPos>,
) -> Action {
    // Pop stale front nodes (agent already there after a multi-tile move).
    while cache.path.front() == Some(&agent.pos) {
        cache.path.pop_front();
    }

    // Recompute A* when the goal changed, the path is exhausted, OR the agent has
    // drifted off the cached path. The drift case matters: a 2-tile speed move can
    // overshoot a turn in the path and land the agent on a tile the next waypoint
    // is no longer adjacent to. Without this check `path_next_dir` returns None →
    // Wait, but the goal (nearest gold) is unchanged so the cache never refreshes —
    // the agent freezes mid-map, re-selecting the same nav goal forever (the
    // "valid A* path but agent never moves" bug).
    let off_path = cache.path.front()
        .map_or(false, |&n| chebyshev(agent.pos, n) != 1);
    // A trap can spawn onto a tile of the already-cached path. The path was valid
    // when computed, so neither the goal nor `off_path` changes — recompute when the
    // next waypoint is now a blocked (trap) tile so the agent reroutes instead of
    // stepping straight into it.
    let next_blocked = cache.path.front().map_or(false, |&n| blocked.contains(&n));
    if cache.cached_goal != Some(target) || cache.path.is_empty() || off_path || next_blocked {
        cache.path        = astar(grid, agent.pos, target, blocked);
        cache.cached_goal = Some(target);
        while cache.path.front() == Some(&agent.pos) {
            cache.path.pop_front();
        }
    }

    match path_next_dir(agent.pos, &cache.path) {
        Some(dir) => { cache.path.pop_front(); Action::Move(dir) }
        None      => Action::Wait,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tile::Tile;

    fn agent_at(pos: GridPos) -> AgentState {
        AgentState {
            pos,
            gold_carried: 0,
            score:        0,
            speed_buff:   0,
            mult_charge:  0,
            trap_buff:    0,
            spawn_pos:    pos,
            base_pos:     pos,
            team:         0,
            hearts:       1,
            ammo:         0,
        }
    }

    /// No trap tiles in these navigation unit tests.
    fn no_blocked() -> FxHashSet<GridPos> { FxHashSet::default() }

    /// Replicate engine::physics movement for `moves` tiles in one direction,
    /// stopping at walls or on the navigation target `stop_at` (mirrors
    /// physics::apply_move, which halts on the goal so it never overshoots it).
    fn step_move(agent: &mut AgentState, grid: &Grid, dir: Dir, moves: u32, stop_at: GridPos) {
        let (dx, dy) = dir.delta();
        for _ in 0..moves {
            if agent.pos == stop_at { break; }
            let next = agent.pos.apply_delta(dx, dy);
            if !grid.is_walkable(next.x, next.y) { break; }
            agent.pos = next;
            if next == stop_at { break; }
        }
    }

    /// Regression for the "valid A* path but the agent never moves" freeze.
    ///
    /// A 2-tile speed move can overshoot a turn in the cached path, leaving the
    /// agent on a tile the next waypoint is no longer adjacent to. The cache used
    /// to recompute ONLY on a goal change or an empty path, so with the goal
    /// unchanged the stale, non-adjacent path persisted: `path_next_dir` returned
    /// None → Wait, the agent didn't move, the goal stayed identical, and it froze
    /// forever. navigate_action must instead detect the divergence and recompute.
    #[test]
    fn diverged_cached_path_recomputes_instead_of_freezing() {
        let grid  = Grid::new(12, 12); // all walkable
        let agent = agent_at(GridPos::new(5, 5));
        let goal  = GridPos::new(9, 5);

        // Seed the cache as if a previous multi-tile move overshot a turn: the
        // front waypoint (5,7) is two tiles from the agent (Chebyshev 2), so it is
        // NOT a legal single step — the exact frozen state.
        let mut cache = NavCache::new();
        cache.path = VecDeque::from(vec![
            GridPos::new(5, 7), GridPos::new(7, 6), goal,
        ]);
        cache.cached_goal = Some(goal);

        let act = navigate_action(&agent, goal, &grid, &mut cache, &no_blocked());

        // Pre-fix this returned Action::Wait (frozen). It must now recompute and
        // step toward the goal (eastward, the +x direction).
        match act {
            Action::Move(dir) => {
                let (dx, _) = dir.delta();
                assert!(dx > 0, "expected an eastward step toward the goal, got {dir:?}");
            }
            Action::Wait => panic!("navigate_action froze on a diverged cached path"),
        }
    }

    /// End-to-end: with a permanent 2-tile (speed-buff) move the agent must still
    /// reach the goal — overshoots are re-synced, never a permanent stall.
    #[test]
    fn speed_buffed_navigation_reaches_goal() {
        let grid  = Grid::new(20, 20);
        let mut agent = agent_at(GridPos::new(1, 1));
        let goal  = GridPos::new(17, 12); // off-axis → the path turns
        let mut cache = NavCache::new();

        let mut ticks = 0;
        while agent.pos != goal && ticks < 500 {
            match navigate_action(&agent, goal, &grid, &mut cache, &no_blocked()) {
                Action::Move(dir) => step_move(&mut agent, &grid, dir, 2, goal), // 2 = speed buff
                Action::Wait      => {}
            }
            ticks += 1;
        }
        assert_eq!(agent.pos, goal, "agent failed to reach goal within {ticks} ticks (frozen?)");
    }

    /// Sanity: ordinary single-tile navigation around an obstacle still arrives.
    #[test]
    fn single_step_navigation_reaches_goal_around_obstacle() {
        let mut grid = Grid::new(12, 12);
        // A vertical wall with a gap, forcing the path to turn.
        for y in 0..10 {
            grid.set(6, y, Tile::Obstacle);
        }
        let mut agent = agent_at(GridPos::new(2, 2));
        let goal  = GridPos::new(10, 2);
        let mut cache = NavCache::new();

        let mut ticks = 0;
        while agent.pos != goal && ticks < 500 {
            match navigate_action(&agent, goal, &grid, &mut cache, &no_blocked()) {
                Action::Move(dir) => step_move(&mut agent, &grid, dir, 1, goal),
                Action::Wait      => break,
            }
            ticks += 1;
        }
        assert_eq!(agent.pos, goal, "agent failed to reach goal around the wall");
    }
}
