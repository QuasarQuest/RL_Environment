// src/algorithm/behavior/goap.rs

#![allow(dead_code)]

use std::time::{Duration, Instant};

// ── Atomics ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct WorldState(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct GoalState(pub u64);

#[derive(Debug, Clone, Copy)]
pub struct Action {
    pub name:         &'static str,
    pub pre_mask:     u64,
    pub pre_value:    u64,
    pub effect_mask:  u64,
    pub effect_value: u64,
    pub cost:         u32,
}

impl Action {
    #[inline(always)]
    pub fn is_valid(&self, state: WorldState) -> bool {
        (state.0 & self.pre_mask) == self.pre_value
    }

    #[inline(always)]
    pub fn apply(&self, state: WorldState) -> WorldState {
        WorldState((state.0 & !self.effect_mask) | self.effect_value)
    }
}

// ── Closed set — 256-slot stack bitset ───────────────────────────────────────

struct ClosedSet([u64; 4]);

impl ClosedSet {
    #[inline(always)]
    fn new() -> Self { Self([0u64; 4]) }

    #[inline(always)]
    fn insert(&mut self, state: WorldState) -> bool {
        let idx  = (state.0 & 0xFF) as usize;
        let word = idx / 64;
        let bit  = idx % 64;
        let mask = 1u64 << bit;
        let fresh = self.0[word] & mask == 0;
        self.0[word] |= mask;
        fresh
    }
}

// ── Planner config & result ───────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct PlanConfig {
    pub max_nodes:        usize,
    pub max_time:         Duration,
    pub heuristic_weight: f32,
}

impl Default for PlanConfig {
    fn default() -> Self {
        Self {
            max_nodes:        512,
            max_time:         Duration::from_millis(1),
            heuristic_weight: 1.5,
        }
    }
}

pub struct PlanResult {
    pub steps: Vec<&'static str>,
}

#[derive(Debug)]
pub enum PlanError {
    EmptyActionList,
    NoPathFound,
    NodeLimitExceeded,
    TimeLimitExceeded,
}

// ── A* internals ──────────────────────────────────────────────────────────────

struct Node {
    state:        WorldState,
    cost_g:       u32,
    action_index: usize,
    parent_index: usize,
}

struct BucketQueue {
    buckets: Vec<Vec<usize>>,
    min_idx: usize,
}

impl BucketQueue {
    fn new(cap: usize) -> Self {
        Self { buckets: vec![Vec::new(); cap], min_idx: usize::MAX }
    }

    #[inline(always)]
    fn push(&mut self, priority: u32, value: usize) {
        let i = priority as usize;
        if i >= self.buckets.len() { self.buckets.resize_with(i + 1, Vec::new); }
        self.buckets[i].push(value);
        if i < self.min_idx { self.min_idx = i; }
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<usize> {
        while self.min_idx < self.buckets.len() {
            if let Some(v) = self.buckets[self.min_idx].pop() { return Some(v); }
            self.min_idx += 1;
        }
        None
    }
}

#[inline(always)]
fn heuristic(state: WorldState, goal: GoalState) -> u32 {
    (goal.0 & !(state.0 & goal.0)).count_ones()
}

pub fn plan(
    start:   WorldState,
    goal:    GoalState,
    actions: &[Action],
    config:  PlanConfig,
) -> Result<PlanResult, PlanError> {
    if actions.is_empty() { return Err(PlanError::EmptyActionList); }

    let t0         = Instant::now();
    let wdenom     = 64u32;
    let weight_num = (config.heuristic_weight * wdenom as f32) as u32;

    let mut open   = BucketQueue::new(64);
    let mut arena: Vec<Node> = Vec::with_capacity(64);
    let mut closed = ClosedSet::new();

    let h0 = (heuristic(start, goal) * weight_num) / wdenom;
    arena.push(Node { state: start, cost_g: 0, action_index: 0, parent_index: 0 });
    open.push(h0, 0);
    closed.insert(start);

    let mut expanded = 0usize;

    while let Some(idx) = open.pop() {
        if expanded >= config.max_nodes { return Err(PlanError::NodeLimitExceeded); }
        if expanded & 15 == 0 && t0.elapsed() > config.max_time {
            return Err(PlanError::TimeLimitExceeded);
        }
        expanded += 1;

        let state  = arena[idx].state;
        let cost_g = arena[idx].cost_g;

        if (state.0 & goal.0) == goal.0 {
            let mut steps = Vec::new();
            let mut i = idx;
            while i != 0 {
                steps.push(actions[arena[i].action_index].name);
                i = arena[i].parent_index;
            }
            steps.reverse();
            return Ok(PlanResult { steps });
        }

        for (ai, action) in actions.iter().enumerate() {
            if !action.is_valid(state) { continue; }
            let next = action.apply(state);
            if !closed.insert(next) { continue; }
            if arena.len() >= config.max_nodes { return Err(PlanError::NodeLimitExceeded); }

            let g  = cost_g + action.cost;
            let wh = (heuristic(next, goal) * weight_num) / wdenom;
            let ni = arena.len();
            arena.push(Node { state: next, cost_g: g, action_index: ai, parent_index: idx });
            open.push(g + wh, ni);
        }
    }

    Err(PlanError::NoPathFound)
}

// ── Domain: gold-collection ───────────────────────────────────────────────────

pub const BIT_HAS_GOLD:       u64 = 1 << 0;
pub const BIT_INVENTORY_FULL: u64 = 1 << 1;
pub const BIT_INVENTORY_HALF: u64 = 1 << 2;
pub const BIT_ON_OWN_BASE:    u64 = 1 << 3;
pub const BIT_GOLD_NEARBY:    u64 = 1 << 4;
pub const BIT_ENEMY_NEARBY:   u64 = 1 << 5;
pub const BIT_BASE_CLOSER:    u64 = 1 << 6;
pub const BIT_LOW_HEALTH:     u64 = 1 << 7;

const NEAR_SQ:           i32 = 10 * 10;
const LOW_HEALTH_THRESH: u8 = 1;

pub const ACT_NAVIGATE_TO_GOLD: &str = "navigate_to_gold";
pub const ACT_COLLECT_GOLD:     &str = "collect_gold";
pub const ACT_NAVIGATE_TO_BASE: &str = "navigate_to_base";
pub const ACT_DROP_GOLD:        &str = "drop_gold";
pub const ACT_FLEE:             &str = "flee";
pub const ACT_WAIT:             &str = "wait";

pub static ACTIONS: &[Action] = &[
    Action {
        name:         ACT_NAVIGATE_TO_GOLD,
        pre_mask:     BIT_INVENTORY_FULL,
        pre_value:    0,
        effect_mask:  BIT_GOLD_NEARBY,
        effect_value: BIT_GOLD_NEARBY,
        cost: 4,
    },
    Action {
        name:         ACT_COLLECT_GOLD,
        pre_mask:     BIT_INVENTORY_FULL | BIT_GOLD_NEARBY,
        pre_value:    BIT_GOLD_NEARBY,
        effect_mask:  BIT_HAS_GOLD | BIT_INVENTORY_HALF | BIT_GOLD_NEARBY,
        effect_value: BIT_HAS_GOLD | BIT_INVENTORY_HALF,
        cost: 1,
    },
    Action {
        name:         ACT_NAVIGATE_TO_BASE,
        pre_mask:     BIT_HAS_GOLD,
        pre_value:    BIT_HAS_GOLD,
        effect_mask:  BIT_ON_OWN_BASE,
        effect_value: BIT_ON_OWN_BASE,
        cost: 4,
    },
    Action {
        name:         ACT_DROP_GOLD,
        pre_mask:     BIT_ON_OWN_BASE | BIT_HAS_GOLD,
        pre_value:    BIT_ON_OWN_BASE | BIT_HAS_GOLD,
        effect_mask:  BIT_HAS_GOLD | BIT_INVENTORY_FULL | BIT_INVENTORY_HALF | BIT_ON_OWN_BASE,
        effect_value: 0,
        cost: 1,
    },
    Action {
        name:         ACT_FLEE,
        pre_mask:     BIT_ENEMY_NEARBY | BIT_LOW_HEALTH,
        pre_value:    BIT_ENEMY_NEARBY | BIT_LOW_HEALTH,
        effect_mask:  BIT_ENEMY_NEARBY,
        effect_value: 0,
        cost: 1,
    },
    Action {
        name:         ACT_WAIT,
        pre_mask:     0,
        pre_value:    0,
        effect_mask:  0,
        effect_value: 0,
        cost: 20,
    },
];

