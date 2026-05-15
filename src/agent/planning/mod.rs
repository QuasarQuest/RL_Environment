// src/agent/planning/mod.rs
//
// Flat layout — one concept per file:
//
//   strategy.rs  — DecisionStrategy trait + all strategies
//                  (FSM, BehaviorTree, Random, GOAP)
//   planner.rs   — PathPlanner trait + all path planners
//                  (A*, D* Lite, None)
//
// Adding a strategy: add a struct + impl in strategy.rs, one arm in registry.rs.
// Adding a planner:  add a struct + impl in planner.rs,  one arm in registry.rs.
//
// reinforcement_learning/ is a subfolder because it will grow into multiple
// files (policy, replay buffer, environment wrapper, trainer).

pub mod planner;
pub mod strategy;
pub mod reinforcement_learning;