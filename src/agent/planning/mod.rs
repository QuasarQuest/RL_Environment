// src/agent/planning/mod.rs
//
// planning/ is flat — no behaviour_planning/ or path_planning/ subfolders.
// Each file is one concept; a subfolder with one file adds navigation cost
// with no grouping benefit.
//
// strategy.rs  — DecisionStrategy trait + FsmStrategy + BtStrategy + RandomStrategy
// planner.rs   — PathPlanner trait + AStarPlanner + DStarPlanner + NoPlanner
//
// reinforcement_learning/ stays as a subfolder because it will grow into
// multiple files (policy, replay buffer, environment wrapper, etc.)

pub mod planner;
pub mod strategy;
pub mod reinforcement_learning;