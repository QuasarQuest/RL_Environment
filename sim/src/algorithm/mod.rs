// src/algorithm/mod.rs
//
// Reusable AI-algorithm library. Only the GOAP planner remains in use (the
// viewer's GOAP showcase mode); the A*/D*-Lite path-planning and behaviour-tree/
// FSM modules were removed with the single-agent gold-rush refactor — engine
// navigation has its own A* in engine/nav.rs.
pub mod behavior;
