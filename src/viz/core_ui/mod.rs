// src/viz/core_ui/mod.rs
//
// Widget helpers — button, panel, text spawners.
// Theme and color come from crate::style.

pub mod button;
pub mod panel;
pub mod text;

pub use button::{spawn_labeled_button, spawn_icon_button};
pub use panel::{spawn_floating_panel, spawn_button_group};
pub use text::{spawn_label, spawn_marked_label};