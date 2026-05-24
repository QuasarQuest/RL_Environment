use bevy::prelude::*;

const TEAM_NAMES: &[&str] = &["Red", "Blue", "Green", "Yellow"];

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Team(pub u8);

impl Team {
    pub fn name(self) -> &'static str {
        TEAM_NAMES.get(self.0 as usize).copied().unwrap_or("Unknown")
    }
}
