use bevy::prelude::*;

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Team(pub u8);

impl Team {
    pub fn name(self) -> &'static str {
        match self.0 {
            0 => "Red",
            1 => "Blue",
            2 => "Green",
            3 => "Yellow",
            _ => "Unknown",
        }
    }

}
