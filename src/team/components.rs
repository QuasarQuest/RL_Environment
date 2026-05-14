// src/team/components.rs

use bevy::prelude::*;
use std::collections::HashMap;
use crate::style::color::team_color;

// ── Team component ────────────────────────────────────────────────────────────

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

    pub fn color(self) -> Color {
        team_color(self.0)
    }
}

// ── TeamScore resource ────────────────────────────────────────────────────────

#[derive(Resource, Default)]
pub struct TeamScore {
    scores: HashMap<u8, u32>,
}

impl TeamScore {
    pub fn add(&mut self, team: Team, amount: u32) {
        *self.scores.entry(team.0).or_insert(0) += amount;
    }

    pub fn get(&self, team: Team) -> u32 {
        *self.scores.get(&team.0).unwrap_or(&0)
    }
}