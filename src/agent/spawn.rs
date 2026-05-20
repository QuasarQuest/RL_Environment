// src/agent/spawn.rs
//
// Agent spawning — reads from ResolvedLayout (computed positions).
//
// Two entry points:
//   spawn_agents        — Bevy Startup system, takes Commands
//   spawn_agent_world   — called from exclusive systems (restart, RL env),
//                         takes &mut World directly

use bevy::prelude::*;
use crate::config;
use crate::factory::AgentConfigIndex;
use crate::team::Team;
use crate::world::layout::{ResolvedAgent, ResolvedLayout};
use super::brain::AgentBrain;
use super::components::{
    Ammo, DeathCount, GoldCarried, GridPos, Hearts, KillCount, Score, SpawnPoint,
};
use super::systems::PendingAction;
use super::registry::make_agent;

#[cfg(feature = "python")]
use crate::rl::marker::RlAgent;

// ── AgentBundle ───────────────────────────────────────────────────────────────

#[derive(Bundle)]
pub struct AgentBundle {
    pub pos:         GridPos,
    pub spawn_point: SpawnPoint,
    pub hearts:      Hearts,
    pub ammo:        Ammo,
    pub gold:        GoldCarried,
    pub score:       Score,
    pub kills:       KillCount,
    pub deaths:      DeathCount,
    pub brain:       AgentBrain,
    pub team:        Team,
    pub pending:     PendingAction,
    pub sprite:      Sprite,
    pub transform:   Transform,
    pub visibility:  Visibility,
}

impl AgentBundle {
    pub fn from_resolved(agent: &ResolvedAgent, base_pos: GridPos, brain: AgentBrain) -> Self {
        let team  = Team(agent.team);
        let color = team.color();
        Self {
            pos:         GridPos::new(agent.x, agent.y),
            spawn_point: SpawnPoint(base_pos),
            hearts:      Hearts::default(),
            ammo:        Ammo::default(),
            gold:        GoldCarried::default(),
            score:       Score::default(),
            kills:       KillCount::default(),
            deaths:      DeathCount::default(),
            brain,
            team,
            pending:     PendingAction::default(),
            sprite: Sprite {
                color,
                custom_size: Some(Vec2::splat(config::TILE_SIZE * 0.8)),
                ..default()
            },
            transform:  Transform::from_xyz(0.0, 0.0, 1.0),
            visibility: Visibility::default(),
        }
    }
}

// ── Startup system (Commands path) ────────────────────────────────────────────

pub fn spawn_agents(mut commands: Commands, layout: Res<ResolvedLayout>) {
    for (idx, agent) in layout.agents.iter().enumerate() {
        let bundle = build_bundle(agent, &layout);
        spawn_with_commands(&mut commands, bundle, agent, idx);
    }
}

// ── Exclusive system path (World) ─────────────────────────────────────────────

/// Called from exclusive systems (viz/restart.rs, rl/env.rs) that hold
/// `&mut World` directly and cannot use Commands.
pub fn spawn_agent_world(world: &mut World, agent: &ResolvedAgent, layout: &ResolvedLayout, idx: usize) {
    let bundle = build_bundle(agent, layout);

    #[cfg(not(feature = "python"))]
    {
        world.spawn((bundle, AgentConfigIndex(idx)));
    }

    #[cfg(feature = "python")]
    {
        let mut entity = world.spawn((bundle, AgentConfigIndex(idx)));
        if agent.team == 1 {
            entity.insert(RlAgent);
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn build_bundle(agent: &ResolvedAgent, layout: &ResolvedLayout) -> AgentBundle {
    let base_pos = layout.bases.iter()
        .find(|b| b.team == agent.team)
        .map(|b| GridPos::new(b.x, b.y))
        .unwrap_or_else(|| GridPos::new(agent.x, agent.y));

    let cfg   = minimal_agent_cfg(agent);
    let brain = AgentBrain(make_agent(&cfg));
    AgentBundle::from_resolved(agent, base_pos, brain)
}

fn spawn_with_commands(
    commands: &mut Commands,
    bundle:   AgentBundle,
    agent:    &ResolvedAgent,
    idx:      usize,
) {
    #[cfg(not(feature = "python"))]
    commands.spawn((bundle, AgentConfigIndex(idx)));

    #[cfg(feature = "python")]
    {
        let mut entity = commands.spawn((bundle, AgentConfigIndex(idx)));
        if agent.team == 1 {
            entity.insert(RlAgent);
        }
    }
}

fn minimal_agent_cfg(agent: &ResolvedAgent) -> crate::world::config::AgentConfig {
    crate::world::config::AgentConfig {
        team:         agent.team,
        strategy:     agent.strategy,
        planner:      agent.planner,
        spawn:        crate::world::config::SpawnIntent::NearBase,
        spawn_offset: 0.0,
    }
}