// src/viz/restart.rs
//
// In-process episode restart for RL training.
//
// Obstacles are regenerated randomly each episode — this is intentional.
// The agent must learn a general navigation policy, not memorise one map.
//
// Flow:
//   RESTART button in end_screen → write RestartMessage
//   → restart_episode (exclusive system, runs in Update after HudUpdate)
//
// assign_display_components must run in Update with Added<AgentBrain>
// so new agents get their labels after respawn. See factory/mod.rs.

use bevy::prelude::*;
use crate::agent::brain::AgentBrain;
use crate::agent::components::{
    Ammo, DeathCount, GoldCarried, GridPos, Hearts, KillCount, Score, SpawnPoint,
};
use crate::agent::systems::PendingAction;
use crate::agent::registry::make_agent;
use crate::factory::AgentConfigIndex;
use crate::item::Item;
use crate::item::spawner::{FreeTilePool, ItemSpawner};
use crate::sim::config::SimConfig;
use crate::team::{Team, TeamScore};
use crate::world::config::WorldConfig;
use crate::world::Grid;
use crate::world::plugin::{apply_fixed_tiles, regenerate_obstacles};
use crate::viz::end_screen::EndScreen;
use crate::viz::hud::components::ScoreboardRow;
use crate::config;

// ── Message ───────────────────────────────────────────────────────────────────

#[derive(Message, Clone)]
pub struct RestartMessage;

// ── Exclusive restart system ──────────────────────────────────────────────────

pub fn restart_episode(world: &mut World) {
    // Drain RestartMessage — only act if one was written this frame.
    let has_msg = {
        let mut msgs = world.resource_mut::<Messages<RestartMessage>>();
        let any = msgs.drain().next().is_some();
        any
    };
    if !has_msg { return; }

    info!("=== Episode restart ===");

    // 1. Reset SimConfig
    {
        let mut cfg = world.resource_mut::<SimConfig>();
        cfg.tick      = 0;
        cfg.game_over = false;
        cfg.paused    = false;
    }

    // 2. Reset TeamScore
    *world.resource_mut::<TeamScore>() = TeamScore::default();

    // 3. Despawn all agents
    let agents: Vec<Entity> = world
        .query_filtered::<Entity, With<AgentBrain>>()
        .iter(world)
        .collect();
    for e in agents { world.despawn(e); }

    // 4. Despawn all items
    let items: Vec<Entity> = world
        .query_filtered::<Entity, With<Item>>()
        .iter(world)
        .collect();
    for e in items { world.despawn(e); }

    // 5. Despawn scoreboard rows — build_scoreboard_rows rebuilds next frame
    let rows: Vec<Entity> = world
        .query_filtered::<Entity, With<ScoreboardRow>>()
        .iter(world)
        .collect();
    for e in rows { world.despawn(e); }

    // 6. Reset grid: clear all, apply fixed tiles, regenerate random obstacles
    {
        let map = world.resource::<WorldConfig>().clone();
        let mut grid = world.resource_mut::<Grid>();
        apply_fixed_tiles(&map, &mut grid);
        regenerate_obstacles(&map, &mut grid);
    }

    // 7. Rebuild FreeTilePool from new grid
    {
        let pool = FreeTilePool::build(world.resource::<Grid>());
        world.insert_resource(pool);
    }

    // 8. Reset ItemSpawner PI controller state
    {
        let map = world.resource::<WorldConfig>().clone();
        world.insert_resource(ItemSpawner::from_map_config(&map));
    }

    // 9. Respawn agents — factory system picks up Added<AgentBrain> next frame
    {
        let map = world.resource::<WorldConfig>().clone();
        for (idx, cfg) in map.agents.iter().enumerate() {
            let team  = Team(cfg.team.unwrap_or(0) as u8);
            let brain = AgentBrain(make_agent(cfg));
            let color = team.color();
            let pos   = GridPos::new(cfg.x, cfg.y);

            world.spawn((
                pos,
                SpawnPoint(pos),
                Hearts::default(),
                Ammo::default(),
                GoldCarried::default(),
                Score::default(),
                KillCount::default(),
                DeathCount::default(),
                brain,
                team,
                PendingAction::default(),
                Sprite {
                    color,
                    custom_size: Some(Vec2::splat(config::TILE_SIZE * 0.8)),
                    ..default()
                },
                Transform::from_xyz(0.0, 0.0, 1.0),
                Visibility::default(),
                AgentConfigIndex(idx),
            ));
        }
    }

    // 10. Hide EndScreen + clear stat card children
    let end_screens: Vec<Entity> = world
        .query_filtered::<Entity, With<EndScreen>>()
        .iter(world)
        .collect();
    for e in end_screens {
        if let Some(mut node) = world.get_mut::<Node>(e) {
            node.display = Display::None;
        }
        if let Some(mut vis) = world.get_mut::<Visibility>(e) {
            *vis = Visibility::Hidden;
        }
    }

    info!("Episode restarted — tick 0, new obstacles generated.");
}