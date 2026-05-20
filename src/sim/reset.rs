// src/sim/reset.rs
//
// Shared episode reset — called by both the viz restart path and the RL
// headless path.  Takes &mut World directly; no Commands, no feature gates.

use bevy::prelude::*;
use crate::agent::brain::AgentBrain;
use crate::agent::spawn::spawn_agent_world;
use crate::item::Item;
use crate::item::spawner::{BandConfig, FreeTilePool, ItemSpawner};
use crate::sim::config::SimConfig;
use crate::team::TeamScore;
use crate::world::config::WorldConfig;
use crate::world::coords::GridPos;
use crate::world::Grid;
use crate::world::layout::{count_free_tiles, item_configs_for_free_count};
use crate::world::plugin::rebuild_grid;
use crate::world::tile::Tile;
use crate::config as global;

/// Full episode reset. Safe to call from any exclusive system.
///
/// What it does:
///   1. Reset SimConfig (tick, game_over, paused)
///   2. Reset TeamScore
///   3. Despawn all agents and items
///   4. Rebuild grid (tiles, safe zones, obstacles)
///   5. Rebuild FreeTilePool and ItemSpawner
///   6. Spawn initial items
///   7. Respawn agents
///   8. Update ResolvedLayout resource
pub fn reset_episode(world: &mut World) {
    // 1. Reset SimConfig
    {
        let mut sim   = world.resource_mut::<SimConfig>();
        sim.tick      = 0;
        sim.game_over = false;
        sim.paused    = false;
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

    // 5. Rebuild grid
    let mut layout = {
        let cfg      = world.resource::<WorldConfig>().clone();
        let mut grid = world.resource_mut::<Grid>();
        rebuild_grid(&cfg, &mut grid)
    };

    // 6. Rebuild FreeTilePool
    {
        let pool = FreeTilePool::build(world.resource::<Grid>());
        world.insert_resource(pool);
    }

    // 7. Compute item configs and rebuild ItemSpawner
    {
        let cfg      = world.resource::<WorldConfig>().clone();
        let grid     = world.resource::<Grid>().clone();
        let tile_vec: Vec<Tile> = grid.iter().map(|(_x, _y, t)| t).collect();
        let free     = count_free_tiles(&tile_vec);
        let item_cfgs = item_configs_for_free_count(&cfg, free);

        // Update ItemSpawner bands
        let bands = item_cfgs.iter().map(|c| {
            BandConfig::new(c.kind, (c.max_on_map / 2).max(1), c.max_on_map, 300)
        }).collect();
        world.insert_resource(ItemSpawner { bands });

        // Store on layout so ResolvedLayout resource is consistent
        layout.item_configs = item_cfgs;
    }

    // 8. Update ResolvedLayout resource
    world.insert_resource(layout.clone());

    // 9. Spawn initial items directly (no Commands available here)
    {
        let cfg  = world.resource::<WorldConfig>().clone();
        let grid = world.resource::<Grid>().clone();

        for item_cfg in &layout.item_configs {
            let initial      = (item_cfg.max_on_map / 2).max(1);
            let mut placed   = 0usize;
            let mut attempts = 0usize;

            while placed < initial && attempts < initial * 200 {
                attempts += 1;
                let x = rand::random_range(0..cfg.width  as i32);
                let y = rand::random_range(0..cfg.height as i32);
                if grid.get(x, y) == Some(Tile::Free) {
                    world.spawn((
                        Item { kind: item_cfg.kind },
                        GridPos::new(x, y),
                        Sprite {
                            color:       item_cfg.kind.color(),
                            custom_size: Some(Vec2::splat(global::TILE_SIZE * 0.6)),
                            ..default()
                        },
                        Transform::from_xyz(0.0, 0.0, item_cfg.kind.z_layer()),
                        Visibility::default(),
                    ));
                    placed += 1;
                }
            }
        }
    }

    // 10. Respawn agents
    for (idx, agent) in layout.agents.iter().enumerate() {
        spawn_agent_world(world, agent, &layout, idx);
    }
}