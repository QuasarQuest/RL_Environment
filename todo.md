# TODO

## Code-review fixes (July 2026) — work top to bottom

Deep-review findings, ordered by severity.
Each item is checked off only after the fix builds, passes tests/lints, and is verified.

### Phase A — training correctness

- [x] A1. `rl/scripts/resume_latest.py`: resume step accounting is inverted.
  SB3 with `reset_num_timesteps=False` treats `total_timesteps` as an additional budget (`total_timesteps += num_timesteps`), so `cur + add` trains ~`cur + add` EXTRA steps.
  Pass the additional budget straight through; delete `current_steps()` and the wrong warning.
- [x] A2. `rl/src/training/smdp.py`: timeout value bootstrap uses γ¹ instead of γ^k.
  SB3 adds `gamma * V(terminal_obs)` to the reward on truncation; correct it to `gamma**k` in `SmdpDiscountCallback._on_step` by subtracting `(γ − γ^k)·V(terminal_obs)` for truncated envs.
- [x] A3. Seeding is dead end-to-end.
  Expose `seed` on `PyBatchEnv::new` → `SimCore::new_with_seed(seed + i)`; implement `BatchVecEnv.seed`/constructor seed; wire `EnvConfig.seed` through `factory.py`; rebuild the `atb` extension.
- [x] A4. `rl/src/training/callbacks/entropy.py`: progress clock breaks on resume.
  Use `model._current_progress_remaining` instead of `num_timesteps / total`.
- [x] A5. `rl/src/training/tune.py`: trials do not match the training pipeline.
  Add the SMDP rollout buffer + `SmdpDiscountCallback`; pin `gamma` to the RON `option_gamma` instead of searching it; include the production `batch_size`/`n_epochs` values in the grids.
- [x] A6. `rl/src/env/factory.py`: `max_episode_steps` is silently ignored when `n_envs > 1`; fail loudly instead.
  Also pass `gamma=ppo.gamma` to `VecNormalize` explicitly.

### Phase B — viewer/sim option semantics (ONNX mode must mirror step_option)

- [x] B1. `sim/src/engine/mod.rs`: expose `pub fn en_route(&self) -> bool`.
  `viewer/src/sim_bridge.rs`: replace the `onnx_moved` position-changed heuristic (re-queries every cadence rest tick) with `en_route()`.
- [x] B2. `viewer/src/sim_bridge.rs`: option termination misses buff pickups.
  Anchor `speed_buff`/`mult_charge` beside gold/score and end the option on an increase, mirroring `step_option`.
- [x] B3. `sim/src/engine/mod.rs`: `SimCore::step` clears committed nav goals every tick, so the viewer path loses the commit-lock and runs a BFS per tick.
  Add a step variant that holds commitments while an option stays committed; clear only at option boundaries in the viewer.
- [x] B4. `viewer/src/viz/hud/systems.rs`: mode cycling can select ONNX with no policy loaded (agent freezes on Wait).
  Skip `PolicyMode::Onnx` when no policy is present.

### Phase C — minor correctness (Python)

- [x] C1. `rl/src/monitoring/trace_recorder.py`: `--algo ppo` crashes — `ActorCriticPolicy.get_distribution()` takes no `action_masks` kwarg; pass masks only to maskable policies.
- [x] C2. `rl/src/training/train.py`: flush/close the stats writer in the `finally` block so Ctrl-C does not lose up to `flush_every − 1` episode rows.
- [x] C3. `rl/src/training/callbacks/plotting.py`: close the previous plot-log file handle before opening a new one (fd leak).
- [x] C4. `rl/scripts/plot_runs.py`: read ALL tfevents files per run (resumed history is dropped today); make the run-dir sort key robust to non-numeric suffixes.
- [x] C5. `rl/src/training/train.py`: on resume, the printed hyperparameter table shows yaml values the checkpoint load overrides; print the model's actual loaded values.
- [x] C6. `rl/pyproject.toml`: add missing runtime deps (hydra-core, omegaconf, python-dotenv, matplotlib, tensorboard, psutil); drop unused (plotext, raylib).
- [x] C7. `rl/src/training/callbacks/checkpoint.py`: run ONNX export only on eval-best (not every 50k), and fail loudly after repeated export errors instead of `except: print`.
- [x] C8. `rl/src/training/callbacks/stats.py`: `steps_per_sec` mixes cumulative steps with post-resume elapsed time; measure against steps since callback start.

### Phase D — viewer correctness + perf

- [x] D1. `viewer/src/viz/plugin.rs`: add `.after(SimSet)` to the world-sync systems (tiles, agents, items, pocket overlay, tooltip, scoreboard stats, debug overlay) so the rendered world cannot lag the HUD.
- [x] D2. Load/restart edges: despawn scoreboard rows on config load; refit the camera after a load with different grid dimensions; replace `std::process::exit(0)` in `end_screen.rs` with an `AppExit` event.
- [x] D3. Per-frame waste: repaint tile colors only on restart/load (not every tick); write `Node`/`display` state only on transitions (scoreboard tab, tooltip, end screen); skip text rebuilds for hidden panels; drop the double obs copy in ONNX inference (`sim_bridge.rs` + `policy.rs`).

### Phase E — sim perf

- [x] E1. `sim/src/engine/mod.rs`: compute `find_clusters` lazily (only when `resolve_nav_goal` actually needs it), preserving the mid-option re-resolve semantics; reuse one `dist_field` per option boundary across mask/telemetry/goal/obs instead of up to four BFS passes.

### Phase G — tooling (user request 2026-07-17)

- [x] G1. Add Ruff + Pyright linting for `rl/`: config in `rl/pyproject.toml`, a repo-root `checker.bash` running the full check suite (ruff, pyright, cargo clippy/test), and a git pre-commit hook that runs it.
  Mirror the setup used in the other workspace projects.
- [x] G2. GitLab CI (`.gitlab-ci.yml`): ruff lint job, clippy `-D warnings` + cargo test job (Bevy system deps installed), and a pyright + end-to-end smoke job (`rl/scripts/ci_smoke.py`: FFI contract, seeded determinism, short CPU training) building the atb extension against the job image's CPython.

### Phase F — hygiene

- [x] F1. Clippy: fix all 5 warnings in `sim` and 10 in `viewer`; keep both crates warning-free.
- [x] F2. Dead code: delete viewer orphans (`viz/components.rs`, `viz/core_ui/text.rs`, `viz/world/mod.rs` stub, `TeamScoreMarker` + `update_team_scores`), `trace_recorder.py` `dec_idx`, unused `AtbMlpExtractor`/`ATB_MLP_POLICY_KWARGS`/`AtbPolicy`, GOAP `ACT_WAIT` (identity effect — unreachable in any plan).
- [x] F3. Stale docs/comments: `configs/ppo/default.yaml` + `algos.py` (`OBS_TOTAL=10222`, GTX 1650 Ti), `fast_iter.yaml` hardware note, `debug_gizmos.rs`/`help_overlay.rs` gizmo claims, `sim_bridge.rs` local `ACTION_NAVIGATE_TO_BASE` → use `ACTION_BASE`.
- [x] F4. `sim/src/rl/env.rs` `get_items`: replace the 0/1/4 magic item codes with a shared named encoding on both ends of the FFI.
- [x] F5. `viewer/src/viz/panels/load_menu.rs`: rescan config/policy directories when the menu opens instead of once at startup.

---

## Retrain (after the fixes above)

Checkpoints in `runs/` predate the current obs contract (`OBS_TOTAL = 8692`: 12-channel 25×25 crop + 4-channel 17×17 minimap + 36 cluster floats, `ACTION_SIZE = 13`).
Old models cannot be loaded, traced, or exported — retrain from scratch:

```bash
cd rl
python scripts/train_sequential.py --name seq              # full curriculum, stages 1→3
python scripts/train_sequential.py --name seq --timesteps 500000   # quick watchable agent
```

Then wire the model into the viewer:

```bash
cp runs/seq_s3_<id>/models/eval_best/policy.onnx assets/model/policy.onnx
cargo test -p viewer loads_exported_policy
cargo run --profile viewer-fast --package viewer --bin atb-view
```

And re-trace stage 3 to verify multiplier use and score stability:

```bash
cd rl
PYTHONPATH=src python -m monitoring.trace_recorder \
    --run ../runs/seq_s3_<id> --config assets/world/config_stage3.ron --episodes 8 --stochastic
PYTHONPATH=src python -m monitoring.trace_recorder summary ../runs/seq_s3_<id>/trace.h5
```

## Deferred / optional

- Track B — primitive 8-direction action space for genuinely emergent step-level behaviour; a separate experiment, not a bolt-on.

## Reference

- Channel layout single source of truth: `sim/src/rl/obs.rs` (mirror counts in `rl/src/network/extractor.py`).
- Design rationale and "old checkpoints incompatible" notes live in agent memory (`obs-layout-2026-06.md`, `cluster-encoding-redesign.md`).
