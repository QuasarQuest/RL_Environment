# TODO

## SimCore refactor + Rayon training + Bevy GUI

Goal: 2-3x overall training speedup by eliminating SubprocVecEnv IPC overhead,
and a visual playback mode where the trained ONNX policy drives the Bevy game.

### Why

Current bottleneck is 64 pipe round-trips per step (SubprocVecEnv + spawn).
GPU sits at ~18% utilization. Extracting the sim from Bevy and batching steps
via Rayon eliminates all IPC — one Python call steps all envs in parallel inside Rust.

Expected: ~4x env throughput → ~2.5x overall training speed (5M run: 42min → ~17min).

---

### 1. Extract SimCore from Bevy

Create `src/sim_core.rs` — a plain `Send + Sync` Rust struct with all game logic,
no Bevy dependency.

```
SimCore {
    grid:   Grid,
    agent:  AgentState,   // pos, gold_carried, hearts
    items:  Vec<ItemState>,
    tick:   u32,
    rng:    SmallRng,
}

impl SimCore {
    fn new(config: &WorldConfig) -> Self
    fn reset(&mut self)
    fn step(&mut self, action: u32) -> (Vec<f32>, f32, bool)
    fn obs(&self) -> Vec<f32>   // same 5×25×25 CNN grid as training
}
```

The logic already exists in:
- `src/rl/env.rs`       — step / reset loop
- `src/rl/obs.rs`       — CNN obs builder
- `src/rl/reward.rs`    — reward computation
- `src/agent/action.rs` — action mapping
- `src/world/`          — grid, tiles, layout

Port the logic, remove Bevy types. `unsafe impl Send for SimCore {}` is safe
because each instance is owned by exactly one thread and never shared.

---

### 2. Rayon batch env (replaces SubprocVecEnv)

Create `src/rl/batch_env.rs`:

```rust
pub struct BatchEnv {
    envs: Vec<SimCore>,
}

impl BatchEnv {
    pub fn step_batch(&mut self, actions: Vec<u32>) -> Vec<(Vec<f32>, f32, bool)> {
        self.envs.par_iter_mut()
            .zip(actions.par_iter())
            .map(|(env, &a)| env.step(a))
            .collect()
    }

    pub fn reset_batch(&mut self) -> Vec<Vec<f32>> {
        self.envs.par_iter_mut().map(|e| e.reset()).collect()
    }
}
```

Add `rayon` to Cargo.toml under the `python` feature.

Expose as `PyBatchEnv` via PyO3. On the Python side, replace `SubprocVecEnv`
with a thin `BatchVecEnv` wrapper that calls `step_batch` / `reset_batch` —
compatible with SB3's `VecEnv` interface.

---

### 3. Bevy GUI wired to SimCore

For visual playback of a trained ONNX policy:

- Add `ort` crate (ONNX Runtime) under an `onnx` feature flag in Cargo.toml.
- Insert `SimCore` as a Bevy `Resource` (the visual game reads state from it).
- Add a Bevy system `onnx_inference_system` that each tick:
  1. Calls `sim_core.obs()` → CNN tensor
  2. Runs `ort::Session` inference → logits
  3. Argmax → action → `sim_core.step(action)`
- Existing Bevy rendering systems read agent/item positions from `SimCore`
  instead of from ECS components directly.
- Wire via `config.ron`: `strategy: Rl, model: "runs/models/final.onnx"`.

No Python process needed at runtime — pure Rust inference inside Bevy.

---

### 4. Python-side VecEnv adapter

Replace `SubprocVecEnv` in `env/factory.py` with a `BatchVecEnv` class that
wraps `PyBatchEnv` and implements SB3's `VecEnv` interface
(`step_async` / `step_wait` / `reset` / `observation_space` / `action_space`).

Training code in `train.py` stays unchanged — `build_vec_env()` returns the
new wrapper transparently.

---

### Order of work

1. SimCore (prerequisite for everything else)
2. Rayon batch env + Python adapter (training speedup)
3. Bevy GUI + ONNX inference (visualization)

Do after reward shaping is confirmed working — no point optimizing the
pipeline before the policy converges.


CHANGE THE GITIGNORE IT INGNORES env