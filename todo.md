# TODO — after the June 2026 sim redesign

This session reworked the sim and removed `win`. The code builds, tests pass, and
the Rust↔Python obs contract is in sync (`OBS_TOTAL = 10231`, 14 crop + 5 minimap
channels, `ACTION_SIZE = 13`, `trace_fields = 12`). What's left is **retraining**
and **re-wiring the viewer model** — everything below is blocked on a fresh run
because the old checkpoints are incompatible.

---

## 1. Retrain (REQUIRED — blocks everything else)

Every checkpoint in `runs/seq_s*` was trained on the old **11770-dim / 16-channel**
observation. The new obs is **10231-dim / 14-channel**, so old models can't be
loaded, traced, or exported. You must retrain from scratch.

```bash
cd rl
# Full curriculum (stages 1→3, ~9M steps total; hot-starts each stage from the
# previous eval_best). Streams output; auto-plots per stage + combined at the end.
python scripts/train_sequential.py --name seq

# Quicker exploratory pass if you just want a watchable agent fast:
python scripts/train_sequential.py --name seq --timesteps 500000
```

Notes:
- Defaults are 2.5M / 3.0M / 3.5M steps for stages 1/2/3 (these are *option*
  decisions, not sim ticks — each is a full A* traversal).
- Algo is `maskable_ppo` (feedforward, non-recurrent). `normalize_obs: false`.

## 2. Wire the trained model into the viewer

The viewer loads `assets/model/policy.onnx`. It's currently **absent** (I removed
the stale old-arch one), so the viewer falls back to the scripted **BT/GOAP agent**.
`train.py` auto-exports an ONNX on every eval-best, so after a run just copy it in:

```bash
# After the sequential run finishes, take the stage-3 best policy:
cp runs/seq_s3_<id>/models/eval_best/policy.onnx assets/model/policy.onnx

# Sanity-check tract can load it (this is the exact check the viewer does):
cargo test -p viewer loads_exported_policy        # must print: ok

# Then run the viewer (loads the policy instead of BT):
cargo run --profile viewer-fast --package viewer --bin atb-view
```

If `loads_exported_policy` ever fails again, it's almost always an obs-layout
mismatch: the viewer forces a `[1, OBS_TOTAL]` input fact, so the ONNX must have
been exported from a build with the *same* `OBS_TOTAL`. Rebuild the `atb`
extension (`cd sim && maturin develop --release --features python`) and re-export.

## 3. Re-trace stage 3 and verify the redesign worked

Once a stage-3 model exists, regenerate the behaviour trace and confirm the goals
of this redesign actually landed:

```bash
cd rl
PYTHONPATH=src python -m monitoring.trace_recorder \
    --run ../runs/seq_s3_<id> --config assets/world/config_stage3.ron --episodes 8 --stochastic
PYTHONPATH=src python -m monitoring.trace_recorder summary ../runs/seq_s3_<id>/trace.h5
```

What to check in the new trace (vs the old run that lost ~half its episodes to traps):
- **Trapped time near zero.** Traps are now impassable terrain (3-tile clusters)
  the navigator routes around; the 250-tick immobilizations should be gone.
- **Multiplier use.** With the consumable-charge redesign, "grab charge → bank a
  load → 2× deposit" is a tight loop; expect NavMultiplier usage to rise and score
  per deposit to roughly double when a charge is held.
- **Score should be much higher / more stable** across episodes (the old run swung
  45–130 almost entirely on how many traps it hit).

Don't be alarmed if **NavSpeed stays rare** — a speed detour costs certain ticks
now for a diffuse future payoff, so ignoring speed boosts can be near-optimal under
the event-only reward. That's the expected ceiling of the options/A* design, not a
bug (see Track B below).

---

## Deferred / optional

- **Track B — primitive 8-direction action space.** The only way to get genuinely
  *emergent* step-level behaviour (dodging, weaving for buffs). Would replace the
  option/A* RL path with per-step moves (keep A* for the viewer only); the CNN obs
  already carries every channel needed. It's a fresh, longer training problem —
  spin it up as its own experiment, don't bolt it onto the working options path.
- **Perf (low priority): `find_clusters` recompute.** `SimCore::tick_once` rebuilds
  the 3×3 gold regions every tick. Caching per-option was intentionally NOT done —
  the spawner can add gold mid-option and per-tick recompute is what lets the agent
  divert to it. Only revisit if profiling shows it matters (A* dominates today).

## Reference
- Design rationale + "old checkpoints incompatible" note is in agent memory
  (`obs-layout-2026-06.md`).
- Channel layout single source of truth: `sim/src/rl/obs.rs` (mirror counts in
  `rl/src/network/extractor.py`).
