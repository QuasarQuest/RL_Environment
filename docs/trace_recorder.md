# `atb-trace` — behaviour trace recorder

Records, for a trained checkpoint, **exactly what the agent saw and chose** at every
high-level decision and every sim tick — so reward shaping can be tuned against
evidence instead of guesswork.

It exists to debug the near-vs-dense gold tradeoff: the agent over-uses
`NavigateToNearestGold` (action 12) when we'd rather it prefer dense clusters unless
gold is genuinely close. The intra-option SMDP discount (`reward.option_gamma`) is
meant to balance this, but its effect is invisible — this tool makes it visible.

## Install / build

The recorder ships with the `atb-rl` package and needs the compiled `atb` extension.

```bash
# Rebuild the Rust extension (only needed after changing sim/ Rust code):
cd sim && maturin develop --release --features python

# Register the console script (only needed after pulling new code):
cd rl && pip install -e .
```

## Usage

```bash
# Record (default command): loads <run>/eval_best deterministically, runs N episodes.
atb-trace --run runs/seq_s1_<id> --episodes 5          # → writes <run>/trace.h5

# Human-readable summary of a recorded file:
atb-trace summary runs/seq_s1_<id>/trace.h5
```

### `record` options

| Flag | Default | Meaning |
|------|---------|---------|
| `--run <dir>` | – | Run dir; derives model (`eval_best/best_model.zip`), vecnorm (`eval_best_vecnorm.pkl`) and output (`<run>/trace.h5`). |
| `--model <path>` | from `--run` | Explicit model `.zip` (overrides `--run`). |
| `--vecnorm <path>` | from `--run` | Explicit `VecNormalize` `.pkl`. Obs-norm is applied only if the run trained with `normalize_obs=true` (otherwise a no-op). |
| `--config <path>` | `config_stage1.ron` | World `.ron` config. |
| `--out <path>` | `<run>/trace.h5` | Output HDF5 file. |
| `--episodes <n>` | `5` | Episodes to record. |
| `--algo <name>` | `maskable_ppo` | `ppo` or `maskable_ppo`. |
| `--deterministic / --stochastic` | deterministic | argmax vs sample from the masked policy. Top-5 probs are logged either way. |

> **Note:** episode maps come from the Rust `SysRng` (no per-episode seed is exposed),
> so world layouts are *not* reproducible run-to-run. Action selection is deterministic;
> the map is not.

## Granularity (why it's faithful)

A navigation option runs many ticks via A*, but **ends the moment a pickup/deposit
fires** (those change `gold_carried`/`score`, a stop condition in
`SimCore::step_option`). So within one option the gold/item map is essentially static:

- **Full positions** (gold/items/obstacles/base) are snapshotted **per decision**.
- **Per tick** we record only what changes: agent position + the reward breakdown.

## Output: `trace.h5` layout

**Root attrs** (self-describing): `model_path`, `config_path`, `episodes`,
`deterministic`, `grid_w/grid_h`, `cluster_k`, `reward_tick/pickup/deposit/wall_hit`,
`option_gamma`, `action_names`, `tick_field_names`.

**Per episode** — group `episode_NNN`:
- attrs: `final_score`, `length_ticks`, `n_decisions`, `total_reward`, `base_x/base_y`
- `map_tiles` — `(H, W)` uint8 (0 free, 1 obstacle, 10+team base, 20+team safezone)

- `decisions/` — parallel arrays, length = number of decisions:
  - scalars: `tick`, `agent_x`, `agent_y`, `gold_carried`, `score`, `value` (critic
    estimate), `chosen_action`, `option_ticks`, `option_reward` (undiscounted Σ),
    `option_return` (Σ γᵗr), `is_cluster`, `own_has_gold`, `skipped_own`, `chosen_dist`
  - `action_probs (n, 14)`, `action_mask (n, 14)`, `top5_actions (n, 5)`, `top5_probs (n, 5)`
  - cluster features, one value per fixed 3×3 region: `cluster_count`, `cluster_dx`,
    `cluster_dy`, `cluster_pathdist` — each `(n, 9)`. **This is the near-vs-dense signal**:
    `count` = how much gold a region holds, `pathdist` = how far it is (BFS, normalised).
  - ragged positions (CSR-style: concat + offsets):
    `gold_xy (total, 2)` + `gold_offsets (n+1,)`; `item_xyk (total, 3)` + `item_offsets (n+1,)`

- `ticks/` — parallel arrays, length = total episode ticks:
  - `decision_idx` (which decision this tick belongs to) plus the 12 `TickRecord`
    columns: `tick, ax, ay, gold_carried, score, r_tick, r_pickup, r_deposit, r_wall,
    r_total, discount, gold_count`

### Reading it

```python
import h5py, numpy as np

with h5py.File("runs/seq_s1_<id>/trace.h5", "r") as f:
    names = [n.decode() for n in f.attrs["action_names"]]
    d = f["episode_000/decisions"]
    acts  = d["chosen_action"][:]            # (n,)
    count = d["cluster_count"][:]            # (n, 9)  gold per region
    pdist = d["cluster_pathdist"][:]         # (n, 9)  distance per region

    # gold positions at decision i:
    off = d["gold_offsets"][:]
    i = 0
    gold_i = d["gold_xy"][off[i]:off[i+1]]   # (g, 2)
```

## `summary` — turns the trace into the answer

```
atb-trace summary runs/seq_s1_<id>/trace.h5 [--dist-margin 0.10]
```

Prints, across all episodes:
- **Action frequency** — how dominant `NavNearestGold` actually is.
- **Chosen-cluster stats** — mean `count` / `pathdist` of regions the agent committed to.
- **The key tuning signal** — fraction of `NavNearestGold` decisions where a *denser*
  region existed at meaningfully larger distance (`> dist-margin`), i.e. the agent took
  close-but-sparse gold over far-but-dense.

`--dist-margin` (default `0.10`) sets how much farther (in normalised path distance) a
region must be to count as "farther".

Example:

```
=== trace.h5 — 183 decisions over 3 episodes ===

Action frequency:
  NavBase              29   15.8%
  NavNearestGold      154   84.2%

NavNearestGold decisions: 154
  ...where a DENSER region existed >0.10 farther: 132 (85.7%)
```

A high percentage ⇒ the agent trades dense-far for sparse-near. Raise `reward.pickup`
or `option_gamma` (in the stage `.ron` config) and re-run record + summary to watch it drop.

## Related code

- Recorder: `rl/src/monitoring/trace_recorder.py`
- Trace capture: `sim/src/engine/mod.rs` (`TickRecord`, `step_option`, `set_trace`)
- Reward components: `sim/src/rl/reward.rs` (`RewardBreakdown`, `compute_components`)
- FFI surface: `sim/src/rl/pyo3.rs` (`set_trace`, `get_trace`, `reward_weights`, `trace_fields`)
